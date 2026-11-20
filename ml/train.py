"""Train/evaluate binary detector and selectable models for five attack patterns."""
from datetime import datetime
from pathlib import Path
import json
import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, average_precision_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "ml" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
FEATURES = ["event_type", "protocol", "dst_port", "failed_login_count", "bytes_sent", "severity_hint"]
CATEGORICAL = ["event_type", "protocol", "severity_hint"]
NUMERIC = ["dst_port", "failed_login_count", "bytes_sent"]
PATTERNS = ["brute_force", "port_scan_like", "unusual_transfer", "suspicious_privilege", "malware_like"]
MODEL_NAMES = ["Random Forest", "XGBoost"]

def _dump(obj, path):
    """Save atomically: the web app may be scoring an event with the old file while a retrain is writing the new one."""
    tmp = Path(str(path) + ".tmp")
    joblib.dump(obj, tmp)
    _replace(tmp, path)


def _replace(tmp, path):
    """os.replace with a short retry: on Windows it fails for a moment if another thread is reading the target."""
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25)


def _model_file(name):
    return name.lower().replace(" ", "_")


def _preprocessor():
    return ColumnTransformer([
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), NUMERIC)
    ])

def _estimator(name):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    raise ValueError(f"Unsupported model: {name}")

def _curve_points(x, y, limit=120):
    if len(x) <= limit:
        indexes = range(len(x))
    else:
        indexes = sorted(set(int(i) for i in np.linspace(0, len(x)-1, limit)))
    return [{"x": round(float(x[i]), 5), "y": round(float(y[i]), 5)} for i in indexes]

def _parameter_importance(pipe):
    """Share of the model's decisions that each INPUT PARAMETER explains (one-hot columns are summed back)."""
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    values = pipe.named_steps["classifier"].feature_importances_
    totals = {f: 0.0 for f in FEATURES}
    for name, value in zip(names, values):
        raw = name.split("__", 1)[1]
        for f in FEATURES:
            if raw == f or raw.startswith(f + "_"):
                totals[f] += float(value)
                break
    total = sum(totals.values()) or 1.0
    return [{"parameter": f, "importance": round(v / total, 4)}
            for f, v in sorted(totals.items(), key=lambda kv: -kv[1])]

def _hyperparameters(pipe):
    est = pipe.named_steps["classifier"]
    keys = (["n_estimators", "max_depth", "class_weight", "random_state"] if isinstance(est, RandomForestClassifier)
            else ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "random_state"])
    params = est.get_params()
    return {k: ("None" if params.get(k) is None else params.get(k)) for k in keys}

def _evaluate(name, pipe, X_test, y_test, target_names=("Negative", "Positive")):
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, prob)
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "roc_curve": _curve_points(fpr, tpr),
        "pr_curve": _curve_points(recall_curve[::-1], precision_curve[::-1]),
        "average_precision": round(float(average_precision_score(y_test, prob)), 4),
        "parameter_importance": _parameter_importance(pipe),
        "hyperparameters": _hyperparameters(pipe),
        "test_points": [
            {"dst_port": int(row.dst_port), "failed_login_count": int(row.failed_login_count),
             "bytes_sent": round(float(row.bytes_sent), 1), "probability": round(float(p), 4),
             "actual": int(a), "predicted": int(q)}
            for (_, row), p, a, q in zip(X_test.iterrows(), prob, y_test, pred)
        ],
        "classification_report": classification_report(y_test, pred, labels=[0,1], target_names=list(target_names), zero_division=0, output_dict=True)
    }

def _provisional_part(provisional):
    """Events that have no ground truth yet but were scored by the models: their PREDICTED label is used as a
    provisional label. They are added to the TRAINING data only - never to the held-out test split - so every
    reported metric is still computed on real ground truth."""
    if provisional is None or len(provisional) == 0:
        return None
    prov = provisional.copy()
    prov["label"] = prov["label"].where(prov["label"].isin(["Benign", "Malicious"]), "Benign")
    prov["attack_type"] = prov["attack_type"].fillna("benign").astype(str).str.strip().str.lower()
    prov.loc[prov["label"] == "Benign", "attack_type"] = "benign"
    return prov.reset_index(drop=True)


def train_from_csv(csv_path=None, df=None, dataset_name=None, provisional=None):
    """Train from a CSV file, or from a DataFrame (e.g. every labelled event in the database).

    `provisional` = optional DataFrame (FEATURES + label + attack_type) of new events with predicted labels."""
    if df is None:
        path = Path(csv_path) if csv_path else ROOT / "data" / "security_events.csv"
        if not path.exists():
            raise FileNotFoundError(f"Training dataset not found: {path}")
        df = pd.read_csv(path)
        dataset_name = dataset_name or path.name
    df = df.copy()
    missing = set(FEATURES + ["label", "attack_type"]) - set(df.columns)
    if missing:
        raise ValueError("Training CSV is missing required columns: " + ", ".join(sorted(missing)))
    df = df[df.label.isin(["Benign", "Malicious"])].copy()
    df["attack_type"] = df["attack_type"].fillna("benign").astype(str).str.strip().str.lower()
    if len(df) < 40 or df.label.nunique() != 2:
        raise ValueError("Need at least 40 records with both Benign and Malicious labels.")
    X = df[FEATURES]
    y = (df.label == "Malicious").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    prov = _provisional_part(provisional)
    if prov is not None:
        X_train = pd.concat([X_train, prov[FEATURES]], ignore_index=True)
        y_train = pd.concat([y_train, (prov["label"] == "Malicious").astype(int)], ignore_index=True)
    binary_results, binary_models = [], {}
    for name in MODEL_NAMES:
        pipe = Pipeline([("preprocessor", _preprocessor()), ("classifier", _estimator(name))])
        pipe.fit(X_train, y_train)
        result = _evaluate(name, pipe, X_test, y_test, ("Benign", "Malicious"))
        binary_results.append(result)
        binary_models[name] = pipe
    best_binary = max(binary_results, key=lambda r: (r["f1"], r["recall"], r["roc_auc"]))
    _dump(binary_models[best_binary["model"]], ARTIFACTS / "malicious_detector.joblib")
    for name, pipe in binary_models.items():                      # both, so the Lab can plot new events with either
        _dump(pipe, ARTIFACTS / f"malicious_detector_{_model_file(name)}.joblib")

    pattern_results = []
    for pattern in PATTERNS:
        pattern_y = (df["attack_type"] == pattern).astype(int)
        positives = int(pattern_y.sum())
        if positives < 8 or pattern_y.nunique() < 2:
            raise ValueError(f"Pattern '{pattern}' needs at least 8 positive and 8 negative records; found {positives}.")
        px_train, px_test, py_train, py_test = train_test_split(X, pattern_y, test_size=.25, random_state=42, stratify=pattern_y)
        if prov is not None:
            prov_y = ((prov["label"] == "Malicious") & (prov["attack_type"] == pattern)).astype(int)
            px_train = pd.concat([px_train, prov[FEATURES]], ignore_index=True)
            py_train = pd.concat([py_train, prov_y], ignore_index=True)
        model_results, fitted = [], {}
        for name in MODEL_NAMES:
            pipe = Pipeline([("preprocessor", _preprocessor()), ("classifier", _estimator(name))])
            pipe.fit(px_train, py_train)
            result = _evaluate(name, pipe, px_test, py_test)
            model_results.append(result)
            fitted[name] = pipe
            _dump(pipe, ARTIFACTS / f"pattern_{pattern}_{_model_file(name)}.joblib")
        best = max(model_results, key=lambda r: (r["f1"], r["recall"], r["roc_auc"]))
        _dump(fitted[best["model"]], ARTIFACTS / f"pattern_{pattern}.joblib")
        pattern_results.append({"pattern": pattern, "positive_records": positives, "selected_model": best["model"], "models": model_results})

    malicious = df[(df.label == "Malicious") & df.attack_type.isin(PATTERNS)].copy()
    if prov is not None:
        malicious = pd.concat([malicious[FEATURES + ["label", "attack_type"]],
                               prov[(prov.label == "Malicious") & prov.attack_type.isin(PATTERNS)][FEATURES + ["label", "attack_type"]]],
                              ignore_index=True)
    family_pipe = Pipeline([("preprocessor", _preprocessor()), ("classifier", RandomForestClassifier(
        n_estimators=250, class_weight="balanced_subsample", random_state=42, n_jobs=-1))])
    family_pipe.fit(malicious[FEATURES], malicious.attack_type)
    _dump(family_pipe, ARTIFACTS / "attack_family_classifier.joblib")
    evaluation = {
        "selected_model": best_binary["model"], "results": binary_results,
        "pattern_results": pattern_results, "patterns": PATTERNS, "model_names": MODEL_NAMES,
        "train_size": len(y_train), "test_size": len(y_test),
        "positive_test_count": int(y_test.sum()), "features": FEATURES,
        "dataset_rows": len(df), "dataset_name": dataset_name or "dataframe", "family_training_rows": len(malicious),
        "provisional_rows": 0 if prov is None else len(prov),
        "provisional_malicious": 0 if prov is None else int((prov["label"] == "Malicious").sum()),
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    tmp = ARTIFACTS / "evaluation.json.tmp"
    tmp.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    _replace(tmp, ARTIFACTS / "evaluation.json")             # the Lab page never reads a half-written file
    return evaluation

def load_models():
    detector_path = ARTIFACTS / "malicious_detector.joblib"
    family_path = ARTIFACTS / "attack_family_classifier.joblib"
    if not detector_path.exists() or not family_path.exists():
        train_from_csv()
    return joblib.load(detector_path), joblib.load(family_path)

def load_pattern_models(model_name=None):
    models = {}
    suffix = (model_name or "").lower().replace(" ", "_")
    for pattern in PATTERNS:
        path = ARTIFACTS / (f"pattern_{pattern}_{suffix}.joblib" if suffix else f"pattern_{pattern}.joblib")
        if not path.exists():
            train_from_csv()
        models[pattern] = joblib.load(path)
    return models


def load_overall_model(model_name):
    """The saved Malicious-vs-Benign model for one algorithm ('Random Forest' / 'XGBoost')."""
    path = ARTIFACTS / f"malicious_detector_{_model_file(model_name)}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def load_pattern_model(pattern, model_name):
    """One specific saved pattern model (e.g. 'brute_force', 'XGBoost'). Raises if it has not been trained."""
    path = ARTIFACTS / f"pattern_{pattern}_{_model_file(model_name)}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)
