from pathlib import Path
from datetime import datetime
import ipaddress
import json
import re
import secrets
import threading
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify, abort, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from config import Config
from extensions import db, login_manager
from models import (AppUser, DirectoryUser, Asset, SecurityEvent, Alert,
                    Incident, AuditLog, MLModelRegistry, IncidentTimeline)
from services import (import_starter_data, import_event_csv, audit, save_model_evaluations,
                      dataset_upsert_events, dataset_append, dataset_remove, next_id)
import otp_service
from ml.train import (train_from_csv, load_models, load_pattern_models, load_pattern_model, load_overall_model,
                      ARTIFACTS, FEATURES, PATTERNS)

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(AppUser, int(user_id))

def role_required(*roles):
    def decorator(fn):
        from functools import wraps
        @wraps(fn)
        @login_required
        def wrapped(*args, **kwargs):
            if current_user.role not in roles:
                abort(403)
            return fn(*args, **kwargs)
        return wrapped
    return decorator

def create_demo_accounts():
    """Create the three demo logins on a brand-new database only.
    (If they were re-created on every start, an account removed by the administrator would come back.)"""
    if AppUser.query.count() > 0:
        return
    accounts = [
        ("admin", "Admin@123", "Administrator"),
        ("analyst", "Analyst@123", "Analyst"),
        ("manager", "Manager@123", "Incident Manager")
    ]
    for username, password, role in accounts:
        user = AppUser.query.filter_by(username=username).first()
        if not user:
            user = AppUser(username=username, role=role, email_verified=True)
            user.set_password(password)
            db.session.add(user)
    db.session.commit()

@app.context_processor
def inject_globals():
    return {"now": datetime.utcnow}

@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("auth_login"))

def _login_page():
    return render_template("login.html", has_photo=(Path(app.static_folder) / "img" / "login-bg.jpg").exists())


def _finish_login(user, remember):
    session.pop("otp_user_id", None); session.pop("otp_remember", None)
    login_user(user, remember=remember)
    audit(user.username, "Logged in", "AppUser", user.id)
    db.session.commit()
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def auth_login():
    if request.method == "POST":
        user = otp_service.find_user_by_login(request.form.get("username", ""))
        if user and user.active and user.email_verified and user.check_password(request.form.get("password","")):
            remember = bool(request.form.get("remember"))
            if not user.email:      # no email on file yet: password-only (administrator should add one)
                return _finish_login(user, remember)
            if not otp_service.issue_otp(user, request.remote_addr):
                flash("We could not send the verification email. Please try again or contact your administrator.", "danger")
                return _login_page()
            session["otp_user_id"] = user.id
            session["otp_remember"] = remember
            audit(user.username, "Login code emailed", "AppUser", user.id)
            db.session.commit()
            return redirect(url_for("auth_verify"))
        flash("Invalid email/username or password.", "danger")
    return _login_page()


def _pending_otp_user():
    uid = session.get("otp_user_id")
    user = db.session.get(AppUser, uid) if uid else None
    return user if user and user.active and user.email else None


@app.route("/login/verify", methods=["GET", "POST"])
def auth_verify():
    user = _pending_otp_user()
    if not user:
        return redirect(url_for("auth_login"))
    if request.method == "POST":
        ok, msg = otp_service.verify_otp(user, request.form.get("code", ""))
        if ok:
            return _finish_login(user, bool(session.get("otp_remember")))
        audit(user.username, "Login code rejected", "AppUser", user.id, msg)
        db.session.commit()
        flash(msg, "danger")
    return render_template("verify_otp.html", masked=otp_service.mask_email(user.email),
                           wait=otp_service.seconds_until_resend(user),
                           ttl_minutes=app.config["OTP_TTL_SECONDS"] // 60)


@app.route("/login/resend", methods=["POST"])
def auth_resend():
    user = _pending_otp_user()
    if not user:
        return redirect(url_for("auth_login"))
    if otp_service.seconds_until_resend(user) > 0:
        flash("Please wait a few seconds before requesting another code.", "danger")
    elif otp_service.issue_otp(user, request.remote_addr):
        audit(user.username, "Login code re-sent", "AppUser", user.id)
        db.session.commit()
        flash("A new code has been sent.", "success")
    else:
        flash("We could not send the email. Please try again.", "danger")
    return redirect(url_for("auth_verify"))

# ---------------------------------------------------------------------------
# Forgot password:  email -> OTP emailed -> verify OTP -> set new password
# ---------------------------------------------------------------------------
RESET_STEP_SECONDS = 600   # the whole reset must be finished within 10 minutes of verifying the code


def _reset_user():
    uid = session.get("reset_user_id")
    user = db.session.get(AppUser, uid) if uid else None
    return user if user and user.active and user.email else None


def _clear_reset_session():
    for k in ("reset_user_id", "reset_verified_at"):
        session.pop(k, None)


@app.route("/forgot", methods=["GET", "POST"])
def auth_forgot():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Enter a valid email address.", "danger")
            return render_template("forgot_password.html")
        user = AppUser.query.filter(db.func.lower(AppUser.email) == email).first()
        if user and user.active:
            if otp_service.seconds_until_resend(user, "reset") == 0:
                if not otp_service.issue_otp(user, request.remote_addr, "reset"):
                    flash("We could not send the email. Please try again later.", "danger")
                    return render_template("forgot_password.html")
                audit(user.username, "Password reset code emailed", "AppUser", user.id)
                db.session.commit()
            session["reset_user_id"] = user.id
            session.pop("reset_verified_at", None)
        else:
            session["reset_user_id"] = 0        # decoy: same screens either way, so emails cannot be guessed
            session.pop("reset_verified_at", None)
        flash("If that email belongs to an account, a verification code has been sent.", "success")
        return redirect(url_for("auth_forgot_verify"))
    return render_template("forgot_password.html")


@app.route("/forgot/verify", methods=["GET", "POST"])
def auth_forgot_verify():
    if session.get("reset_user_id") is None:
        return redirect(url_for("auth_forgot"))
    user = _reset_user()            # None for an unknown email: the page looks identical, the code never matches
    ttl = app.config["OTP_TTL_SECONDS"] // 60
    if request.method == "POST":
        if not user:
            flash("Incorrect code. Please check your email or request a new one.", "danger")
        else:
            ok, msg = otp_service.verify_otp(user, request.form.get("code", ""), "reset")
            if ok:
                session["reset_verified_at"] = datetime.utcnow().timestamp()
                audit(user.username, "Password reset code verified", "AppUser", user.id)
                db.session.commit()
                return redirect(url_for("auth_reset"))
            audit(user.username, "Password reset code rejected", "AppUser", user.id, msg)
            db.session.commit()
            flash(msg, "danger")
    return render_template("verify_otp.html", reset=True, ttl_minutes=ttl,
                           masked=otp_service.mask_email(user.email) if user else "your email",
                           wait=otp_service.seconds_until_resend(user, "reset") if user else 0)


@app.route("/forgot/resend", methods=["POST"])
def auth_forgot_resend():
    user = _reset_user()
    if not user:
        return redirect(url_for("auth_forgot"))
    if otp_service.seconds_until_resend(user, "reset") > 0:
        flash("Please wait a few seconds before requesting another code.", "danger")
    elif otp_service.issue_otp(user, request.remote_addr, "reset"):
        audit(user.username, "Password reset code re-sent", "AppUser", user.id)
        db.session.commit()
        flash("A new code has been sent.", "success")
    else:
        flash("We could not send the email. Please try again.", "danger")
    return redirect(url_for("auth_forgot_verify"))


@app.route("/forgot/reset", methods=["GET", "POST"])
def auth_reset():
    user = _reset_user()
    verified = session.get("reset_verified_at")
    if not user or not verified or datetime.utcnow().timestamp() - verified > RESET_STEP_SECONDS:
        _clear_reset_session()
        flash("Please verify your email again to reset the password.", "danger")
        return redirect(url_for("auth_forgot"))
    if request.method == "POST":
        pw, confirm = request.form.get("password", ""), request.form.get("confirm", "")
        if len(pw) < 8:
            flash("Password must be at least 8 characters.", "danger")
        elif pw != confirm:
            flash("The two passwords do not match.", "danger")
        elif user.check_password(pw):
            flash("Choose a password different from your current one.", "danger")
        else:
            user.set_password(pw)
            audit(user.username, "Password reset via email OTP", "AppUser", user.id)
            db.session.commit()
            _clear_reset_session()
            flash("Password updated. Please sign in with your new password.", "success")
            return redirect(url_for("auth_login"))
    return render_template("reset_password.html", masked=otp_service.mask_email(user.email))



# ---------------------------------------------------------------------------
# Sign up:  username + email + password -> OTP emailed -> verify -> account active
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
USERNAME_RE = re.compile(r"[A-Za-z0-9_.-]{3,40}")


def _signup_user():
    uid = session.get("signup_user_id")
    user = db.session.get(AppUser, uid) if uid else None
    return user if user and not user.email_verified and user.email else None


@app.route("/signup", methods=["GET", "POST"])
def auth_signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        try:
            if not USERNAME_RE.fullmatch(username):
                raise ValueError("Username must be 3-40 characters: letters, numbers, dot, dash or underscore.")
            if not EMAIL_RE.fullmatch(email):
                raise ValueError("Enter a valid email address.")
            if len(password) < 8:
                raise ValueError("Password must be at least 8 characters.")
            if password != request.form.get("confirm", ""):
                raise ValueError("The two passwords do not match.")
            found = {u.id: u for u in (otp_service.find_user_by_email(email),
                     AppUser.query.filter(db.func.lower(AppUser.username) == username.lower()).first()) if u}.values()
            if any(u.email_verified for u in found):
                raise ValueError("That email or username is already registered. Try signing in instead.")
            # sign-ups that never entered their code are abandoned: reuse the first one, drop any others
            stale = list(found)
            for extra in stale[1:]:
                db.session.delete(extra)
            user = stale[0] if stale else AppUser(role=app.config["SIGNUP_DEFAULT_ROLE"], active=False, email_verified=False)
            db.session.flush()
            user.username, user.email = username, email
            user.set_password(password)
            db.session.add(user)
            db.session.flush()
            if not otp_service.issue_otp(user, request.remote_addr, purpose="signup"):
                db.session.rollback()
                raise ValueError("We could not send the verification email. Please try again.")
            audit(username, "Sign-up code emailed", "AppUser", user.id)
            db.session.commit()
            session["signup_user_id"] = user.id
            return redirect(url_for("auth_signup_verify"))
        except ValueError as exc:
            db.session.rollback()
            flash(str(exc), "danger")
    return render_template("signup.html")


@app.route("/signup/verify", methods=["GET", "POST"])
def auth_signup_verify():
    user = _signup_user()
    if not user:
        return redirect(url_for("auth_signup"))
    if request.method == "POST":
        ok, msg = otp_service.verify_otp(user, request.form.get("code", ""), "signup")
        if ok:
            user.email_verified = True
            user.active = True
            audit(user.username, "Account created (email verified)", "AppUser", user.id, f"role={user.role}")
            db.session.commit()
            session.pop("signup_user_id", None)
            flash("Email verified - your account is ready. Please sign in.", "success")
            return redirect(url_for("auth_login"))
        flash(msg, "danger")
    return render_template("verify_otp.html", signup=True, masked=otp_service.mask_email(user.email),
                           wait=otp_service.seconds_until_resend(user, "signup"),
                           ttl_minutes=app.config["OTP_TTL_SECONDS"] // 60)


@app.route("/signup/resend", methods=["POST"])
def auth_signup_resend():
    user = _signup_user()
    if not user:
        return redirect(url_for("auth_signup"))
    if otp_service.seconds_until_resend(user, "signup") > 0:
        flash("Please wait a few seconds before requesting another code.", "danger")
    elif otp_service.issue_otp(user, request.remote_addr, purpose="signup"):
        flash("A new code has been sent.", "success")
    else:
        flash("We could not send the email. Please try again.", "danger")
    return redirect(url_for("auth_signup_verify"))


@app.route("/logout")
@login_required
def auth_logout():
    audit(current_user.username, "Logged out", "AppUser", current_user.id)
    db.session.commit()
    logout_user()
    return redirect(url_for("auth_login"))

EVENT_COLUMNS = ["event_id", "event_time", "src_ip", "dst_ip", "event_type", "protocol", "dst_port",
                 "failed_login_count", "bytes_sent", "label", "attack_type", "predicted_label",
                 "predicted_attack_type", "model_score", "ingested_at"]
INCIDENT_STAGES = ["New", "Investigating", "Pending Review", "Contained", "Resolved", "Closed"]
CLOSED_ALERT = ("Escalated", "Closed", "Closed - Benign")


def events_dataframe():
    """Every event currently in the database (starter + imported + entered), one DataFrame row each."""
    cols = [getattr(SecurityEvent, c) for c in EVENT_COLUMNS]
    return pd.DataFrame(db.session.query(*cols).all(), columns=EVENT_COLUMNS)


def _effective(frame):
    """Ground truth when it exists, otherwise the ML prediction (events typed into the form have no label)."""
    frame = frame.copy()
    truth = frame["label"].isin(["Benign", "Malicious"])
    frame["is_new"] = ~truth
    frame["malicious"] = frame["label"].where(truth, frame["predicted_label"]).eq("Malicious")
    frame["pattern"] = frame["attack_type"].where(truth, frame["predicted_attack_type"]).fillna("").astype(str).str.lower()
    frame["ts"] = pd.to_datetime(frame["event_time"], dayfirst=True, format="mixed", errors="coerce")
    frame["ts"] = frame["ts"].fillna(pd.to_datetime(frame["ingested_at"], errors="coerce"))
    return frame[truth | frame["predicted_label"].notna()]


def _grouped(query, order=None):
    """Query rows of (name, count) -> ordered {"labels": [...], "values": [...]} (None names become 'Unknown')."""
    rows = [((name or "Unknown"), int(count)) for name, count in query]
    rows.sort(key=(lambda r: order.index(r[0]) if r[0] in order else len(order)) if order else (lambda r: -r[1]))
    return {"labels": [r[0] for r in rows], "values": [r[1] for r in rows]}


def _dashboard_series():
    """All numbers the dashboard draws. Used for the first render and for the live refresh."""
    frame = _effective(events_dataframe())
    mal = frame[frame["malicious"]]
    ben = frame[~frame["malicious"]]
    days = pd.date_range(frame["ts"].min().floor("D"), frame["ts"].max().floor("D"), freq="D") if len(frame) else []
    daily = lambda part: part.groupby(part["ts"].dt.floor("D")).size().reindex(days, fill_value=0).tolist()
    hourly = lambda part: part.groupby(part["ts"].dt.hour).size().reindex(range(24), fill_value=0).tolist()
    scores = pd.to_numeric(frame["model_score"], errors="coerce").dropna()
    hist = np.histogram(scores, bins=10, range=(0, 1))[0].tolist() if len(scores) else [0] * 10
    top = mal["src_ip"].value_counts().head(6)
    alerts = Alert.query.count()
    incidents = Incident.query.count()
    resolved = Incident.query.filter(Incident.status.in_(["Resolved", "Closed"])).count()
    stage_counts = dict(db.session.query(Incident.status, db.func.count(Incident.incident_id)).group_by(Incident.status).all())
    stages = {s: 0 for s in INCIDENT_STAGES}
    for name, count in stage_counts.items():
        stages[name if name in stages else "New"] += count
    open_alerts = Alert.query.filter(Alert.status.in_(["New", "Escalated", "In Progress"])).count()
    return {
        "kpi": {"total": int(len(frame)), "malicious": int(len(mal)), "benign": int(len(ben)),
                "new_events": int(frame["is_new"].sum()), "alerts": alerts, "open_alerts": open_alerts,
                "incidents": incidents, "unresolved": incidents - resolved,
                "analyst_queue": Alert.query.filter(Alert.status.in_(["New", "In Progress"])).count(),
                "critical": Alert.query.filter(db.func.lower(Alert.severity) == "critical").count(),
                "high": Alert.query.filter(db.func.lower(Alert.severity) == "high").count()},
        "timeline": {"labels": [d.strftime("%d %b") for d in days], "malicious": daily(mal), "benign": daily(ben)},
        "hours": {"malicious": hourly(mal), "benign": hourly(ben)},
        "patterns": {"labels": PATTERNS, "values": [int((mal["pattern"] == p).sum()) for p in PATTERNS]},
        "scores": {"labels": [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)], "values": hist},
        "sources": {"labels": top.index.tolist(), "values": [int(v) for v in top.values]},
        "funnel": {"labels": ["Events ingested", "Flagged malicious", "Alerts raised", "Incidents opened", "Incidents resolved"],
                   "values": [int(len(frame)), int(len(mal)), alerts, incidents, resolved]},
        "severity": _grouped(db.session.query(Alert.severity, db.func.count(Alert.alert_id)).group_by(Alert.severity).all(),
                             ["Critical", "High", "Medium", "Low"]),
        "alert_status": _grouped(db.session.query(Alert.status, db.func.count(Alert.alert_id)).group_by(Alert.status).all()),
        "incident_stages": {"labels": INCIDENT_STAGES, "values": [stages[name] for name in INCIDENT_STAGES]},
        "incident_priority": _grouped(db.session.query(Incident.priority, db.func.count(Incident.incident_id)).group_by(Incident.priority).all(),
                                      ["Critical", "High", "Medium", "Low"]),
    }


@app.route("/dashboard")
@login_required
def dashboard():
    series = _dashboard_series()
    k = series["kpi"]
    recent = SecurityEvent.query.order_by(SecurityEvent.ingested_at.desc()).limit(8).all()
    incident_statuses = [(n, c) for n, c in zip(series["incident_stages"]["labels"], series["incident_stages"]["values"]) if c]
    return render_template("dashboard.html", total=k["total"], malicious=k["malicious"], benign=k["benign"],
                           alerts=k["alerts"], open_alerts=k["open_alerts"], incidents=k["incidents"],
                           incident_statuses=incident_statuses, recent=recent, critical=k["critical"],
                           high=k["high"], role=current_user.role, analyst_queue=k["analyst_queue"],
                           unresolved_incidents=k["unresolved"], series=series)


@app.route("/api/dashboard/series")
@login_required
def dashboard_series():
    series = _dashboard_series()
    if current_user.role == "Incident Manager":      # managers have no access to event-level data
        for key in ("timeline", "hours", "patterns", "scores", "sources"):
            series.pop(key, None)
        for key in ("total", "malicious", "benign", "new_events"):
            series["kpi"].pop(key, None)
    return jsonify(series)


def score_unscored_events(event_ids=None, unlabelled_only=False):
    """Run current detector and all five pattern models on newly ingested rows."""
    query = SecurityEvent.query.filter(SecurityEvent.predicted_label.is_(None))
    if unlabelled_only:                      # start-up catch-up: starter rows already carry ground truth
        query = query.filter(SecurityEvent.label.is_(None))
    if event_ids is not None:
        if not event_ids:
            return 0
        query = query.filter(SecurityEvent.event_id.in_(event_ids))
    pending = query.all()
    if not pending:
        return 0
    detector, _family_model = load_models()
    pattern_models = load_pattern_models()
    for event in pending:
        X = pd.DataFrame([{key: getattr(event, key) for key in FEATURES}])
        probability = float(detector.predict_proba(X)[0, 1])
        predicted = "Malicious" if probability >= 0.5 else "Benign"
        event.predicted_label = predicted
        event.model_score = probability
        scores = {name: float(model.predict_proba(X)[0, 1]) for name, model in pattern_models.items()}
        best_pattern = max(scores, key=scores.get)
        event.predicted_attack_type = best_pattern if predicted == "Malicious" else None
        if predicted == "Malicious" and not Alert.query.filter_by(event_id=event.event_id).first():
            severity = "Critical" if probability >= .90 else ("High" if probability >= .75 else "Medium")
            alert_id = f"ALT-{datetime.utcnow().strftime('%y%m%d%H%M%S%f')}"
            db.session.add(Alert(alert_id=alert_id, event_id=event.event_id,
                alert_title=f"ML detected {best_pattern.replace('_', ' ').title()}",
                model_score=probability, severity=severity, status="New", ground_truth_label=None))
    db.session.commit()
    return len(pending)


# ---------------------------------------------------------------------------
# Dataset sync + automatic retraining
# ---------------------------------------------------------------------------
_train_lock = threading.Lock()
_train_state = {"running": False, "pending": False, "error": None, "reason": "", "finished": None}


def _read_evaluation():
    path = ARTIFACTS / "evaluation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _sync_dataset(events):
    """Write events to data/security_events.csv. Returns a warning text if the file could not be written
    (for example when it is open and locked in Excel) - the database copy is saved either way."""
    try:
        dataset_upsert_events(events)
        return None
    except Exception as exc:
        return f"The event was saved to the database but data/security_events.csv could not be updated ({exc}). Close the file if it is open in another program."


def _training_frames():
    """(labelled events, events that only have a model prediction) as DataFrames of FEATURES + label + attack_type."""
    cols = FEATURES + ["label", "attack_type"]
    labelled = pd.DataFrame([{k: getattr(e, k) for k in cols} for e in
                             SecurityEvent.query.filter(SecurityEvent.label.in_(["Benign", "Malicious"])).all()], columns=cols)
    provisional = pd.DataFrame(columns=cols)
    if app.config.get("TRAIN_ON_PREDICTED", True):
        rows = SecurityEvent.query.filter(SecurityEvent.label.is_(None), SecurityEvent.predicted_label.isnot(None)).all()
        provisional = pd.DataFrame([{**{k: getattr(e, k) for k in FEATURES}, "label": e.predicted_label,
                                     "attack_type": e.predicted_attack_type} for e in rows], columns=cols)
    return labelled, provisional


def _retrain_worker(labelled, provisional):
    while True:
        error = None
        try:
            if len(labelled) >= 40:
                result = train_from_csv(df=labelled, provisional=provisional, dataset_name="database (labelled events)")
            else:
                result = train_from_csv()
            with app.app_context():
                save_model_evaluations(result)
                audit("system", "Retrained ML models", "Model", result["selected_model"],
                      f"labelled={result['dataset_rows']}; provisional(new events)={result.get('provisional_rows', 0)}")
                db.session.commit()
        except Exception as exc:                                   # keep the app alive; show the error in the Lab
            error = f"{type(exc).__name__}: {exc}"
        with _train_lock:
            _train_state["error"] = error
            _train_state["finished"] = datetime.utcnow().isoformat(timespec="seconds")
            again = _train_state["pending"]
            _train_state["pending"] = False
            _train_state["running"] = again
        if not again:
            return
        with app.app_context():                                    # events arrived while training: go again, once
            labelled, provisional = _training_frames()


def request_retrain(reason=""):
    """Retrain every model in the background so the request that saved the event does not wait.
    Events that arrive during a training run are collected into ONE follow-up run."""
    with _train_lock:
        if _train_state["running"]:
            _train_state["pending"] = True
            return "queued"
        _train_state.update(running=True, pending=False, error=None, reason=reason)
    try:
        frames = _training_frames()
        threading.Thread(target=_retrain_worker, args=frames, daemon=True, name="ml-retrain").start()
    except Exception:
        with _train_lock:
            _train_state["running"] = False
        raise
    return "started"


def _training_status():
    evaluation = _read_evaluation() or {}
    with _train_lock:
        state = dict(_train_state)
    return {"running": state["running"], "pending": state["pending"], "error": state["error"],
            "reason": state["reason"], "trained_at": evaluation.get("trained_at", ""),
            "labelled_rows": evaluation.get("dataset_rows", 0),
            "provisional_rows": evaluation.get("provisional_rows", 0)}


@app.route("/api/ml/status")
@role_required("Administrator", "Analyst")
def ml_status():
    return jsonify(_training_status())


def _validate_event_form(form):
    """Clean values for a new event; raises ValueError with a readable message."""
    event_id = form["event_id"].strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,30}", event_id):
        raise ValueError("Event ID must be 3-30 letters, numbers, dash or underscore.")
    clean = {"event_id": event_id, "event_time": form["event_time"].strip(),
             "event_type": form["event_type"].strip().lower(), "protocol": form["protocol"].strip().upper(),
             "severity_hint": form["severity_hint"]}
    if pd.isna(pd.to_datetime(clean["event_time"], dayfirst=True, errors="coerce")):
        raise ValueError("Event time is not a valid date/time. Use the format 18-11-2026 14:30.")
    for field in ("src_ip", "dst_ip"):
        try:
            clean[field] = str(ipaddress.ip_address(form[field].strip()))
        except ValueError:
            raise ValueError(f"{field.replace('_', ' ').title()} is not a valid IP address.")
    for field, high in (("dst_port", 65535), ("failed_login_count", 10 ** 6), ("bytes_sent", 2 ** 31 - 1)):
        value = int(form[field])
        if not 0 <= value <= high:
            raise ValueError(f"{field.replace('_', ' ').title()} must be between 0 and {high}.")
        clean[field] = value
    if not clean["event_type"] or not clean["protocol"]:
        raise ValueError("Event type and protocol are required.")
    clean["user_id"] = form.get("user_id") or None
    clean["asset_id"] = form.get("asset_id") or None
    return clean

@app.route("/events")
@role_required("Administrator", "Analyst")
def events():
    q = SecurityEvent.query
    search = request.args.get("q","").strip()
    label = request.args.get("label","")
    if search:
        q = q.filter(db.or_(SecurityEvent.event_id.contains(search),
                            SecurityEvent.src_ip.contains(search),
                            SecurityEvent.dst_ip.contains(search),
                            SecurityEvent.event_type.contains(search)))
    if label in ("Benign","Malicious"):
        q = q.filter_by(label=label)
    page = max(int(request.args.get("page",1)), 1)
    pagination = q.order_by(SecurityEvent.event_time.desc()).paginate(page=page, per_page=25, error_out=False)
    return render_template("events.html", pagination=pagination, search=search, label=label)

@app.route("/events/import", methods=["POST"])
@role_required("Administrator", "Analyst")
def events_import():
    file = request.files.get("file")
    if not file or not file.filename:
        flash("Choose a CSV file first.", "warning")
        return redirect(url_for("events"))
    try:
        added, skipped, added_ids = import_event_csv(file, current_user.username)
        score_unscored_events(added_ids)
        retrain = request_retrain(f"CSV import ({added} events)") if added else None
        flash(f"Import complete. Added {added}; skipped duplicate IDs {skipped}. New events were scored by the ML models"
              + (" and added to data/security_events.csv. The models are retraining in the background; ML Model Lab refreshes itself." if retrain else "."), "success")
    except Exception as exc:
        db.session.rollback()
        flash(f"Import failed: {exc}", "danger")
    return redirect(url_for("events"))

@app.route("/events/new", methods=["GET","POST"])
@role_required("Administrator", "Analyst")
def event_new():
    if request.method == "POST":
        try:
            values = _validate_event_form(request.form)
            event_id = values["event_id"]
            if db.session.get(SecurityEvent, event_id):
                raise ValueError("That event_id already exists.")
            event = SecurityEvent(**values, label=None, attack_type=None)
            db.session.add(event)
            db.session.flush()
            score_unscored_events([event_id])
            event = db.session.get(SecurityEvent, event_id)
            audit(current_user.username, "Created and scored event", "SecurityEvent", event_id,
                  f"Predicted={event.predicted_label}; score={event.model_score:.4f}")
            db.session.commit()
            warning = _sync_dataset([event])                       # the event becomes part of the dataset ...
            retrain = request_retrain(f"new event {event_id}")     # ... and the models learn from it
            flash(f"Event scored: {event.predicted_label} ({event.model_score:.1%} malicious probability). "
                  "It was added to data/security_events.csv and the models are retraining in the background - "
                  "ML Model Lab, the graphs and the Dashboard refresh themselves.", "success")
            if warning:
                flash(warning, "warning")
            return redirect(url_for("event_detail", event_id=event_id))
        except Exception as exc:
            db.session.rollback()
            flash(f"Could not add event: {exc}", "danger")
    users = DirectoryUser.query.order_by(DirectoryUser.user_id).all()
    assets = Asset.query.order_by(Asset.asset_id).all()
    suggested_id = next_id([r[0] for r in db.session.query(SecurityEvent.event_id).all()], "EVT", 5)
    return render_template("event_new.html", users=users, assets=assets, suggested_id=suggested_id,
                           suggested_time=datetime.now().strftime("%d-%m-%Y %H:%M"), form=request.form)

@app.route("/events/<event_id>")
@role_required("Administrator", "Analyst")
def event_detail(event_id):
    event = SecurityEvent.query.get_or_404(event_id)
    related_alerts = Alert.query.filter_by(event_id=event_id).all()
    return render_template("event_detail.html", event=event, related_alerts=related_alerts, patterns=PATTERNS)

@app.route("/events/<event_id>/label", methods=["POST"])
@role_required("Administrator", "Analyst")
def event_label(event_id):
    """An analyst confirms what a new (unlabelled) event really was. It becomes ground truth for training."""
    event = SecurityEvent.query.get_or_404(event_id)
    if event.label in ("Benign", "Malicious"):
        flash("This event already has a ground-truth label.", "warning")
        return redirect(url_for("event_detail", event_id=event_id))
    label = request.form.get("label", "")
    attack = (request.form.get("attack_type") or "").strip().lower()
    if label not in ("Benign", "Malicious") or (label == "Malicious" and attack not in PATTERNS):
        flash("Choose Benign, or Malicious together with the attack pattern.", "danger")
        return redirect(url_for("event_detail", event_id=event_id))
    event.label = label
    event.attack_type = attack if label == "Malicious" else None
    for alert in Alert.query.filter_by(event_id=event_id).all():
        alert.ground_truth_label = label
    audit(current_user.username, "Confirmed event label", "SecurityEvent", event_id,
          f"label={label}; attack_type={event.attack_type or 'none'}; model had predicted {event.predicted_label}")
    db.session.commit()
    warning = _sync_dataset([event])
    request_retrain(f"label confirmed for {event_id}")
    flash(f"{event_id} is now labelled {label}. The dataset was updated and the models are retraining.", "success")
    if warning:
        flash(warning, "warning")
    return redirect(url_for("event_detail", event_id=event_id))


@app.route("/alerts")
@role_required("Administrator", "Analyst", "Incident Manager")
def alerts():
    items = Alert.query.order_by(Alert.created_at.desc()).all()
    return render_template("alerts.html", items=items)

@app.route("/alerts/<alert_id>/status", methods=["POST"])
@role_required("Administrator","Analyst","Incident Manager")
def alert_status(alert_id):
    item = Alert.query.get_or_404(alert_id)
    item.status = request.form.get("status","In Progress")
    audit(current_user.username, "Updated alert status", "Alert", alert_id, item.status)
    db.session.commit()
    flash("Alert status updated.", "success")
    return redirect(url_for("alerts"))

def _timeline(incident_id, action, description="", user=None):
    db.session.add(IncidentTimeline(incident_id=incident_id, action=action, description=description,
                                    performed_by=user or (current_user.username if current_user.is_authenticated else "system")))


@app.route("/incidents")
@role_required("Administrator", "Analyst", "Incident Manager")
def incidents():
    items = Incident.query.order_by(Incident.created_at.desc()).all()
    board = {stage: [] for stage in INCIDENT_STAGES}
    for item in items:
        board[item.status if item.status in board else "New"].append(item)
    counts = {stage: len(rows) for stage, rows in board.items()}
    timelines = {}
    for entry in IncidentTimeline.query.order_by(IncidentTimeline.created_at.desc()).all():
        timelines.setdefault(entry.incident_id, [])
        if len(timelines[entry.incident_id]) < 4:
            timelines[entry.incident_id].append(entry)
    with_incident = db.session.query(Incident.related_alert_id).filter(Incident.related_alert_id.isnot(None))
    waiting = Alert.query.filter(Alert.status.notin_(CLOSED_ALERT), ~Alert.alert_id.in_(with_incident)).all()
    rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    waiting.sort(key=lambda a: (rank.get((a.severity or "").lower(), 9), -(a.model_score or 0)))
    series = _dashboard_series()
    return render_template("incidents.html", items=items, stages=INCIDENT_STAGES, board=board, counts=counts,
                           timelines=timelines, waiting=waiting[:8], waiting_total=len(waiting),
                           funnel=series["funnel"], priority=series["incident_priority"],
                           can_edit=current_user.role in ("Administrator", "Incident Manager"))


@app.route("/incidents/create/<alert_id>", methods=["POST"])
@role_required("Administrator","Analyst","Incident Manager")
def incident_create(alert_id):
    alert = Alert.query.get_or_404(alert_id)
    if Incident.query.filter_by(related_alert_id=alert_id).first():
        flash("An incident already exists for that alert.", "warning")
        return redirect(url_for("incidents"))
    incident_id = f"INC-{datetime.utcnow().strftime('%y%m%d%H%M%S%f')}"
    db.session.add(Incident(incident_id=incident_id,
        incident_title=f"Investigation: {alert.alert_title}", related_alert_id=alert_id,
        status="New", priority=alert.severity, assigned_to=alert.assigned_to,
        root_cause="Pending analyst investigation"))
    alert.status = "Escalated"
    db.session.flush()
    _timeline(incident_id, "Incident opened", f"Escalated from alert {alert_id} ({alert.severity}).")
    audit(current_user.username, "Created incident from alert", "Incident", incident_id, alert_id)
    db.session.commit()
    flash("Incident created from alert and placed in the pipeline at stage New.", "success")
    return redirect(url_for("incidents"))


@app.route("/incidents/<incident_id>/stage", methods=["POST"])
@role_required("Administrator", "Incident Manager")
def incident_stage(incident_id):
    """Move an incident one step along the pipeline (or straight to a chosen stage)."""
    item = Incident.query.get_or_404(incident_id)
    current = item.status if item.status in INCIDENT_STAGES else "New"
    target, direction = request.form.get("stage"), request.form.get("direction")
    index = INCIDENT_STAGES.index(current)
    if target in INCIDENT_STAGES:
        new = target
    elif direction == "next":
        new = INCIDENT_STAGES[min(index + 1, len(INCIDENT_STAGES) - 1)]
    elif direction == "back":
        new = INCIDENT_STAGES[max(index - 1, 0)]
    else:
        abort(400)
    if new != item.status:
        old, item.status = item.status, new
        _timeline(incident_id, f"Stage: {old} → {new}", "Moved on the incident pipeline.")
        audit(current_user.username, "Moved incident stage", "Incident", incident_id, f"{old} -> {new}")
        db.session.commit()
        flash(f"{incident_id} moved to {new}.", "success")
    return redirect(url_for("incidents") + f"#{incident_id}")


@app.route("/incidents/<incident_id>/update", methods=["POST"])
@role_required("Administrator","Incident Manager")
def incident_update(incident_id):
    item = Incident.query.get_or_404(incident_id)
    old_status = item.status
    item.status = request.form.get("status", item.status)
    item.root_cause = request.form.get("root_cause", item.root_cause)
    item.resolution_notes = request.form.get("resolution_notes", item.resolution_notes)
    if item.status != old_status:
        _timeline(incident_id, f"Stage: {old_status} → {item.status}", "Changed in the incident form.")
    else:
        _timeline(incident_id, "Details updated", "Root cause / resolution notes edited.")
    audit(current_user.username, "Updated incident", "Incident", incident_id, item.status)
    db.session.commit()
    flash("Incident updated.", "success")
    return redirect(url_for("incidents") + f"#{incident_id}")

@app.route("/copilot", methods=["GET"])
@login_required
def copilot():
    return render_template("copilot.html")

@app.route("/api/copilot", methods=["POST"])
@login_required
def copilot_chat():
    """Local, database-grounded investigation assistant (no external LLM/API)."""
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("message", "")).strip()[:500]
    if not question:
        return jsonify({"reply": "Please enter an event ID, source IP, alert ID, or attack pattern."}), 400

    q = question.lower()
    events_q = SecurityEvent.query
    matched_event = None
    # Exact identifiers and IPs are the most reliable evidence anchors.
    exact = SecurityEvent.query.filter(db.or_(
        SecurityEvent.event_id == question,
        SecurityEvent.src_ip == question,
        SecurityEvent.dst_ip == question
    )).first()
    if exact:
        matched_event = exact
    else:
        aliases = {
            "brute force": "brute_force", "brute-force": "brute_force",
            "port scan": "port_scan_like", "port scanning": "port_scan_like",
            "unusual transfer": "unusual_transfer", "data transfer": "unusual_transfer",
            "suspicious privilege": "suspicious_privilege", "privilege": "suspicious_privilege",
            "malware": "malware_like"
        }
        family = next((v for k, v in aliases.items() if k in q), None)
        if family:
            matched_event = SecurityEvent.query.filter(db.func.lower(SecurityEvent.attack_type) == family).order_by(
                SecurityEvent.event_time.desc()).first()
        elif "malicious" in q or "attack" in q or "threat" in q:
            matched_event = SecurityEvent.query.filter(db.or_(
                SecurityEvent.label == "Malicious", SecurityEvent.predicted_label == "Malicious")).order_by(
                SecurityEvent.event_time.desc()).first()
        elif "benign" in q:
            matched_event = SecurityEvent.query.filter_by(label="Benign").first()

    if matched_event:
        e = matched_event
        asset = db.session.get(Asset, e.asset_id) if e.asset_id else None
        related_alerts = Alert.query.filter_by(event_id=e.event_id).all()
        lines = [f"I found event {e.event_id} in your SentinelOps database.",
                 f"Event type: {e.event_type}; protocol: {e.protocol}; destination port: {e.dst_port}.",
                 f"Source IP: {e.src_ip}; destination IP: {e.dst_ip}; failed logins: {e.failed_login_count}; bytes sent: {e.bytes_sent}.",
                 f"Dataset label: {e.label or 'not labelled'}; attack family: {e.attack_type or 'not applicable'}.",
                 f"Model prediction: {e.predicted_label or 'not scored'}; malicious probability: {((e.model_score or 0)*100):.1f}%."]
        if asset:
            lines.append(f"Asset: {asset.asset_name} ({asset.asset_id}), criticality {asset.criticality}, environment {asset.environment}.")
        if related_alerts:
            lines.append("Related alerts: " + ", ".join(f"{a.alert_id} ({a.severity}, {a.status})" for a in related_alerts) + ".")
        if e.attack_type == "brute_force" or (e.failed_login_count or 0) >= 5:
            guidance = "Triage repeated authentication failures: confirm the account owner, inspect nearby login events and source reputation, then follow your approved account-containment procedure if confirmed."
        elif e.attack_type == "port_scan_like":
            guidance = "Review the source IP's connection attempts across destination ports and assets. Confirm whether the scanning activity is authorized before containment."
        elif e.attack_type == "unusual_transfer":
            guidance = "Review transfer volume, destination, asset criticality and business justification. Compare with the user's normal transfer pattern before escalation."
        elif e.attack_type == "suspicious_privilege":
            guidance = "Verify who initiated the privilege change, whether approval exists, and whether the account recently showed unusual authentication activity."
        elif e.attack_type == "malware_like":
            guidance = "Check process and file activity on the affected asset, validate endpoint protection results, and isolate only under your approved incident process."
        else:
            guidance = "Correlate this event with nearby events, asset criticality and related alerts. Confirm the evidence with an analyst before escalating."
        reply = "\n".join(lines + ["Recommended investigation: " + guidance,
            "Note: this assistant retrieves real project database evidence and applies local investigation rules. It does not call an external generative-AI service or perform response actions."])
        audit(current_user.username, "Asked database-grounded copilot", "SecurityEvent", e.event_id, question)
        db.session.commit()
        return jsonify({"reply": reply, "event_id": e.event_id})

    # Alert/incident lookup by exact ID when no event was matched.
    alert = db.session.get(Alert, question)
    if alert:
        event = db.session.get(SecurityEvent, alert.event_id) if alert.event_id else None
        return jsonify({"reply": f"Alert {alert.alert_id}: {alert.alert_title}. Severity: {alert.severity}. Status: {alert.status}. Event: {alert.event_id}. " + (f"Source IP: {event.src_ip}; event type: {event.event_type}." if event else "No linked event found.")})
    return jsonify({"reply": "I couldn't find matching evidence. Try an exact event ID (for example EVT001), an alert ID, a source IP, or one of these patterns: brute force, port scanning, unusual data transfer, suspicious privilege, malware-like activity."})

GRAPH_DIR = Path(__file__).resolve().parent / "graphs"
PATTERN_GRAPHS = [
    {"key": "brute_force", "tab": "Brute force", "file": "1_brute_force.png", "title": "Brute force activity",
     "parameters": ["failed_login_count", "event_type", "event_hour", "src_ip"],
     "shows": "Failed logins per hour (blue, left axis) against the number of distinct source IPs (red, right axis). The grey line is normal traffic and the dotted line is the alert threshold.",
     "tip": "Peaks in both lines in the same hour mean many sources are guessing credentials together."},
    {"key": "port_scan_like", "tab": "Port scanning", "file": "2_port_scan.png", "title": "Port scanning activity",
     "parameters": ["dst_port", "protocol", "connection_count_5min"],
     "shows": "Destination ports being probed (blue, left axis, log scale) against the rolling 5-minute connection count (red, right axis). The dotted line marks 100 connections.",
     "tip": "A scan touches many different ports while the connection count climbs."},
    {"key": "unusual_transfer", "tab": "Unusual transfer", "file": "3_unusual_transfer.png", "title": "Unusual data transfer activity",
     "parameters": ["bytes_sent", "total_bytes_received", "bytes_per_second"],
     "shows": "Bytes sent and received (blue and green, left axis) with the transfer speed (red, right axis). Both axes are log scale; the dotted line is the normal 99th percentile of bytes sent.",
     "tip": "Far more bytes sent than received is the signature of data leaving the network."},
    {"key": "suspicious_privilege", "tab": "Privilege abuse", "file": "4_suspicious_privilege.png", "title": "Suspicious privilege activity",
     "parameters": ["event_type", "protocol", "connection_count_5min"],
     "shows": "privilege_change events per day (blue, left axis) against the average connection count per 5 minutes (red, right axis), compared with normal privilege changes.",
     "tip": "Days where privilege changes jump above the normal line deserve a review."},
    {"key": "malware_like", "tab": "Malware-like", "file": "5_malware_like.png", "title": "Malware-like activity",
     "parameters": ["event_type", "protocol", "bytes_sent", "flow_duration_ms"],
     "shows": "Bytes sent (blue, left axis) and flow duration in milliseconds (red, right axis) for process_execution events. Dotted lines are the normal medians.",
     "tip": "Flows far above the normal medians suggest beaconing or data staging."},
    {"key": "curves", "tab": "Parameter curves", "file": "6_parameter_curves.png", "title": "Parameter curves: malicious vs benign",
     "parameters": ["key parameter of each pattern"],
     "shows": "Cumulative distribution of each pattern's key parameter: malicious events (red, dashed) against benign events (blue).",
     "tip": "The wider the gap between the two curves, the better that parameter separates attacks from normal traffic."},
]

def _graph_gallery():
    items = []
    for g in PATTERN_GRAPHS:
        path = GRAPH_DIR / g["file"]
        items.append({**g, "exists": path.exists(),
                      "version": int(path.stat().st_mtime) if path.exists() else 0})
    return items

@app.route("/graphs/<path:name>")
@role_required("Administrator", "Analyst")
def graph_image(name):
    from flask import send_from_directory
    if name not in {g["file"] for g in PATTERN_GRAPHS}:
        abort(404)
    return send_from_directory(GRAPH_DIR, name, max_age=0)

_graph_lock = threading.Lock()
SIGNATURE_FILE = GRAPH_DIR / ".signature"


def _graph_signature():
    """Changes whenever an event is added, imported or (re)scored, so the graphs know they are out of date."""
    total = SecurityEvent.query.count()
    scored = SecurityEvent.query.filter(SecurityEvent.predicted_label.isnot(None)).count()
    last = db.session.query(db.func.max(SecurityEvent.ingested_at)).scalar()
    return f"{total}|{scored}|{last.isoformat() if last else ''}"


def refresh_graphs(force=False):
    """Redraw the six pattern graphs from the LIVE event table when it changed. Returns True if redrawn."""
    signature = _graph_signature()
    GRAPH_DIR.mkdir(exist_ok=True)

    def up_to_date():
        return (SIGNATURE_FILE.exists() and SIGNATURE_FILE.read_text().strip() == signature
                and all((GRAPH_DIR / g["file"]).exists() for g in PATTERN_GRAPHS))

    if not force and up_to_date():
        return False
    with _graph_lock:                                  # matplotlib is not thread-safe
        if not force and up_to_date():
            return False
        import matplotlib
        matplotlib.use("Agg")
        import sentinelops_pattern_graphs as g
        g.generate_all(g.prepare_data(events_dataframe()))
        SIGNATURE_FILE.write_text(signature)
    return True


def _graph_meta():
    files = [GRAPH_DIR / g["file"] for g in PATTERN_GRAPHS if (GRAPH_DIR / g["file"]).exists()]
    return {"signature": SIGNATURE_FILE.read_text().strip() if SIGNATURE_FILE.exists() else "",
            "total": SecurityEvent.query.count(),
            "new_events": SecurityEvent.query.filter(SecurityEvent.label.is_(None),
                                                      SecurityEvent.predicted_label.isnot(None)).count(),
            "updated": datetime.fromtimestamp(max(f.stat().st_mtime for f in files)).strftime("%d %b %Y %H:%M:%S") if files else "never",
            "versions": {g["file"]: int((GRAPH_DIR / g["file"]).stat().st_mtime * 1000)
                         for g in PATTERN_GRAPHS if (GRAPH_DIR / g["file"]).exists()}}


@app.route("/api/graphs/status")
@role_required("Administrator", "Analyst")
def graphs_status():
    """Polled by the ML Model Lab page: redraws the graphs if events changed, then reports what is current."""
    try:
        refresh_graphs()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(_graph_meta())


@app.route("/graphs/regenerate", methods=["POST"])
@role_required("Administrator")
def graphs_regenerate():
    """Force a rebuild of every pattern graph from the events currently in the database."""
    try:
        refresh_graphs(force=True)
        audit(current_user.username, "Regenerated pattern graphs", "MLModel", "graphs")
        db.session.commit()
        flash("Pattern graphs regenerated from the current database events.", "success")
    except Exception as exc:
        db.session.rollback()
        flash(f"Could not regenerate graphs: {exc}", "danger")
    return redirect(url_for("reports"))

def _live_points(pattern, model_name, evaluation):
    """New (unlabelled, ML-scored) events as extra scatter points, scored by the SELECTED model."""
    events = SecurityEvent.query.filter(SecurityEvent.label.is_(None), SecurityEvent.predicted_label.isnot(None)
                                        ).order_by(SecurityEvent.ingested_at.desc()).limit(200).all()
    if not events:
        return []
    pipe = None
    try:
        if pattern == "overall":
            try:
                pipe = load_overall_model(model_name)
            except FileNotFoundError:               # artifacts from before both models were saved
                if evaluation and model_name == evaluation.get("selected_model"):
                    pipe = load_models()[0]
        else:
            pipe = load_pattern_model(pattern, model_name)
    except Exception:
        pipe = None
    if pipe is None:
        return []
    X = pd.DataFrame([{key: getattr(e, key) for key in FEATURES} for e in events])
    probabilities = pipe.predict_proba(X)[:, 1]
    return [{"event_id": e.event_id, "dst_port": int(e.dst_port or 0), "failed_login_count": int(e.failed_login_count or 0),
             "bytes_sent": float(e.bytes_sent or 0), "probability": round(float(p), 4)}
            for e, p in zip(events, probabilities)]

@app.route("/reports")
@role_required("Administrator", "Analyst")
def reports():
    """ML Model Lab: Random Forest vs XGBoost, curves and scatter plots built from the input parameters."""
    evaluation = _read_evaluation()
    patterns = (["overall"] + evaluation.get("patterns", [])) if evaluation else []
    model_names = evaluation.get("model_names", []) if evaluation else []
    selected_pattern = request.args.get("pattern", "overall")
    if patterns and selected_pattern not in patterns:
        selected_pattern = patterns[0]
    selected_model = request.args.get("model", "")
    compare, selected_result, best_model = [], None, ""
    if evaluation:
        if selected_pattern == "overall":
            compare, best_model = evaluation.get("results", []), evaluation.get("selected_model", "")
        else:
            entry = next((pr for pr in evaluation.get("pattern_results", []) if pr["pattern"] == selected_pattern), None)
            compare, best_model = (entry["models"], entry.get("selected_model", "")) if entry else ([], "")
        if compare:
            # default to the model that performed best for this pattern; the user may still pick the other one
            wanted = selected_model if selected_model in {r["model"] for r in compare} else (best_model or compare[0]["model"])
            selected_result = next(r for r in compare if r["model"] == wanted)
            selected_model = selected_result["model"]
    metric_keys = ["precision", "recall", "f1", "roc_auc", "average_precision"]
    metric_labels = {"precision": "Precision", "recall": "Recall", "f1": "F1 score", "roc_auc": "ROC AUC",
                     "average_precision": "Average precision"}
    selected_metrics = [{"label": metric_labels[key], "value": selected_result.get(key, 0)}
                        for key in metric_keys] if selected_result else []
    selected_signals = [x["parameter"] for x in selected_result.get("parameter_importance", [])[:3]] if selected_result else []
    try:
        refresh_graphs()                      # redraw the pattern graphs if events were added since the last draw
    except Exception as exc:
        flash(f"Pattern graphs could not be refreshed: {exc}", "warning")
    live_stats = _ml_live_stats()
    live_points = _live_points(selected_pattern, selected_model, evaluation) if selected_result else []
    pattern_title = "Malicious" if selected_pattern == "overall" else selected_pattern.replace("_", " ").title()
    return render_template("reports.html", evaluation=evaluation, patterns=patterns, pattern_title=pattern_title,
                           model_names=model_names, selected_pattern=selected_pattern,
                           selected_model=selected_model, selected_result=selected_result,
                           compare=compare, best_model=best_model, selected_metrics=selected_metrics,
                           gallery=_graph_gallery(), gallery_default=("curves" if selected_pattern == "overall" else selected_pattern),
                           selected_signals=selected_signals, live_stats=live_stats,
                           live_points=live_points, graph_meta=_graph_meta(), train_status=_training_status())

def _ml_live_stats():
    """Numbers for the 'Live event predictions' charts - recomputed from the database on every call."""
    total = SecurityEvent.query.count()
    predicted_malicious = SecurityEvent.query.filter_by(predicted_label="Malicious").count()
    predicted_benign = SecurityEvent.query.filter_by(predicted_label="Benign").count()
    rows = db.session.query(SecurityEvent.predicted_attack_type, db.func.count(SecurityEvent.event_id)).filter(
        SecurityEvent.predicted_attack_type.isnot(None)).group_by(SecurityEvent.predicted_attack_type).all()
    scored = SecurityEvent.query.filter(SecurityEvent.model_score.isnot(None)).order_by(
        SecurityEvent.ingested_at.desc()).limit(500).all()
    bins = [0] * 10
    for e in scored:
        bins[min(int(e.model_score * 10), 9)] += 1
    recent = [{"id": e.event_id, "score": round(float(e.model_score), 4), "label": e.predicted_label,
               "pattern": (e.predicted_attack_type or "benign").replace("_", " ")} for e in scored[:15][::-1]]
    return {"total_events": total, "predicted_malicious": predicted_malicious,
            "predicted_benign": predicted_benign,
            "pattern_labels": [r[0] for r in rows], "pattern_counts": [int(r[1]) for r in rows],
            "score_bins": bins, "recent": recent}

@app.route("/api/ml/live-stats")
@role_required("Administrator", "Analyst")
def ml_live_stats():
    return jsonify(_ml_live_stats())

@app.route("/models/train", methods=["POST"])
@role_required("Administrator")
def model_train():
    try:
        state = request_retrain("manual retrain (administrator)")
        audit(current_user.username, "Requested ML retraining", "Model", "all")
        db.session.commit()
        flash("Training started in the background. This page refreshes by itself when the new models are ready."
              if state == "started" else "Training is already running; a follow-up run is queued.", "success")
    except Exception as exc:
        db.session.rollback()
        flash(f"Training failed: {exc}", "danger")
    return redirect(url_for("reports"))

APP_ROLES = ["Administrator", "Analyst", "Incident Manager"]
DIRECTORY_STATUSES = ["Active", "Disabled"]
CRITICALITY_LEVELS = ["Low", "Medium", "High", "Critical"]
ASSET_TYPES = ["Laptop", "Workstation", "Web Server", "File Server", "Database Server", "Server"]
ENVIRONMENTS = ["Production", "Testing", "Development"]
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{2,20}")


def _counts(column):
    """{key: number of events} for SecurityEvent.user_id / asset_id, so the admin sees what a removal affects."""
    return {k: int(n) for k, n in db.session.query(column, db.func.count(SecurityEvent.event_id))
            .filter(column.isnot(None)).group_by(column).all()}


@app.route("/users")
@role_required("Administrator")
def users():
    return render_template("users.html", users=AppUser.query.order_by(AppUser.username).all(),
                           directory=DirectoryUser.query.order_by(DirectoryUser.user_id).all(),
                           event_counts=_counts(SecurityEvent.user_id), roles=APP_ROLES,
                           statuses=DIRECTORY_STATUSES,
                           suggested_user_id=next_id([r[0] for r in db.session.query(DirectoryUser.user_id).all()], "USR", 3))


@app.route("/users/add", methods=["POST"])
@role_required("Administrator")
def user_add():
    """Create an application account with one of the three roles. Saved to the app_users table."""
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    role = request.form.get("role", "")
    email = request.form.get("email", "").strip().lower()
    try:
        if not EMAIL_RE.fullmatch(email):
            raise ValueError("Enter a valid email address (login codes are sent to it).")
        if otp_service.find_user_by_email(email):
            raise ValueError("That email is already used by another account.")
        if not USERNAME_RE.fullmatch(username):
            raise ValueError("Username must be 3-40 characters: letters, numbers, dot, dash or underscore.")
        if role not in APP_ROLES:
            raise ValueError("Choose a valid role.")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        if AppUser.query.filter(db.func.lower(AppUser.username) == username.lower()).first():
            raise ValueError(f"The username '{username}' already exists.")
        account = AppUser(username=username, role=role, active=True, email=email, email_verified=True)
        account.set_password(password)
        db.session.add(account)
        db.session.flush()
        audit(current_user.username, "Created application account", "AppUser", account.id, f"username={username}; role={role}; email={email}")
        db.session.commit()
        flash(f"Account '{username}' created with role {role}.", "success")
    except ValueError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
    return redirect(url_for("users"))


@app.route("/users/<int:account_id>/edit", methods=["POST"])
@role_required("Administrator")
def user_edit(account_id):
    """Edit an account (username, email, role, active, optional new password). Changes are saved to the database."""
    account = AppUser.query.get_or_404(account_id)
    is_self = account.id == current_user.id
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip().lower()
    role = request.form.get("role", account.role)
    new_password = request.form.get("password", "")
    active = bool(request.form.get("active")) if not is_self else True      # you cannot lock yourself out
    if is_self:
        role = account.role                                                  # ... or change your own role
    try:
        if not USERNAME_RE.fullmatch(username):
            raise ValueError("Username must be 3-40 characters: letters, numbers, dot, dash or underscore.")
        if not EMAIL_RE.fullmatch(email):
            raise ValueError("Enter a valid email address.")
        if role not in APP_ROLES:
            raise ValueError("Choose a valid role.")
        if new_password and len(new_password) < 8:
            raise ValueError("New password must be at least 8 characters.")
        if AppUser.query.filter(db.func.lower(AppUser.username) == username.lower(), AppUser.id != account.id).first():
            raise ValueError(f"The username '{username}' already exists.")
        if AppUser.query.filter(db.func.lower(AppUser.email) == email, AppUser.id != account.id).first():
            raise ValueError("That email is already used by another account.")
        loses_admin = account.role == "Administrator" and account.active and (role != "Administrator" or not active)
        if loses_admin and AppUser.query.filter_by(role="Administrator", active=True).count() <= 1:
            raise ValueError("At least one active Administrator account must remain.")
        changes = []
        for label, old, new in (("username", account.username, username), ("email", account.email, email),
                                ("role", account.role, role), ("active", account.active, active)):
            if old != new:
                changes.append(f"{label}: {old} -> {new}")
        account.username, account.email, account.role, account.active = username, email, role, active
        account.email_verified = True
        if new_password:
            account.set_password(new_password)
            changes.append("password changed")
        audit(current_user.username, "Edited application account", "AppUser", account.id, "; ".join(changes) or "no changes")
        db.session.commit()
        flash(f"Account '{username}' updated." if changes else "No changes to save.", "success")
    except ValueError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
    return redirect(url_for("users"))


@app.route("/users/<int:account_id>/delete", methods=["POST"])
@role_required("Administrator")
def user_delete(account_id):
    account = AppUser.query.get_or_404(account_id)
    if account.id == current_user.id:
        flash("You cannot remove the account you are signed in with.", "danger")
    elif account.role == "Administrator" and account.active and \
            AppUser.query.filter_by(role="Administrator", active=True).count() <= 1:
        flash("At least one active Administrator account must remain.", "danger")
    else:
        name = account.username
        db.session.delete(account)
        audit(current_user.username, "Removed application account", "AppUser", account_id, f"username={name}; role={account.role}")
        db.session.commit()
        flash(f"Account '{name}' was removed.", "success")
    return redirect(url_for("users"))


@app.route("/users/directory/add", methods=["POST"])
@role_required("Administrator")
def directory_user_add():
    """Add a person to the user directory (the users that security events refer to)."""
    form = request.form
    try:
        user_id = form.get("user_id", "").strip().upper() or next_id(
            [r[0] for r in db.session.query(DirectoryUser.user_id).all()], "USR", 3)
        username = form.get("username", "").strip()
        status = form.get("account_status", "Active")
        if not ID_PATTERN.fullmatch(user_id):
            raise ValueError("User ID must be 2-20 letters, numbers, dash or underscore.")
        if not username or len(username) > 80:
            raise ValueError("Username is required (80 characters at most).")
        if status not in DIRECTORY_STATUSES:
            raise ValueError("Choose a valid account status.")
        if db.session.get(DirectoryUser, user_id):
            raise ValueError(f"User ID {user_id} already exists.")
        record = {"user_id": user_id, "username": username, "department": form.get("department", "").strip(),
                  "role": form.get("role", "").strip(), "account_status": status,
                  "created_date": datetime.now().strftime("%d-%m-%Y")}
        db.session.add(DirectoryUser(**record))
        audit(current_user.username, "Added directory user", "DirectoryUser", user_id, f"username={username}")
        db.session.commit()
        dataset_append("users.csv", record)
        flash(f"User {user_id} ({username}) added to the directory.", "success")
    except ValueError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
    except OSError as exc:
        flash(f"User saved to the database, but data/users.csv could not be updated ({exc}).", "warning")
    return redirect(url_for("users"))


@app.route("/users/directory/<user_id>/delete", methods=["POST"])
@role_required("Administrator")
def directory_user_delete(user_id):
    person = DirectoryUser.query.get_or_404(user_id)
    name = person.username
    # Old events and alerts are evidence: they are kept, only the link to the removed person is cleared.
    detached = SecurityEvent.query.filter_by(user_id=user_id).update({"user_id": None})
    Alert.query.filter_by(assigned_to=user_id).update({"assigned_to": None})
    Incident.query.filter_by(assigned_to=user_id).update({"assigned_to": None})
    db.session.delete(person)
    audit(current_user.username, "Removed directory user", "DirectoryUser", user_id,
          f"username={name}; events detached={detached}")
    db.session.commit()
    try:
        dataset_remove("users.csv", "user_id", user_id)
    except OSError as exc:
        flash(f"Removed from the database, but data/users.csv could not be updated ({exc}).", "warning")
    flash(f"{user_id} ({name}) was removed. {detached} event(s) were kept and detached from this user.", "success")
    return redirect(url_for("users"))


@app.route("/assets")
@role_required("Administrator", "Analyst", "Incident Manager")
def assets():
    return render_template("assets.html", items=Asset.query.order_by(Asset.asset_id).all(),
                           event_counts=_counts(SecurityEvent.asset_id), can_manage=current_user.role == "Administrator",
                           criticality_levels=CRITICALITY_LEVELS, asset_types=ASSET_TYPES, environments=ENVIRONMENTS,
                           suggested_asset_id=next_id([r[0] for r in db.session.query(Asset.asset_id).all()], "AST", 3))


@app.route("/assets/add", methods=["POST"])
@role_required("Administrator")
def asset_add():
    form = request.form
    try:
        asset_id = form.get("asset_id", "").strip().upper() or next_id(
            [r[0] for r in db.session.query(Asset.asset_id).all()], "AST", 3)
        name = form.get("asset_name", "").strip()
        criticality = form.get("criticality", "")
        environment = form.get("environment", "").strip()
        asset_type = form.get("asset_type", "").strip()
        if not ID_PATTERN.fullmatch(asset_id):
            raise ValueError("Asset ID must be 2-20 letters, numbers, dash or underscore.")
        if not name or len(name) > 100:
            raise ValueError("Asset name is required (100 characters at most).")
        if not asset_type or not environment:
            raise ValueError("Asset type and environment are required.")
        try:
            ip_address = str(ipaddress.ip_address(form.get("ip_address", "").strip()))
        except ValueError:
            raise ValueError("IP address is not valid.")
        if criticality not in CRITICALITY_LEVELS:
            raise ValueError("Choose a valid criticality.")
        if db.session.get(Asset, asset_id):
            raise ValueError(f"Asset ID {asset_id} already exists.")
        record = {"asset_id": asset_id, "asset_name": name, "asset_type": asset_type, "ip_address": ip_address,
                  "criticality": criticality, "environment": environment}
        db.session.add(Asset(**record))
        audit(current_user.username, "Added asset", "Asset", asset_id, f"name={name}; ip={ip_address}; criticality={criticality}")
        db.session.commit()
        dataset_append("assets.csv", record)
        flash(f"Asset {asset_id} ({name}) added.", "success")
    except ValueError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
    except OSError as exc:
        flash(f"Asset saved to the database, but data/assets.csv could not be updated ({exc}).", "warning")
    return redirect(url_for("assets"))


@app.route("/assets/<asset_id>/delete", methods=["POST"])
@role_required("Administrator")
def asset_delete(asset_id):
    asset = Asset.query.get_or_404(asset_id)
    name = asset.asset_name
    detached = SecurityEvent.query.filter_by(asset_id=asset_id).update({"asset_id": None})
    db.session.delete(asset)
    audit(current_user.username, "Removed asset", "Asset", asset_id, f"name={name}; events detached={detached}")
    db.session.commit()
    try:
        dataset_remove("assets.csv", "asset_id", asset_id)
    except OSError as exc:
        flash(f"Removed from the database, but data/assets.csv could not be updated ({exc}).", "warning")
    flash(f"Asset {asset_id} ({name}) was removed. {detached} event(s) were kept and detached from this asset.", "success")
    return redirect(url_for("assets"))

@app.route("/audit")
@role_required("Administrator","Incident Manager")
def audit_logs():
    items = AuditLog.query.order_by(AuditLog.created_at.desc()).limit(300).all()
    return render_template("audit.html", items=items)

@app.errorhandler(403)
def forbidden(_):
    return render_template("error.html", code=403, message="You do not have permission to access this page."), 403

@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, message="The requested record or page was not found."), 404

with app.app_context():
    db.create_all()
    otp_service.ensure_schema()
    create_demo_accounts()
    otp_service.seed_account_emails()
    import_starter_data()
    try:    # events that are in the dataset file but have no prediction yet (e.g. after the database was rebuilt)
        score_unscored_events(unlabelled_only=True)
    except Exception as exc:
        db.session.rollback()
        print(f"[SentinelOps] start-up scoring skipped: {exc}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT", 9000))
    app.run(host="0.0.0.0", port=port, debug=False)
