"""Dataset ingestion and audit helpers implemented with SQLAlchemy ORM."""
import io
import threading
from pathlib import Path
from datetime import datetime

import pandas as pd
from werkzeug.utils import secure_filename
from extensions import db
from models import (
    DirectoryUser, Asset, SecurityEvent, Alert, Incident, AuditLog,
    IncidentTimeline, InvestigationNote, MLModelRegistry,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


def audit(username, action, entity_type="", entity_id="", details=""):
    db.session.add(AuditLog(
        username=username or "system", action=action,
        entity_type=entity_type, entity_id=str(entity_id), details=details,
    ))


# ---------------------------------------------------------------------------
# Dataset files (data/*.csv).  Everything entered or removed in the application is
# written through to these files so the "dataset" always matches the database, and a
# restart (or a rebuilt database) does not bring removed records back.
# ---------------------------------------------------------------------------
EVENT_DATASET_COLUMNS = ["event_id", "event_time", "user_id", "asset_id", "src_ip", "dst_ip", "event_type",
                         "protocol", "dst_port", "failed_login_count", "bytes_sent", "severity_hint",
                         "label", "attack_type"]
_dataset_lock = threading.Lock()


def _read_dataset(filename):
    """Read a dataset CSV as plain text (so 'None', ports and IDs are written back exactly as they were)."""
    path = DATA_DIR / filename
    return pd.read_csv(path, dtype=str, keep_default_na=False), path


def _write_dataset(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\r\n")
    tmp.replace(path)                       # atomic: a crash can never leave a half-written dataset


def _text(value):
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)


def event_dataset_row(event):
    """One SecurityEvent as a dataset row. Unknown ground truth is left blank; Benign uses 'None' like the starter file."""
    row = {column: _text(getattr(event, column, None)) for column in EVENT_DATASET_COLUMNS}
    if row["label"] == "Benign" and not row["attack_type"]:
        row["attack_type"] = "None"
    return row


def dataset_upsert_events(events):
    """Add events to data/security_events.csv; an event_id that is already there is updated (label / attack_type)."""
    events = list(events)
    if not events:
        return 0
    with _dataset_lock:
        frame, path = _read_dataset("security_events.csv")
        for column in EVENT_DATASET_COLUMNS:
            if column not in frame:
                frame[column] = ""
        index = {eid: i for i, eid in enumerate(frame["event_id"])}
        new_rows = []
        for event in events:
            row = event_dataset_row(event)
            if row["event_id"] in index:
                frame.loc[index[row["event_id"]], EVENT_DATASET_COLUMNS] = [row[c] for c in EVENT_DATASET_COLUMNS]
            else:
                new_rows.append(row)
        if new_rows:
            frame = pd.concat([frame[EVENT_DATASET_COLUMNS], pd.DataFrame(new_rows)[EVENT_DATASET_COLUMNS]],
                              ignore_index=True)
        _write_dataset(frame[EVENT_DATASET_COLUMNS], path)
    return len(events)


def dataset_append(filename, row):
    """Append one record (dict) to data/users.csv or data/assets.csv."""
    with _dataset_lock:
        frame, path = _read_dataset(filename)
        record = {column: _text(row.get(column)) for column in frame.columns}
        frame = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
        _write_dataset(frame, path)


def dataset_remove(filename, key_column, key):
    """Delete one record from a dataset CSV. Returns how many rows were removed."""
    with _dataset_lock:
        frame, path = _read_dataset(filename)
        keep = frame[key_column] != str(key)
        removed = int((~keep).sum())
        if removed:
            _write_dataset(frame[keep], path)
    return removed


def next_id(existing_ids, prefix, width):
    """Next free sequential ID: ('USR001','USR050') -> 'USR051'. Ignores IDs that do not follow the pattern."""
    numbers = [int(i[len(prefix):]) for i in existing_ids if i and i.startswith(prefix) and i[len(prefix):].isdigit()]
    return f"{prefix}{(max(numbers) if numbers else 0) + 1:0{width}d}"


def _records(filename):
    """Read CSV rows and convert pandas NaN values to Python None."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Starter dataset file not found: {path}")
    frame = pd.read_csv(path)
    frame = frame.astype(object).where(pd.notna(frame), None)
    return frame.to_dict("records")


def _as_int(value):
    return None if value is None or pd.isna(value) else int(value)


def _as_float(value):
    return None if value is None or pd.isna(value) else float(value)


def import_starter_data():
    """Idempotently import every supplied dataset CSV through ORM models.

    Missing primary keys are inserted even if a table already contains some rows.
    Existing records are preserved, so restarting the app does not duplicate data.
    """
    added = {"users": 0, "assets": 0, "security_events": 0,
             "alerts": 0, "incidents": 0, "incident_timeline": 0,
             "investigation_notes": 0}

    # 1. Directory users must exist before security events and alert assignments.
    for row in _records("users.csv"):
        key = str(row["user_id"])
        if db.session.get(DirectoryUser, key) is None:
            db.session.add(DirectoryUser(**row))
            added["users"] += 1
    db.session.flush()

    # 2. Assets are referenced by security events.
    for row in _records("assets.csv"):
        key = str(row["asset_id"])
        if db.session.get(Asset, key) is None:
            db.session.add(Asset(**row))
            added["assets"] += 1
    db.session.flush()

    # 3. Import all dataset events (labelled starter rows and events entered in the application).
    known_users = {u for (u,) in db.session.query(DirectoryUser.user_id).all()}
    known_assets = {a for (a,) in db.session.query(Asset.asset_id).all()}
    for row in _records("security_events.csv"):
        key = str(row["event_id"])
        if db.session.get(SecurityEvent, key) is None:
            row["event_id"] = key
            # a user / asset removed by an administrator must not break the foreign key of old events
            if row.get("user_id") not in known_users:
                row["user_id"] = None
            if row.get("asset_id") not in known_assets:
                row["asset_id"] = None
            for field in ("dst_port", "failed_login_count", "bytes_sent"):
                row[field] = _as_int(row.get(field))
            db.session.add(SecurityEvent(**row))
            added["security_events"] += 1
    db.session.flush()

    # 4. Seed alerts after their referenced events exist.
    for row in _records("seed_alerts.csv"):
        key = str(row["alert_id"])
        if db.session.get(Alert, key) is None:
            row["alert_id"] = key
            row["model_score"] = _as_float(row.get("model_score"))
            db.session.add(Alert(**row))
            added["alerts"] += 1
    db.session.flush()

    # 5. Seed incidents after their referenced alerts exist.
    for row in _records("seed_incidents.csv"):
        key = str(row["incident_id"])
        if db.session.get(Incident, key) is None:
            row["incident_id"] = key
            db.session.add(Incident(**row))
            added["incidents"] += 1
    db.session.flush()

    # 6. Create a first timeline entry and an evidence note for each starter incident.
    for incident in Incident.query.order_by(Incident.incident_id).all():
        timeline_exists = IncidentTimeline.query.filter_by(
            incident_id=incident.incident_id,
            action="Starter incident imported",
        ).first()
        if timeline_exists is None:
            db.session.add(IncidentTimeline(
                incident_id=incident.incident_id,
                action="Starter incident imported",
                description="Incident record loaded from the supplied SentinelOps AI starter dataset.",
                performed_by="system",
            ))
            added["incident_timeline"] += 1

        note_exists = InvestigationNote.query.filter_by(
            incident_id=incident.incident_id,
            note_type="Starter evidence",
        ).first()
        if note_exists is None:
            linked_alert = db.session.get(Alert, incident.related_alert_id) if incident.related_alert_id else None
            linked_event = db.session.get(SecurityEvent, linked_alert.event_id) if linked_alert else None
            evidence_text = (
                f"Starter incident context. Root cause: {incident.root_cause or 'Not supplied'}. "
                f"Resolution notes: {incident.resolution_notes or 'Not supplied'}. "
            )
            if linked_alert:
                evidence_text += f"Related alert: {linked_alert.alert_id} ({linked_alert.alert_title}). "
            if linked_event:
                evidence_text += (
                    f"Related event: {linked_event.event_id}; type={linked_event.event_type}; "
                    f"label={linked_event.label or 'Unlabelled'}; attack_type={linked_event.attack_type or 'N/A'}."
                )
            db.session.add(InvestigationNote(
                incident_id=incident.incident_id,
                alert_id=incident.related_alert_id,
                event_id=linked_event.event_id if linked_event else None,
                author="system", note_type="Starter evidence", note_text=evidence_text,
            ))
            added["investigation_notes"] += 1

    if any(added.values()):
        audit("system", "Imported starter dataset using SQLAlchemy ORM",
              "Dataset", "SentinelOps_AI_Starter_Dataset",
              "; ".join(f"{name}={count}" for name, count in added.items()))
    db.session.commit()
    return added


def save_model_evaluations(evaluation, dataset_name="data/security_events.csv"):
    """Persist each model's held-out metrics in the ML model registry table."""
    results = evaluation.get("results", [])
    selected = evaluation.get("selected_model")
    for result in results:
        name = result.get("model", "Unknown model")
        row = MLModelRegistry.query.filter_by(model_name=name).first()
        if row is None:
            row = MLModelRegistry(model_name=name)
            db.session.add(row)
        row.model_version = "1.0"
        row.model_type = "Binary malicious-event classifier"
        row.training_dataset = dataset_name
        row.accuracy = _as_float(result.get("accuracy"))
        row.precision = _as_float(result.get("precision"))
        row.recall = _as_float(result.get("recall"))
        row.f1_score = _as_float(result.get("f1"))
        row.is_active = name == selected
        row.trained_at = datetime.utcnow()
        row.notes = f"Selected model: {selected}. Evaluation is based on the supplied synthetic starter dataset."
    db.session.commit()


def import_event_csv(file_storage, username):
    filename = secure_filename(file_storage.filename or "")
    if not filename.lower().endswith(".csv"):
        raise ValueError("Please upload a CSV file.")
    raw = file_storage.read()
    frame = pd.read_csv(io.BytesIO(raw))
    required = {"event_id", "event_time", "user_id", "asset_id", "src_ip", "dst_ip",
                "event_type", "protocol", "dst_port", "failed_login_count", "bytes_sent",
                "severity_hint", "label", "attack_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("CSV is missing columns: " + ", ".join(sorted(missing)))

    added, skipped = 0, 0
    added_ids, added_events = [], []
    known_users = {u for (u,) in db.session.query(DirectoryUser.user_id).all()}
    known_assets = {a for (a,) in db.session.query(Asset.asset_id).all()}
    allowed = {column.name for column in SecurityEvent.__table__.columns}
    for row in frame.to_dict("records"):
        event_id = str(row["event_id"])
        if db.session.get(SecurityEvent, event_id) is not None:
            skipped += 1
            continue
        row["event_id"] = event_id
        for key in ("dst_port", "failed_login_count", "bytes_sent"):
            row[key] = _as_int(row.get(key))
        for key, value in list(row.items()):
            if pd.isna(value):
                row[key] = None
        row = {key: value for key, value in row.items() if key in allowed}
        if row.get("user_id") not in known_users:
            row["user_id"] = None
        if row.get("asset_id") not in known_assets:
            row["asset_id"] = None
        event = SecurityEvent(**row)
        db.session.add(event)
        added_events.append(event)
        added_ids.append(event_id)
        added += 1
    db.session.flush()
    audit(username, "Imported event CSV", "SecurityEvent", "",
          f"Added={added}; skipped={skipped}; file={filename}")
    db.session.commit()
    dataset_upsert_events(added_events)          # imported events become part of the dataset too
    return added, skipped, added_ids
