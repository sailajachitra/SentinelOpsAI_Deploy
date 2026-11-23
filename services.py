
"""Dataset ingestion and audit helpers implemented with SQLAlchemy ORM."""

import io
import threading
from pathlib import Path
from datetime import datetime

import pandas as pd
from werkzeug.utils import secure_filename

from extensions import db
from models import (
    DirectoryUser,
    Asset,
    SecurityEvent,
    Alert,
    Incident,
    AuditLog,
    IncidentTimeline,
    InvestigationNote,
    MLModelRegistry,
)


DATA_DIR = Path(__file__).resolve().parent / "data"


def audit(username, action, entity_type="", entity_id="", details=""):
    db.session.add(
        AuditLog(
            username=username or "system",
            action=action,
            entity_type=entity_type,
            entity_id=str(entity_id),
            details=details,
        )
    )


# ---------------------------------------------------------------------------
# Dataset files (data/*.csv)
# ---------------------------------------------------------------------------

EVENT_DATASET_COLUMNS = [
    "event_id", "event_time", "user_id", "asset_id",
    "src_ip", "dst_ip", "event_type", "protocol",
    "dst_port", "failed_login_count", "bytes_sent",
    "severity_hint", "label", "attack_type"
]

_dataset_lock = threading.Lock()


def _read_dataset(filename):
    """Read a dataset CSV as plain text."""
    path = DATA_DIR / filename
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    ), path


def _write_dataset(frame, path):
    """Write atomically to avoid partially written CSV files."""
    tmp = path.with_suffix(".tmp")

    frame.to_csv(
        tmp,
        index=False,
        lineterminator="\r\n"
    )

    tmp.replace(path)


def _text(value):
    if value is None:
        return ""

    if not isinstance(value, str) and pd.isna(value):
        return ""

    return str(value)


def event_dataset_row(event):
    """Convert a SecurityEvent into a CSV dataset row."""

    row = {
        column: _text(getattr(event, column, None))
        for column in EVENT_DATASET_COLUMNS
    }

    if row["label"] == "Benign" and not row["attack_type"]:
        row["attack_type"] = "None"

    return row


def dataset_upsert_events(events):
    """
    Add or update events in data/security_events.csv.
    Existing event IDs are updated rather than duplicated.
    """

    events = list(events)

    if not events:
        return 0

    with _dataset_lock:

        frame, path = _read_dataset("security_events.csv")

        for column in EVENT_DATASET_COLUMNS:
            if column not in frame:
                frame[column] = ""

        index = {
            eid: i
            for i, eid in enumerate(frame["event_id"])
        }

        new_rows = []

        for event in events:

            row = event_dataset_row(event)

            if row["event_id"] in index:

                frame.loc[
                    index[row["event_id"]],
                    EVENT_DATASET_COLUMNS
                ] = [
                    row[c] for c in EVENT_DATASET_COLUMNS
                ]

            else:
                new_rows.append(row)

                # Avoid duplicates within the same incoming batch.
                index[row["event_id"]] = -1

        if new_rows:
            frame = pd.concat(
                [
                    frame[EVENT_DATASET_COLUMNS],
                    pd.DataFrame(new_rows)[EVENT_DATASET_COLUMNS]
                ],
                ignore_index=True
            )

        _write_dataset(
            frame[EVENT_DATASET_COLUMNS],
            path
        )

    return len(events)


def dataset_append(filename, row):
    """Append one record to data/users.csv or data/assets.csv."""

    with _dataset_lock:

        frame, path = _read_dataset(filename)

        record = {
            column: _text(row.get(column))
            for column in frame.columns
        }

        frame = pd.concat(
            [frame, pd.DataFrame([record])],
            ignore_index=True
        )

        _write_dataset(frame, path)


def dataset_remove(filename, key_column, key):
    """Remove a record from a dataset CSV."""

    with _dataset_lock:

        frame, path = _read_dataset(filename)

        keep = frame[key_column] != str(key)

        removed = int((~keep).sum())

        if removed:
            _write_dataset(frame[keep], path)

    return removed


def next_id(existing_ids, prefix, width):
    """Generate the next sequential ID."""

    numbers = [
        int(i[len(prefix):])
        for i in existing_ids
        if i
        and i.startswith(prefix)
        and i[len(prefix):].isdigit()
    ]

    return f"{prefix}{(max(numbers) if numbers else 0) + 1:0{width}d}"


def _records(filename):
    """Read CSV records and convert NaN values to None."""

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Starter dataset file not found: {path}"
        )

    frame = pd.read_csv(path)

    frame = frame.astype(object).where(
        pd.notna(frame),
        None
    )

    return frame.to_dict("records")


def _as_int(value):
    return (
        None
        if value is None or pd.isna(value)
        else int(value)
    )


def _as_float(value):
    return (
        None
        if value is None or pd.isna(value)
        else float(value)
    )


# ---------------------------------------------------------------------------
# Database batch helpers
# ---------------------------------------------------------------------------

def _chunks(values, size=500):
    """
    Divide values into smaller batches.

    A batch size of 500 also keeps queries manageable
    when running against local SQLite during development.
    """

    values = list(values)

    for start in range(0, len(values), size):
        yield values[start:start + size]


def _existing_ids(model, column, values):
    """
    Fetch existing IDs using batched SELECT queries.
    Avoids one database request per CSV record.
    """

    values = list(dict.fromkeys(
        str(value)
        for value in values
        if value is not None
    ))

    existing = set()

    if not values:
        return existing

    for batch in _chunks(values):

        rows = (
            db.session.query(column)
            .filter(column.in_(batch))
            .all()
        )

        existing.update(
            str(row[0])
            for row in rows
        )

    return existing


def _fetch_by_ids(model, column, values):
    """Fetch ORM records in batches and return an ID-to-record dictionary."""

    values = list(dict.fromkeys(
        str(value)
        for value in values
        if value is not None
    ))

    records = {}

    if not values:
        return records

    for batch in _chunks(values):

        rows = (
            db.session.query(model)
            .filter(column.in_(batch))
            .all()
        )

        for record in rows:

            key = str(
                getattr(record, column.key)
            )

            records[key] = record

    return records


# ---------------------------------------------------------------------------
# Optimized starter dataset import
# ---------------------------------------------------------------------------

def import_starter_data():
    """
    Import starter dataset records efficiently.

    Features:
    - Batch database queries.
    - Insert only missing records.
    - Preserve existing cloud database records.
    - Avoid duplicate CSV records during import.
    - Maintain foreign-key relationships.
    - Roll back if the import fails.
    """

    added = {
        "users": 0,
        "assets": 0,
        "security_events": 0,
        "alerts": 0,
        "incidents": 0,
        "incident_timeline": 0,
        "investigation_notes": 0,
    }

    try:

        # ------------------------------------------------
        # 1. Import directory users
        # ------------------------------------------------

        user_rows = _records("users.csv")

        existing_users = _existing_ids(
            DirectoryUser,
            DirectoryUser.user_id,
            [row["user_id"] for row in user_rows]
        )

        new_users = []

        for row in user_rows:

            key = str(row["user_id"])

            if key not in existing_users:

                existing_users.add(key)

                new_users.append(
                    DirectoryUser(**row)
                )

        db.session.add_all(new_users)

        added["users"] = len(new_users)

        db.session.flush()

        # ------------------------------------------------
        # 2. Import assets
        # ------------------------------------------------

        asset_rows = _records("assets.csv")

        existing_assets = _existing_ids(
            Asset,
            Asset.asset_id,
            [row["asset_id"] for row in asset_rows]
        )

        new_assets = []

        for row in asset_rows:

            key = str(row["asset_id"])

            if key not in existing_assets:

                existing_assets.add(key)

                new_assets.append(
                    Asset(**row)
                )

        db.session.add_all(new_assets)

        added["assets"] = len(new_assets)

        db.session.flush()

        # ------------------------------------------------
        # 3. Get valid user and asset IDs
        # ------------------------------------------------

        known_users = {
            row[0]
            for row in db.session.query(
                DirectoryUser.user_id
            ).all()
        }

        known_assets = {
            row[0]
            for row in db.session.query(
                Asset.asset_id
            ).all()
        }

        # ------------------------------------------------
        # 4. Import security events
        # ------------------------------------------------

        event_rows = _records("security_events.csv")

        existing_events = _existing_ids(
            SecurityEvent,
            SecurityEvent.event_id,
            [row["event_id"] for row in event_rows]
        )

        new_events = []

        for row in event_rows:

            key = str(row["event_id"])

            if key in existing_events:
                continue

            existing_events.add(key)

            row["event_id"] = key

            if row.get("user_id") not in known_users:
                row["user_id"] = None

            if row.get("asset_id") not in known_assets:
                row["asset_id"] = None

            for field in (
                "dst_port",
                "failed_login_count",
                "bytes_sent"
            ):
                row[field] = _as_int(row.get(field))

            new_events.append(
                SecurityEvent(**row)
            )

        db.session.add_all(new_events)

        added["security_events"] = len(new_events)

        db.session.flush()

        # ------------------------------------------------
        # 5. Import alerts
        # ------------------------------------------------

        alert_rows = _records("seed_alerts.csv")

        existing_alerts = _existing_ids(
            Alert,
            Alert.alert_id,
            [row["alert_id"] for row in alert_rows]
        )

        new_alerts = []

        for row in alert_rows:

            key = str(row["alert_id"])

            if key not in existing_alerts:

                existing_alerts.add(key)

                row["alert_id"] = key

                row["model_score"] = _as_float(
                    row.get("model_score")
                )

                new_alerts.append(
                    Alert(**row)
                )

        db.session.add_all(new_alerts)

        added["alerts"] = len(new_alerts)

        db.session.flush()

        # ------------------------------------------------
        # 6. Import incidents
        # ------------------------------------------------

        incident_rows = _records("seed_incidents.csv")

        existing_incidents = _existing_ids(
            Incident,
            Incident.incident_id,
            [row["incident_id"] for row in incident_rows]
        )

        new_incidents = []

        for row in incident_rows:

            key = str(row["incident_id"])

            if key not in existing_incidents:

                existing_incidents.add(key)

                row["incident_id"] = key

                new_incidents.append(
                    Incident(**row)
                )

        db.session.add_all(new_incidents)

        added["incidents"] = len(new_incidents)

        db.session.flush()

        # ------------------------------------------------
        # 7. Fetch existing incidents
        # ------------------------------------------------

        incidents = (
            Incident.query
            .order_by(Incident.incident_id)
            .all()
        )

        incident_ids = [
            incident.incident_id
            for incident in incidents
        ]

        # ------------------------------------------------
        # 8. Find existing starter timeline entries
        # ------------------------------------------------

        timeline_exists = set()

        for batch in _chunks(incident_ids):

            rows = (
                db.session.query(
                    IncidentTimeline.incident_id
                )
                .filter(
                    IncidentTimeline.incident_id.in_(batch),
                    IncidentTimeline.action
                    == "Starter incident imported"
                )
                .all()
            )

            timeline_exists.update(
                str(row[0])
                for row in rows
            )

        # ------------------------------------------------
        # 9. Find existing starter investigation notes
        # ------------------------------------------------

        notes_exists = set()

        for batch in _chunks(incident_ids):

            rows = (
                db.session.query(
                    InvestigationNote.incident_id
                )
                .filter(
                    InvestigationNote.incident_id.in_(batch),
                    InvestigationNote.note_type
                    == "Starter evidence"
                )
                .all()
            )

            notes_exists.update(
                str(row[0])
                for row in rows
            )

        # ------------------------------------------------
        # 10. Fetch related alerts in batches
        # ------------------------------------------------

        related_alert_ids = {
            incident.related_alert_id
            for incident in incidents
            if incident.related_alert_id
        }

        alerts_by_id = _fetch_by_ids(
            Alert,
            Alert.alert_id,
            related_alert_ids
        )

        related_event_ids = {
            alert.event_id
            for alert in alerts_by_id.values()
            if alert.event_id
        }

        events_by_id = _fetch_by_ids(
            SecurityEvent,
            SecurityEvent.event_id,
            related_event_ids
        )

        # ------------------------------------------------
        # 11. Create missing timeline and evidence records
        # ------------------------------------------------

        new_timelines = []
        new_notes = []

        for incident in incidents:

            incident_id = str(incident.incident_id)

            if incident_id not in timeline_exists:

                new_timelines.append(
                    IncidentTimeline(
                        incident_id=incident.incident_id,
                        action="Starter incident imported",
                        description=(
                            "Incident record loaded from the "
                            "supplied SentinelOps AI starter dataset."
                        ),
                        performed_by="system",
                    )
                )

                timeline_exists.add(incident_id)

            if incident_id not in notes_exists:

                linked_alert = (
                    alerts_by_id.get(
                        str(incident.related_alert_id)
                    )
                    if incident.related_alert_id
                    else None
                )

                linked_event = (
                    events_by_id.get(
                        str(linked_alert.event_id)
                    )
                    if linked_alert and linked_alert.event_id
                    else None
                )

                evidence_text = (
                    f"Starter incident context. "
                    f"Root cause: "
                    f"{incident.root_cause or 'Not supplied'}. "
                    f"Resolution notes: "
                    f"{incident.resolution_notes or 'Not supplied'}. "
                )

                if linked_alert:

                    evidence_text += (
                        f"Related alert: "
                        f"{linked_alert.alert_id} "
                        f"({linked_alert.alert_title}). "
                    )

                if linked_event:

                    evidence_text += (
                        f"Related event: "
                        f"{linked_event.event_id}; "
                        f"type={linked_event.event_type}; "
                        f"label={linked_event.label or 'Unlabelled'}; "
                        f"attack_type="
                        f"{linked_event.attack_type or 'N/A'}."
                    )

                new_notes.append(
                    InvestigationNote(
                        incident_id=incident.incident_id,
                        alert_id=incident.related_alert_id,
                        event_id=(
                            linked_event.event_id
                            if linked_event
                            else None
                        ),
                        author="system",
                        note_type="Starter evidence",
                        note_text=evidence_text,
                    )
                )

                notes_exists.add(incident_id)

        db.session.add_all(new_timelines)
        db.session.add_all(new_notes)

        added["incident_timeline"] = len(new_timelines)
        added["investigation_notes"] = len(new_notes)

        # ------------------------------------------------
        # 12. Save all imported records
        # ------------------------------------------------

        if any(added.values()):

            audit(
                "system",
                "Imported starter dataset using optimized SQLAlchemy ORM",
                "Dataset",
                "SentinelOps_AI_Starter_Dataset",
                "; ".join(
                    f"{name}={count}"
                    for name, count in added.items()
                ),
            )

        db.session.commit()

        print(
            "\n[SentinelOps] Starter dataset import completed."
        )

        print(
            "[SentinelOps] Import summary:",
            added
        )

        return added

    except Exception:

        db.session.rollback()

        raise


# ---------------------------------------------------------------------------
# ML model evaluation storage
# ---------------------------------------------------------------------------

def save_model_evaluations(
    evaluation,
    dataset_name="data/security_events.csv"
):
    """Persist model evaluation metrics in the ML registry."""

    results = evaluation.get("results", [])

    selected = evaluation.get("selected_model")

    for result in results:

        name = result.get(
            "model",
            "Unknown model"
        )

        row = MLModelRegistry.query.filter_by(
            model_name=name
        ).first()

        if row is None:

            row = MLModelRegistry(
                model_name=name
            )

            db.session.add(row)

        row.model_version = "1.0"

        row.model_type = "Binary malicious-event classifier"

        row.training_dataset = dataset_name

        row.accuracy = _as_float(
            result.get("accuracy")
        )

        row.precision = _as_float(
            result.get("precision")
        )

        row.recall = _as_float(
            result.get("recall")
        )

        row.f1_score = _as_float(
            result.get("f1")
        )

        row.is_active = name == selected

        row.trained_at = datetime.utcnow()

        row.notes = (
            f"Selected model: {selected}. "
            "Evaluation is based on the supplied synthetic starter dataset."
        )

    db.session.commit()


# ---------------------------------------------------------------------------
# CSV event import
# ---------------------------------------------------------------------------

def import_event_csv(file_storage, username):
    """Import new security events from an uploaded CSV file."""

    filename = secure_filename(
        file_storage.filename or ""
    )

    if not filename.lower().endswith(".csv"):
        raise ValueError("Please upload a CSV file.")

    raw = file_storage.read()

    frame = pd.read_csv(
        io.BytesIO(raw)
    )

    required = {
        "event_id",
        "event_time",
        "user_id",
        "asset_id",
        "src_ip",
        "dst_ip",
        "event_type",
        "protocol",
        "dst_port",
        "failed_login_count",
        "bytes_sent",
        "severity_hint",
        "label",
        "attack_type",
    }

    missing = required - set(frame.columns)

    if missing:

        raise ValueError(
            "CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    added = 0
    skipped = 0

    added_ids = []
    added_events = []

    # Fetch known user and asset IDs once.
    known_users = {
        row[0]
        for row in db.session.query(
            DirectoryUser.user_id
        ).all()
    }

    known_assets = {
        row[0]
        for row in db.session.query(
            Asset.asset_id
        ).all()
    }

    allowed = {
        column.name
        for column in SecurityEvent.__table__.columns
    }

    # Fetch existing event IDs in batches before processing CSV.
    incoming_ids = frame["event_id"].astype(str).tolist()

    existing_events = _existing_ids(
        SecurityEvent,
        SecurityEvent.event_id,
        incoming_ids
    )

    seen_ids = set(existing_events)

    for row in frame.to_dict("records"):

        event_id = str(row["event_id"])

        if event_id in seen_ids:

            skipped += 1

            continue

        seen_ids.add(event_id)

        row["event_id"] = event_id

        for key in (
            "dst_port",
            "failed_login_count",
            "bytes_sent"
        ):

            row[key] = _as_int(
                row.get(key)
            )

        for key, value in list(row.items()):

            if pd.isna(value):
                row[key] = None

        row = {
            key: value
            for key, value in row.items()
            if key in allowed
        }

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

    audit(
        username,
        "Imported event CSV",
        "SecurityEvent",
        "",
        f"Added={added}; skipped={skipped}; file={filename}",
    )

    db.session.commit()

    # Keep the dataset CSV synchronized.
    dataset_upsert_events(added_events)

    return added, skipped, added_ids
