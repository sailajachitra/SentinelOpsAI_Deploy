from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db

class AppUser(UserMixin, db.Model):
    __tablename__ = "app_users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(30), nullable=False, default="Analyst")
    active = db.Column(db.Boolean, default=True)
    email_verified = db.Column(db.Boolean, default=True, nullable=False)   # False only for a sign-up that has not entered its code yet
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class DirectoryUser(db.Model):
    __tablename__ = "users"
    user_id = db.Column(db.String(20), primary_key=True)
    username = db.Column(db.String(80))
    department = db.Column(db.String(80))
    role = db.Column(db.String(80))
    account_status = db.Column(db.String(30))
    created_date = db.Column(db.String(30))

class Asset(db.Model):
    __tablename__ = "assets"
    asset_id = db.Column(db.String(20), primary_key=True)
    asset_name = db.Column(db.String(100))
    asset_type = db.Column(db.String(80))
    ip_address = db.Column(db.String(45))
    criticality = db.Column(db.String(30))
    environment = db.Column(db.String(50))

class SecurityEvent(db.Model):
    __tablename__ = "security_events"
    event_id = db.Column(db.String(30), primary_key=True)
    event_time = db.Column(db.String(40))
    user_id = db.Column(db.String(20), db.ForeignKey("users.user_id"))
    asset_id = db.Column(db.String(20), db.ForeignKey("assets.asset_id"))
    src_ip = db.Column(db.String(45))
    dst_ip = db.Column(db.String(45))
    event_type = db.Column(db.String(80))
    protocol = db.Column(db.String(20))
    dst_port = db.Column(db.Integer)
    failed_login_count = db.Column(db.Integer)
    bytes_sent = db.Column(db.Integer)
    severity_hint = db.Column(db.String(30))
    label = db.Column(db.String(30))
    attack_type = db.Column(db.String(60))
    predicted_label = db.Column(db.String(30))
    predicted_attack_type = db.Column(db.String(60))
    model_score = db.Column(db.Float)
    ingested_at = db.Column(db.DateTime, default=datetime.utcnow)

class Alert(db.Model):
    __tablename__ = "alerts"
    alert_id = db.Column(db.String(30), primary_key=True)
    event_id = db.Column(db.String(30), db.ForeignKey("security_events.event_id"))
    alert_title = db.Column(db.String(200))
    model_score = db.Column(db.Float)
    severity = db.Column(db.String(30))
    status = db.Column(db.String(30), default="New")
    assigned_to = db.Column(db.String(20))
    is_false_positive_review_candidate = db.Column(db.String(10))
    ground_truth_label = db.Column(db.String(30))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Incident(db.Model):
    __tablename__ = "incidents"
    incident_id = db.Column(db.String(30), primary_key=True)
    incident_title = db.Column(db.String(200))
    related_alert_id = db.Column(db.String(30), db.ForeignKey("alerts.alert_id"))
    status = db.Column(db.String(30))
    priority = db.Column(db.String(30))
    assigned_to = db.Column(db.String(20))
    root_cause = db.Column(db.Text)
    resolution_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AuditLog(db.Model):
    __tablename__ = "audit_logs"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    action = db.Column(db.String(200))
    entity_type = db.Column(db.String(50))
    entity_id = db.Column(db.String(50))
    details = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class IncidentTimeline(db.Model):
    """Auditable sequence of actions and events attached to an incident."""
    __tablename__ = "incident_timeline"
    id = db.Column(db.Integer, primary_key=True)
    incident_id = db.Column(db.String(30), db.ForeignKey("incidents.incident_id"), nullable=False, index=True)
    action = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text)
    performed_by = db.Column(db.String(80))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class InvestigationNote(db.Model):
    """Analyst notes and findings recorded during alert/incident investigation."""
    __tablename__ = "investigation_notes"
    id = db.Column(db.Integer, primary_key=True)
    incident_id = db.Column(db.String(30), db.ForeignKey("incidents.incident_id"), nullable=True, index=True)
    alert_id = db.Column(db.String(30), db.ForeignKey("alerts.alert_id"), nullable=True, index=True)
    event_id = db.Column(db.String(30), db.ForeignKey("security_events.event_id"), nullable=True, index=True)
    author = db.Column(db.String(80), nullable=False)
    note_type = db.Column(db.String(40), default="Observation")
    note_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class MLModelRegistry(db.Model):
    """Stores model versions and evaluation details used by the application."""
    __tablename__ = "ml_model_registry"
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(120), nullable=False)
    model_version = db.Column(db.String(40), nullable=False, default="1.0")
    model_type = db.Column(db.String(60))
    training_dataset = db.Column(db.String(200))
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=False, nullable=False)
    trained_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    notes = db.Column(db.Text)


class EmailOTP(db.Model):
    """One-time login codes emailed to a user. Only a hash of the code is stored, never the code itself."""
    __tablename__ = "email_otps"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False, index=True)
    email = db.Column(db.String(255), nullable=False)
    code_hash = db.Column(db.String(64), nullable=False)
    purpose = db.Column(db.String(30), nullable=False, default="login")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    attempts = db.Column(db.Integer, default=0, nullable=False)
    consumed_at = db.Column(db.DateTime, nullable=True)
    request_ip = db.Column(db.String(45))
