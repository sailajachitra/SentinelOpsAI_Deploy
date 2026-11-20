"""Email one-time-password (OTP) helpers: generate, store (hashed), email, verify."""
import hashlib
import hmac
import secrets
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from flask import current_app
from sqlalchemy import inspect, text
from extensions import db
from models import AppUser, EmailOTP


def _hash(code: str, user_id: int, purpose: str = "login") -> str:
    key = current_app.config["SECRET_KEY"].encode()
    return hmac.new(key, f"{purpose}:{user_id}:{code}".encode(), hashlib.sha256).hexdigest()


def find_user_by_login(identifier: str):
    """Look a user up by email or username (case-insensitive)."""
    ident = (identifier or "").strip().lower()
    if not ident:
        return None
    col = AppUser.email if "@" in ident else AppUser.username
    return AppUser.query.filter(db.func.lower(col) == ident).first()


def find_user_by_email(email: str):
    """Look a user up by email address only (case-insensitive)."""
    email = (email or "").strip().lower()
    if not email:
        return None
    return AppUser.query.filter(db.func.lower(AppUser.email) == email).first()


def mask_email(email: str) -> str:
    name, _, domain = email.partition("@")
    return (name[:2] + "*" * max(len(name) - 2, 1)) + "@" + domain


def ensure_schema():
    """db.create_all() does not add columns to existing tables, so add app_users.email if missing."""
    cols = {c["name"] for c in inspect(db.engine).get_columns("app_users")}
    if "email" not in cols:
        db.session.execute(text("ALTER TABLE app_users ADD COLUMN email VARCHAR(255)"))
        db.session.commit()
    if "email_verified" not in cols:      # accounts that existed before sign-up was added are already verified
        db.session.execute(text("ALTER TABLE app_users ADD COLUMN email_verified BOOLEAN NOT NULL DEFAULT 1"))
        db.session.commit()


def seed_account_emails():
    """Give the built-in accounts the emails configured in .env if they have none yet.
    Sign-in is by email code only, so an account without an email cannot sign in."""
    cfg = current_app.config
    for username, key in (("admin", "ADMIN_EMAIL"), ("analyst", "ANALYST_EMAIL"), ("manager", "MANAGER_EMAIL")):
        email = (cfg.get(key) or "").strip().lower()
        acct = AppUser.query.filter_by(username=username).first()
        if acct and email and not acct.email and not AppUser.query.filter_by(email=email).first():
            acct.email = email
    db.session.commit()


def _send_email(to_addr: str, code: str, minutes: int, purpose: str = "login") -> bool:
    cfg = current_app.config
    if not (cfg["SMTP_USERNAME"] and cfg["SMTP_PASSWORD"]):
        if cfg["OTP_CONSOLE_FALLBACK"]:
            print(f"[SentinelOps] SMTP not configured - {purpose} code for {to_addr}: {code}")
            return True
        return False
    msg = EmailMessage()
    reset = purpose == "reset"
    msg["Subject"] = "Your SentinelOps AI password reset code" if reset else "Your SentinelOps AI sign-in code"
    msg["From"] = cfg["MAIL_FROM"] or cfg["SMTP_USERNAME"]
    msg["To"] = to_addr
    msg.set_content(
        f"Your SentinelOps AI {'password reset' if reset else 'verification'} code is {code}.\n\n"
        f"It expires in {minutes} minutes. If you did not request this, ignore this email "
        f"and tell your SOC administrator."
    )
    try:
        with smtplib.SMTP(cfg["SMTP_HOST"], cfg["SMTP_PORT"], timeout=15) as smtp:
            smtp.starttls()
            smtp.login(cfg["SMTP_USERNAME"], cfg["SMTP_PASSWORD"])
            smtp.send_message(msg)
        return True
    except (smtplib.SMTPException, OSError) as exc:
        print(f"[SentinelOps] could not send OTP email: {exc}")
        return False


def seconds_until_resend(user: AppUser, purpose: str = "login") -> int:
    last = (EmailOTP.query.filter_by(user_id=user.id, purpose=purpose).order_by(EmailOTP.created_at.desc()).first())
    if not last:
        return 0
    wait = current_app.config["OTP_RESEND_SECONDS"] - (datetime.utcnow() - last.created_at).total_seconds()
    return max(int(wait) + (1 if wait > 0 else 0), 0)


def issue_otp(user: AppUser, ip: str = "", purpose: str = "login") -> bool:
    """Create a new code, invalidate older ones, email it. Returns True if the email went out."""
    cfg = current_app.config
    now = datetime.utcnow()
    EmailOTP.query.filter_by(user_id=user.id, purpose=purpose, consumed_at=None).update({"consumed_at": now})
    code = "".join(str(secrets.randbelow(10)) for _ in range(cfg["OTP_LENGTH"]))
    db.session.add(EmailOTP(
        user_id=user.id, email=user.email, code_hash=_hash(code, user.id, purpose), purpose=purpose,
        expires_at=now + timedelta(seconds=cfg["OTP_TTL_SECONDS"]), request_ip=(ip or "")[:45]))
    db.session.commit()
    return _send_email(user.email, code, cfg["OTP_TTL_SECONDS"] // 60, purpose)


def verify_otp(user: AppUser, code: str, purpose: str = "login"):
    """Returns (ok, message). Burns the code after too many wrong tries or on success."""
    cfg = current_app.config
    now = datetime.utcnow()
    otp = (EmailOTP.query.filter_by(user_id=user.id, purpose=purpose, consumed_at=None)
           .order_by(EmailOTP.created_at.desc()).first())
    if not otp:
        return False, "No active code. Please request a new one."
    if otp.expires_at < now:
        otp.consumed_at = now
        db.session.commit()
        return False, "That code has expired. Please request a new one."
    otp.attempts += 1
    if hmac.compare_digest(otp.code_hash, _hash((code or "").strip(), user.id, purpose)):
        otp.consumed_at = now
        db.session.commit()
        return True, ""
    if otp.attempts >= cfg["OTP_MAX_ATTEMPTS"]:
        otp.consumed_at = now
        db.session.commit()
        return False, "Too many wrong attempts. Please request a new code."
    db.session.commit()
    return False, f"Incorrect code. {cfg['OTP_MAX_ATTEMPTS'] - otp.attempts} attempt(s) left."
