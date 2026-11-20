import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-this-secret")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{ROOT / 'sentinelops_ai.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    # Retrain on new events that only have a model prediction (their predicted label is used as a provisional
    # label for TRAINING only; held-out test metrics always use real ground truth). Set to 0 to train on
    # analyst-confirmed labels only.
    TRAIN_ON_PREDICTED = os.getenv("TRAIN_ON_PREDICTED", "1") == "1"

    # ---- Email OTP (two-step login) ----
    # Gmail: turn on 2-Step Verification, create an "App password" and put it in SMTP_PASSWORD.
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    MAIL_FROM = os.getenv("MAIL_FROM", "") or os.getenv("SMTP_USERNAME", "")
    OTP_LENGTH = 6
    OTP_TTL_SECONDS = int(os.getenv("OTP_TTL_SECONDS", "300"))        # code valid for 5 minutes
    OTP_MAX_ATTEMPTS = int(os.getenv("OTP_MAX_ATTEMPTS", "5"))        # wrong tries before the code is burned
    OTP_RESEND_SECONDS = int(os.getenv("OTP_RESEND_SECONDS", "30"))   # minimum gap between codes
    # Email set on the built-in "admin" account the first time it has none.
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "sailajadinesh17@gmail.com")
    # Role given to people who create their own account on the sign-up page (an Administrator can change it).
    SIGNUP_DEFAULT_ROLE = os.getenv("SIGNUP_DEFAULT_ROLE", "Analyst")
    # Optional: emails for the other two built-in accounts (otherwise add them under User Management).
    ANALYST_EMAIL = os.getenv("ANALYST_EMAIL", "")
    MANAGER_EMAIL = os.getenv("MANAGER_EMAIL", "")
    # If SMTP is not configured, print the code in the server console instead (development only).
    OTP_CONSOLE_FALLBACK = os.getenv("OTP_CONSOLE_FALLBACK", "1") == "1"
