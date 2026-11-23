import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.engine import URL
import certifi

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


# --------------------------------------------------
# Database Configuration
# --------------------------------------------------

TIDB_HOST = os.getenv("TIDB_HOST", "").strip()

if TIDB_HOST:

    required = ["TIDB_USER", "TIDB_PASSWORD"]

    missing = [
        key for key in required
        if not os.getenv(key)
    ]

    if missing:
        raise RuntimeError(
            f"Missing TiDB environment variables: {', '.join(missing)}"
        )

    DATABASE_URI = URL.create(
        drivername="mysql+pymysql",
        username=os.getenv("TIDB_USER"),
        password=os.getenv("TIDB_PASSWORD"),
        host=TIDB_HOST,
        port=int(os.getenv("TIDB_PORT", "4000")),
        database=os.getenv("TIDB_DB_NAME", "sentinelopsai")
    )

    ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "connect_args": {
            "ssl_ca": certifi.where(),
            "ssl_verify_cert": True,
            "ssl_verify_identity": True,
            "connect_timeout": 15
        }
    }

else:
    DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{ROOT / 'sentinelops_ai.db'}"
    )
    ENGINE_OPTIONS = {}


class Config:

    SECRET_KEY = os.getenv(
        "SECRET_KEY",
        "dev-only-change-this-secret"
    )

    SQLALCHEMY_DATABASE_URI = DATABASE_URI
    SQLALCHEMY_ENGINE_OPTIONS = ENGINE_OPTIONS
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAX_CONTENT_LENGTH = 10 * 1024 * 1024

    TRAIN_ON_PREDICTED = (
        os.getenv("TRAIN_ON_PREDICTED", "1") == "1"
    )

    # Email OTP configuration

    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

    MAIL_FROM = (
        os.getenv("MAIL_FROM", "")
        or os.getenv("SMTP_USERNAME", "")
    )

    OTP_LENGTH = 6
    OTP_TTL_SECONDS = int(os.getenv("OTP_TTL_SECONDS", "300"))
    OTP_MAX_ATTEMPTS = int(os.getenv("OTP_MAX_ATTEMPTS", "5"))
    OTP_RESEND_SECONDS = int(os.getenv("OTP_RESEND_SECONDS", "30"))

    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")

    SIGNUP_DEFAULT_ROLE = os.getenv(
        "SIGNUP_DEFAULT_ROLE", "Analyst"
    )

    ANALYST_EMAIL = os.getenv("ANALYST_EMAIL", "")
    MANAGER_EMAIL = os.getenv("MANAGER_EMAIL", "")

    OTP_CONSOLE_FALLBACK = (
        os.getenv("OTP_CONSOLE_FALLBACK", "1") == "1"
    )