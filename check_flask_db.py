from app import app
from extensions import db
from sqlalchemy import text

with app.app_context():

    print("\nChecking Flask database connection...")

    dialect = db.engine.dialect.name

    print("Database type:", dialect)

    if dialect == "mysql":

        result = db.session.execute(
            text("SELECT DATABASE(), VERSION()")
        ).fetchone()

        print("Database name:", result[0])
        print("Database version:", result[1])

        print("\nFlask is connected to MySQL-compatible TiDB!")

    elif dialect == "sqlite":

        print("\nWARNING: Flask is using local SQLite.")

    else:

        print("\nUnexpected database:", dialect)