import os
import certifi
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():

    try:
        connection = mysql.connector.connect(
            host=os.getenv("TIDB_HOST"),
            port=int(os.getenv("TIDB_PORT", "4000")),
            user=os.getenv("TIDB_USER"),
            password=os.getenv("TIDB_PASSWORD"),
            database=os.getenv("TIDB_DB_NAME"),
            ssl_ca=certifi.where(),
            ssl_verify_cert=True,
            ssl_verify_identity=True,
            use_pure=True
        )

        return connection

    except mysql.connector.Error as error:
        print("Database connection error:", error)
        raise