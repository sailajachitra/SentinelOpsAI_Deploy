from database import get_db_connection

connection = None

try:
    connection = get_db_connection()

    cursor = connection.cursor()

    cursor.execute("SELECT DATABASE(), VERSION()")

    result = cursor.fetchone()

    print("Cloud Database Connected Successfully!")
    print("Database:", result[0])
    print("Version:", result[1])

    cursor.execute("SHOW TABLES")

    print("\nAvailable Tables:")

    for table in cursor.fetchall():
        print(table[0])

    cursor.close()

except Exception as error:
    print("Connection failed:", error)

finally:
    if connection:
        connection.close()