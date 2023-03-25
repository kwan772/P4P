import os

import mysql.connector

# Set up a connection to the MySQL database
db = mysql.connector.connect(
  host="localhost",
  user="root",
  password= os.environ.get("DB_PASSWORD"),
  database="p4p"
)

# # Use the cursor() method to create a cursor object
# cursor = db.cursor()
#
# # Execute a SQL query
# cursor.execute("SELECT * FROM yourtable")
#
# # Fetch all the rows returned by the query
# rows = cursor.fetchall()
#
# # Print each row
# for row in rows:
#     print(row)
#
# # Close the cursor and database connection
# cursor.close()
db.close()