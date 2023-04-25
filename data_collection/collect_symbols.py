import pandas as pd
from db_connection import db


data = pd.read_csv('data/stocks-list.csv')
# create a cursor object
cursor = db.cursor()
# iterate over each row of the DataFrame and insert it into the database
for index, row in data.iterrows():
    sql = "INSERT INTO symbols (symbol, company_name, industry, market_cap) VALUES (%s, %s,%s, %s)"
    mcap = 0
    if not pd.isna(row['Market Cap']):
        mcap = round(row['Market Cap'])

    print(mcap)
    print(row['Symbol'])
    values = (row['Symbol'], row['Company Name'], row['Industry'], mcap)
    try:
        cursor.execute(sql, values)
    except Exception:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(row['Market Cap'])
        print(mcap)

# commit the changes to the database
db.commit()

# close the cursor and connection objects
cursor.close()
db.close()