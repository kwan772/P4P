from db_connection import db


data = [
    # ['BTC', 'bitcoin', 'crypto', 582435242349, 'crypto'],
    # ['ETH', 'ethereum', 'crypto', 231140396606, 'crypto'],
    # ['ADA', 'cardano', 'crypto', 14153226979, 'crypto'],
    # ['DOGE', 'dogecoin', 'crypto', 11573890593, 'crypto'],
    # ['MATIC', 'polygon', 'crypto', 10267146099, 'crypto'],
    ['SOL', 'solana', 'crypto', 9387522042, 'crypto']
    # ['DOT', 'polkadot', 'crypto', 7575272389, 'crypto'],
    # ['SHIB', 'shiba inu', 'crypto', 6442868905, 'crypto'],
    # ['AVAX', 'avalanche', 'crypto', 6034167462, 'crypto']
        ]
# create a cursor object
cursor = db.cursor()
# iterate over each row of the DataFrame and insert it into the database
for row in data:
    sql = "INSERT INTO symbols (symbol, company_name, industry, market_cap, symbol_type) VALUES (%s, %s,%s, %s, %s)"
    values = (row[0], row[1], row[2], row[3], row[4])
    cursor.execute(sql, values)

# commit the changes to the database
db.commit()

# close the cursor and connection objects
cursor.close()
db.close()