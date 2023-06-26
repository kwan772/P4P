import matplotlib.pyplot as plt
from data_collection.db_connection import db


cursor = db.cursor()
query = f"""SELECT count(*)c, symbol from reddit_posts where symbol is not null group by symbol order by c desc limit 10;"""
cursor.execute(query)
result = cursor.fetchall()
columns = []
values = []

for i in range(10):
    columns.append(result[i][1])
    values.append(result[i][0])

print(columns)
print(values)

# converting string values to float
values = [float(x) for x in values]

# create bar graph
plt.bar(columns, values)

# optional enhancements
plt.title('Top 10 Symbols')
plt.xlabel('Symbols')
plt.ylabel('Number of Posts')

# display the plot
plt.show()
plt.savefig('symbol_posts.png', dpi=300)