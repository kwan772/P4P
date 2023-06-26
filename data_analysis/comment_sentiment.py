import matplotlib.pyplot as plt
from data_collection.db_connection import db


cursor = db.cursor()
query = f"""select avg(sentiment) avg, parent_author_id, count(id) c from wsb_comments where parent_author_id is not null and author_id != 'None' group by parent_author_id order by c desc limit 10;"""
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
plt.title('Top 10 Authors')
plt.xlabel('Authors')
plt.ylabel('Average Comment Sentiment')

# display the plot
plt.show()
plt.savefig('comment_sentiment.png', dpi=300)