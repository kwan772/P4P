import matplotlib.pyplot as plt
from data_collection.db_connection import db


cursor = db.cursor()
query = f"""select count(*) c, author_id from reddit_posts where author_id is not null and author_id != 'None' group by author_id order by c desc;"""
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
plt.ylabel('Number of Posts')

# display the plot
plt.show()
plt.savefig('author_posts.png', dpi=300)