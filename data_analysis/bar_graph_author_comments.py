import matplotlib.pyplot as plt
from data_collection.db_connection import db


cursor = db.cursor()
query = f"""select sum(num_comments) c, author_id from reddit_posts group by author_id order by c desc;"""
cursor.execute(query)
result = cursor.fetchall()
columns = []
values = []

for i in range(9):
    columns.append(result[i][1])
    values.append(result[i][0])

columns.append('other')
sum = 0

for i in range(9, len(result)):
    sum += result[i][0]
values.append(sum)

print(columns)
print(values)

# converting string values to float
values = [int(x) for x in values]

# create bar graph
plt.bar(columns, values)

# optional enhancements
plt.title('Author Comments')
plt.xlabel('Authors')
plt.ylabel('Number of Comments')

# display the plot
plt.show()
plt.savefig('author_comment.png', dpi=300)