import matplotlib.pyplot as plt
from data_collection.db_connection import db

cursor = db.cursor()
query = "select sum(num_comments)c from reddit_posts"
cursor.execute(query)
total_comments = cursor.fetchall()
total_comments = total_comments[0][0]

cursor = db.cursor()
query = f"""select sum(num_comments) c, author_id from reddit_posts where author_id is not null and author_id != 'None' group by author_id order by c desc;"""
cursor.execute(query)
result = cursor.fetchall()
columns = []
values = []

for i in range(9):
    columns.append(result[i][1])
    values.append(format(result[i][0]/total_comments *100, '.2f'))

columns.append('other')
sum = 0

for i in range(9, len(result)):
    sum += result[i][0]
values.append(format(sum/total_comments * 100, '.2f'))

print(columns)
print(values)

# converting string values to float
values = [float(x) for x in values]

# create bar graph
plt.bar(columns, values)

# optional enhancements
plt.title('Author Comment Percentage')
plt.xlabel('Author')
plt.ylabel('Percentage(%)')

# display the plot
plt.show()
plt.savefig('plot.png', dpi=300)