import matplotlib.pyplot as plt
import numpy as np

from data_collection.db_connection import db


cursor = db.cursor()
query = f"""select avg(sentiment_body_score) a, avg(sentiment_title_score) b, sum(num_comments) c, author_id from reddit_posts where author_id is not null and author_id != 'None' group by author_id order by c desc limit 10;"""
cursor.execute(query)
result = cursor.fetchall()
columns = []
v1 = []
v2 = []

for i in range(10):
    columns.append(result[i][3])
    v1.append(result[i][0])
    v2.append(result[i][1])

print(columns)
print(v2)

# converting string values to float
v1 = [float(format(x,'.2f')) for x in v1]
v2 = [float(format(x,'.2f')) for x in v2]

print(v1)
print(v2)


x = np.arange(len(columns))

# Set the width of the bars
width = 0.35

# Plotting the bars
fig, ax = plt.subplots()
bar1 = ax.bar(x - width/2, v1, width, label='Body')
bar2 = ax.bar(x + width/2, v2, width, label='Title')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Authors')
ax.set_ylabel('Average Post Sentiment')
ax.set_title('Top 10 Authors Average Sentiment Score')
ax.set_xticks(x)
ax.set_xticklabels(columns)
ax.legend()

# Save the plot as a PNG image file
plt.savefig('sentiment_author.png')

# Display the plot
plt.show()
