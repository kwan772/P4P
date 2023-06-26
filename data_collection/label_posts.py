import pandas as pd

from data_collection.db_connection import db
from data_collection.label import Label
from data_collection.label_thread import LabelThread

cursor = db.cursor()
query = "SELECT lower(symbol), lower(company_name) FROM symbols"
# for insertion in insertions:
#     if insertion[0] == 'ei1foq':
#         print("@@@@@@@@@@@@@@@")
cursor.execute(query)

data = cursor.fetchall()
cursor.close()

symbols = {}
post_ids = set()

for row in data:
    sb = "$" + row[0]
    symbols[sb] = row[0]
    # symbols[row[1]] = row[0]

print(symbols)
# cursor = db.cursor()
# query = f"SELECT * FROM reddit_posts Limit {0},{1000}"
# # for insertion in insertions:
# #     if insertion[0] == 'ei1foq':
# #         print("@@@@@@@@@@@@@@@")
# cursor.execute(query)
#
# d= cursor.fetchall()
# cursor.close()
# db.close()

# print(d)

number_of_rows = 500
number_of_processed_rows = 0
stop_loop = False



while not stop_loop:
    cursor = db.cursor()
    query = f"SELECT * FROM reddit_posts Limit {number_of_processed_rows},{number_of_rows}"
    cursor.execute(query)

    d = cursor.fetchall()
    cursor.close()
    num_of_rows = len(d)
    number_of_processed_rows += num_of_rows
    if num_of_rows != number_of_rows:
        stop_loop = True

    threads = []

    for post in d:
        # Get post data for each chunk of post IDs
        # post = reddit.submission(id=post_id)
        #
        # print(post)
        # Extract the relevant data you need from each post
        # post_dict = {
        #     'id': post.id,
        #     'title': post.title,
        #     'created_utc': post.created_utc,
        #     'num_comments': post.num_comments,
        #     'score': post.score
        # }
        # post_data.append(post_dict)
        if not post_ids.__contains__(post[0]):

            thread = Label(post, symbols, db)
            thread.run()
            post_ids.add(post[0])

    #     thread = LabelThread(post, symbols)
    #     threads.append(thread)
    #     thread.start()
    #     # print(post.title)
    #
    for thread in threads:
        thread.join()

    print(f"Processed: {number_of_processed_rows} rows")
db.close()