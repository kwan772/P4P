import concurrent
import queue
from datetime import datetime
from time import sleep

import praw
from db_connection import db

reddit = praw.Reddit(client_id='QkUsWUsdKxz5ZW0Cw_JmlQ', client_secret='dgPACtoDgLVOlnz7FkAxxFxAvHJVIg',
                         user_agent='kwan772@aucklanduni.ac.nz')
# Replace 'POST_ID' with the actual ID of the Reddit post you want to get the comments for
# post = reddit.submission(id='10fikna')

# Get all comments for the post using the 'post.comments.list()' method
# comments = post.comments.list()
# print(post.author)
# data = []
cursor = db.cursor()
query = "select author_id, sum(num_comments) as snc from reddit_posts where author_id is not null and not author_id = 'None' group by author_id order by snc desc limit 100"
# for insertion in insertions:
#     if insertion[0] == 'ei1foq':
#         print("@@@@@@@@@@@@@@@")
cursor.execute(query)
data = cursor.fetchall()
db.commit()

data = [d[0] for d in data]
print(data)

cursor = db.cursor()
query = f"select id, author_id, symbol from reddit_posts where author_id in ({','.join(['%s'] * len(data))})"
# for insertion in insertions:
#     if insertion[0] == 'ei1foq':
#         print("@@@@@@@@@@@@@@@")
print(query)
cursor.execute(query,data)
result = cursor.fetchall()
db.commit()
print(len(result))

def bulk_insert_data(data):
    keys = ['edited', 'body', 'created_utc', 'author_id', 'id', 'link_id', 'parent_id', 'parent_symbol', 'parent_author_id']

    if data:
        cursor = db.cursor()
        query = "INSERT INTO wsb_comments ({}) VALUES ({})".format(
            ', '.join(keys),
            ', '.join(['%s'] * len(data[0]))
        )
        # for insertion in insertions:
        #     if insertion[0] == 'ei1foq':
        #         print("@@@@@@@@@@@@@@@")
        cursor.executemany(query, data)
        db.commit()
        print(cursor.rowcount, "record inserted.")

data = []
row_count = 1
i=0

def get_comments(res):
    post = reddit.submission(id=res[0])
    # print("here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("here@@@@@@@@@@@@@@@@@@@@@@@")
    for comment in post.comments:
        data.append(
            [1 if comment.edited else 0, comment.body, comment.created_utc, comment.author.name, comment.id, comment.link_id,
             res[0], res[2], res[1]])

while i <= len(result):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for x in range(50):
            if i < len(result):
                futures.append(executor.submit(get_comments, result[i]))
                i += 1

        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

        bulk_insert_data(data)
        print("rows processed: " + str(i))
        data = []
