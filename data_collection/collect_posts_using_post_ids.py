import praw
import pandas as pd

from data_collection.fetch_thread import FetchThread
from db_connection import db

def bulk_insert_data(data):
    keys = ['id','clicked','distinguished','edited','is_original_content','is_self','over_18','selftext','spoiler','stickied','title','upvote_ratio','created_utc','num_comments','score']

    if data:
        cursor = db.cursor()
        query = "INSERT INTO reddit_posts ({}) VALUES ({})".format(
            ', '.join(keys),
            ', '.join(['%s'] * len(data[0]))
        )
        # for insertion in insertions:
        #     if insertion[0] == 'ei1foq':
        #         print("@@@@@@@@@@@@@@@")
        cursor.executemany(query, data)
        db.commit()
        print(cursor.rowcount, "record inserted.")

number_of_rows = 500
number_of_processed_rows = 0
stop_loop = False
while not stop_loop:
    reddit = praw.Reddit(client_id='QkUsWUsdKxz5ZW0Cw_JmlQ', client_secret='dgPACtoDgLVOlnz7FkAxxFxAvHJVIg',
                         user_agent='kwan772@aucklanduni.ac.nz')
    # Assuming your post IDs are stored in a MySQL table named 'post_ids' with column 'post_id'
    query = f"SELECT id FROM push_shift_posts where created_utc < '2022-11-03 19:31:57' and created_utc > '2021-11-24 23:30:20' Limit {number_of_processed_rows},{number_of_rows}"
    df = pd.read_sql_query(query, db)  # conn is your MySQL connection object

    post_ids = list(df['id'])
    num_of_rows = len(post_ids)
    number_of_processed_rows += num_of_rows
    if num_of_rows != number_of_rows:
        stop_loop = True

    # Create an empty list to store post data
    post_data = []
    threads = []

    print(len(post_ids))

    for post_id in post_ids:
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

        thread = FetchThread(post_id, reddit, post_data)
        threads.append(thread)
        thread.start()
        # print(post.title)

    for thread in threads:
        thread.join()

    bulk_insert_data(post_data)
    print(f"Processed: {number_of_processed_rows} rows")
