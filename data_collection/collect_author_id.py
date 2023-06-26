import praw
import pandas as pd

from data_collection.db_connection import db
from data_collection.fetch_author_id_thread import FetchAuthorIdThread

number_of_rows = 100
number_of_processed_rows = 0
stop_loop = False
while not stop_loop:
    reddit = praw.Reddit(client_id='QkUsWUsdKxz5ZW0Cw_JmlQ', client_secret='dgPACtoDgLVOlnz7FkAxxFxAvHJVIg',
                         user_agent='kwan772@aucklanduni.ac.nz')
    # Assuming your post IDs are stored in a MySQL table named 'post_ids' with column 'post_id'
    query = f"SELECT id FROM reddit_posts order by id asc Limit {number_of_processed_rows},{number_of_rows}"
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

    update_data = []

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

        thread = FetchAuthorIdThread(post_id, reddit, update_data)
        threads.append(thread)
        thread.start()
        # print(post.title)

    for thread in threads:
        thread.join()

    cursor = db.cursor()
    for update in update_data:
        query = f"update reddit_posts set author_id = '{update[0]}' where id = '{update[1]}'"
        res = cursor.execute(query)
        # print("labeled " + update[1] + " with " + update[0])

    db.commit()
    cursor.close()

    print(f"Processed: {number_of_processed_rows} rows")

db.close()
