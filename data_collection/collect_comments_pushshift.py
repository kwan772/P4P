from datetime import datetime
from db_connection import db

import requests

post_id = 'l8rf4k'
end_date = datetime.today()
end_time = int(end_date.timestamp())
start_date = datetime(2012, 4, 12)
start_time = int(start_date.timestamp())
symbols = []
subreddit = "wallstreetbets"

# cursor = db.cursor()
# query = "select symbol"
# # for insertion in insertions:
# #     if insertion[0] == 'ei1foq':
# #         print("@@@@@@@@@@@@@@@")
# cursor.executemany(query, insertions)
# db.commit()



def bulk_insert_data(data, symbol):
    insertions = []
    keys = ['subreddit_id', 'author_is_blocked', 'comment_type', 'edited', 'subreddit','body','created_utc','id','symbol']
    for post in data:
        # print("Title:", post['title'])
        # print("utc:", datetime.fromtimestamp(post['created_utc']))

        # post["author_created_utc"] = datetime.utcfromtimestamp(post["author_created_utc"])
        # post["created_utc"] = datetime.utcfromtimestamp(post["created_utc"])
        # post["retrieved_utc"] = datetime.utcfromtimestamp(post["retrieved_utc"])
        # post["updated_utc"] = datetime.utcfromtimestamp(post["updated_utc"])
        # post["all_awardings"] = json.dumps(post["all_awardings"])
        # post["author_flair_richtext"] = json.dumps(post["author_flair_richtext"])
        # post["awarders"] = json.dumps(post["awarders"])
        # post["gildings"] = json.dumps(post["gildings"])
        # post["link_flair_richtext"] = json.dumps(post["link_flair_richtext"])
        # post["media_embed"] = json.dumps(post["media_embed"])
        # post["secure_media_embed"] = json.dumps(post["secure_media_embed"])
        # post["treatment_tags"] = json.dumps(post["treatment_tags"])

        insertion = {}
        insertion['subreddit_id'] = post['subreddit_id']
        insertion['author_is_blocked'] = post['author_is_blocked']
        insertion['comment_type'] = post['comment_type']
        insertion['edited'] = post['edited']
        insertion['subreddit'] = post['subreddit']
        insertion['body'] = post['body']
        insertion['created_utc'] = post['utc_datetime_str']
        insertion['id'] = post['id']
        insertion['symbol'] = symbol
        insertions.append(tuple(insertion.values()))

    if insertions:
        cursor = db.cursor()
        query = "INSERT INTO comments ({}) VALUES ({})".format(
            ', '.join(keys),
            ', '.join(['%s'] * len(insertions[0]))
        )
        # for insertion in insertions:
        #     if insertion[0] == 'ei1foq':
        #         print("@@@@@@@@@@@@@@@")
        cursor.executemany(query, insertions)
        db.commit()
        print(cursor.rowcount, "record inserted.")
        print(datetime.fromtimestamp(data[0]['created_utc']))


def fetch_comments_for_symbol(symbol):
    while True:
        try:
            posts = []
            # set query parameters
            query_params = {
                'q': '$'+symbol,
                'size': 500,
                'before': end_time,
                # 'subreddit': subreddit,
            }
            res = requests.get('https://api.pushshift.io/reddit/comment/search', params=query_params)
            print("@2222222")
            # print(res.json())
            data = res.json()
            if not data:
                break
            else:
                data = data['data']

            last_post = data[-1]
            last_post_timestamp = last_post['created_utc']
            bulk_insert_data(data, symbol)
            for d in data:
                print(d['utc_datetime_str'])
                print(d['body'])
            if last_post_timestamp <= start_time:
                # posts.extend([post for post in data if post['created_utc'] > start_time])
                # print(datetime.fromtimestamp(last_post_timestamp))
                # print(datetime.fromtimestamp(start_time))
                # print(last_post_timestamp <= start_time)
                # print("done")
                # bulk_insert_data(posts)
                break

            # bulk_insert_data(posts)
            query_params['before'] = last_post_timestamp

        except Exception as e:
            print(e)
            continue

fetch_comments_for_symbol("AAPL")
