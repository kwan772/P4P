
import praw
from datetime import datetime, timezone, timedelta
import requests
import pytz
from db_connection import db
import json

subreddit_name = 'wallstreetbets'
# subreddit_name = 'wallstreetbets'
start_date = datetime(2021, 11, 24)
#Jan 31, 2012
end_date = datetime.now() - timedelta(days=30)
end_date = datetime(2022, 11, 3)
utc = pytz.UTC
start_date_utc = utc.localize(start_date)
end_date_utc = utc.localize(end_date)

start_time = int(start_date_utc.timestamp())
end_time = int(end_date_utc.timestamp())


start_time = int(start_date.timestamp())
end_time = int(end_date.timestamp())
print(start_time)
print(end_time)

url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit_name}&size=1000&before={end_time}'

posts = []
res = requests.get(url)
data = res.json()['data']
posts.extend(data)
print(datetime.utcfromtimestamp(data[-1]['created_utc']))

def bulk_insert_data(data):
    insertions = []
    keys = ['id', 'created_utc', 'selftext', 'url', 'title']
    ids = set()
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
        insertion['id'] = post['id']
        insertion['created_utc'] = datetime.utcfromtimestamp(post["created_utc"])
        insertion['selftext'] = post['selftext']
        insertion['url'] = post['url']
        insertion['title'] = post['title']
        if not ids.__contains__(insertion['id']):
            ids.add(insertion['id'])
            insertions.append(tuple(insertion.values()))

    if insertions:
        cursor = db.cursor()
        query = "INSERT INTO push_shift_posts ({}) VALUES ({})".format(
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
# for post in posts:
#     for key in post:
#         print(f"{key}=>>>>>>>>>>>>>>>>>>>>>>>>>>>{post[key]}")
while True:
    try:
        print("1111111111111")
        posts = []
        res = requests.get(url)
        print("@2222222")
        # print(res.json())
        data = res.json()
        if not data:
            continue
        else:
            data = data['data']

        last_post = data[-1]
        last_post_timestamp = last_post['created_utc']
        if last_post_timestamp <= start_time:
            posts.extend([post for post in data if post['created_utc'] > start_time])
            # print(datetime.fromtimestamp(last_post_timestamp))
            # print(datetime.fromtimestamp(start_time))
            # print(last_post_timestamp <= start_time)
            # print("done")
            bulk_insert_data(posts)
            break

        posts.extend(data)
        bulk_insert_data(posts)
        url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit_name}&size=1000&before={last_post_timestamp}'
    except Exception:
        print(Exception)
        continue





