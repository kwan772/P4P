
import praw
from datetime import datetime, timezone, timedelta
import requests
import json
import pytz
from db_connection import db

subreddit_name = 'wallstreetbets'
# subreddit_name = 'wallstreetbets'
start_date = datetime(2012, 1, 1)
#Jan 31, 2012
end_date = datetime.now() - timedelta(days=30)
end_date = datetime(2020, 1, 1)
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
# res = requests.get(url)
# data = res.json()['data']
# posts.extend(data)
# print(datetime.utcfromtimestamp(data[-1]['created_utc']))
while True:
    res = requests.get(url)
    # print(res.json())
    data = res.json()['data']
    if not data:
        print(url)
        break
    last_post = data[-1]
    last_post_timestamp = last_post['created_utc']
    if last_post_timestamp <= start_time:
        posts.extend([post for post in data if post['created_utc'] > start_time])
        print(datetime.fromtimestamp(last_post_timestamp))
        print(datetime.fromtimestamp(start_time))
        print(last_post_timestamp <= start_time)
        print("done")
        break
    posts.extend(data)
    url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit_name}&size=1000&before={last_post_timestamp}'
    print("@@@@@@@@@@@@")
    print(datetime.fromtimestamp(last_post_timestamp))

for post in posts:
    print("Title:", post['title'])
    print("utc:", datetime.fromtimestamp(post['created_utc']))