
import praw
from datetime import datetime, timezone, timedelta
import requests
import json
import pytz

subreddit_name = 'python'
# subreddit_name = 'wallstreetbets'
start_date = datetime(2018, 1, 31)
#Jan 31, 2012
end_date = datetime.now() - timedelta(days=30)
end_date = datetime(2020, 11, 30)
utc = pytz.UTC
start_date_utc = utc.localize(start_date)
end_date_utc = utc.localize(end_date)

start_time = int(start_date_utc.timestamp())
end_time = int(end_date_utc.timestamp())


start_time = int(start_date.timestamp())
end_time = int(end_date.timestamp())
print(start_time)
print(end_time)

# start_time = 1483228800 # Unix timestamp for January 1, 2017
# end_time = 1514764800 # Unix timestamp for January 1, 2018

# url = f'https://api.pushshift.io/reddit/search/submission/?ids=11nw54y'
# url1 = f'https://api.pushshift.io/reddit/search/submission/?ids=11nhpu4'
# url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit_name}&after={start_time}&size=1000'
# url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit_name}&before={end_time}&size=1000'
# url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit_name}&size=1000&sort=asc'

submissions = []
# while True:
#     response = requests.get(url)
#     data = json.loads(response.text)
#     if not data['data']:
#         break
#     submissions.extend(data['data'])
#     url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit_name}&after={submissions[-1]['created_utc']}&before={end_time}&size=1000"
url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit_name}&size={1000}&before={end_time}"

response = requests.get(url)
data = json.loads(response.text)
# r1 = requests.get(url1)
# d = json.loads(r1.text)
# print(data['data'][0])
# print(d['data'][0])

for post in data['data']:
    print(post['title'])
    print(datetime.utcfromtimestamp(post["created_utc"]))
    print(post['id'])
    print(post['url'])

reddit = praw.Reddit(
    client_id="QkUsWUsdKxz5ZW0Cw_JmlQ",
    client_secret="dgPACtoDgLVOlnz7FkAxxFxAvHJVIg",
    user_agent="myRedditDataCollector/1.0 NeighborhoodHungry60",
)
all_posts = []
# Retrieve the most recent posts
# subreddit = reddit.subreddit(subreddit_name)
# next_posts = subreddit.new(limit=None, params={'before': end_time})
# recent_posts = subreddit.new(limit=None)
# all_posts.extend(recent_posts)

n = 1
# Keep retrieving posts using the before parameter until there are no more posts
# while n>0:
#     time = end_time
#     if len(all_posts)>0:
#         last_post = all_posts[-1]
#         time = last_post.created_utc
#
#     next_posts = subreddit.new(limit=None, params={'before': time})
#     all_posts.extend(next_posts)
#     n-=1
#
# # Iterate through all posts and print the title and author
# for post in next_posts:
#     print('Post Title:', post.title)
#     print('Post Author:', post.author)
#     print(datetime.utcfromtimestamp(post.created_utc))

# Print the post title and author
# print('Post Title:', post.title)
# print('Post Author:', post.author)
# print('Post upvotes:', post.score)
# print('Post upvotes ratio:', post.upvote_ratio)