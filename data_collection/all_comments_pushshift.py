import requests
import json

def get_comments(post_id):
    url = f"https://api.pushshift.io/reddit/comment/search/?q=quantum&after=24h"
    response = requests.get(url)
    data = json.loads(response.text)

    print(data)

    return data['data']

def parse_comments(comments):
    for comment in comments:
        print(comment['body'])

post_id = '10013dm'
comments = get_comments(post_id)
parse_comments(comments)