import json
import pprint
import pandas as pd
import requests

from data_collection.db_connection import db

headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; PPC Mac OS X 10_8_7 rv:5.0; en-US) AppleWebKit/533.31.5 (KHTML, like Gecko) Version/4.0 Safari/533.31.5',
    "Authorization": f"Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IlNIQTI1NjpzS3dsMnlsV0VtMjVmcXhwTU40cWY4MXE2OWFFdWFyMnpLMUdhVGxjdWNZIiwidHlwIjoiSldUIn0.eyJzdWIiOiJ1c2VyIiwiZXhwIjoxNjkwMjYxNjA4LjI4OTEwNywiaWF0IjoxNjkwMTc1MjA4LjI4OTEwNywianRpIjoiUmhZeWQ1YlUtejlnLWMyR2RZR0ladWZ0ZHgyMVd3IiwiY2lkIjoiUWtVc1dVc2RLeHo1WlcwQ3dfSm1sUSIsImxpZCI6InQyX2dyNzFkZnZrIiwiYWlkIjoidDJfZ3I3MWRmdmsiLCJsY2EiOjE2Mzc0NDgzNTc1MDcsInNjcCI6ImVKeUtWdEpTaWdVRUFBRF9fd056QVNjIiwiZmxvIjo5fQ.ipcRU5hsrxTYw2xIwiZ1AOyU0b_ET2irm3cDej6TypgTEv8GudE8Hi9iIiLvx6njCzs8RA3lMGhtkO72I4rUSNoI_QUzlGfVdrLunt0MO8a0QhtnX_NYQNYvIrXPosC9RHmnfcezz0KrMXUjsrdc0Xxm9MMYuL_zKQVmVa9jVkpZyC88mr0WNU2RQ7norPo_8sBet-VID9iJb9iXeFOv1HTiEiBap2HejoH9iUt0gbJ3l5mtQVOfQ80iYeAw0739s8Sdk-QZGYE0_Z1Zpy5AcAF0fv4TmJx26FI9PJYlE4GR_XlQm4AOLzW97rkNbFUSbdbJS5iZFiZJgPp_1Rv7YQ"
}


def get_all_comments(post_id):
    response = requests.get(f"http://oauth.reddit.com/comments/{post_id}?&threded=false", headers=headers)

    # print(response)
    response_json = json.loads(response.text)

    # print(response_json)
    pprint.pprint(response_json)

    return

def get_one_comment(post_id, comment_ids, subreddit):
    for comment_id in comment_ids:
        response = requests.get(f"https://oauth.reddit.com/r/{subreddit}/comments/10pw79e/_/j6os47a.json", headers=headers)
        response_json = json.loads(response.text)
# https://www.reddit.com/r/{subreddit}/comments/{article}/_/{comment_id}.json



if __name__ == "__main__":
    post_id = "lkbfl4"
    subreddit = "wallstreetbets"
    # # comment_ids = get_all_comments(post_id)
    # # get_one_comment(post_id, comment_ids, subreddit)
    # response = requests.get(f"https://oauth.reddit.com/r/{subreddit}/comments/{post_id}/_/j6os47a.json", headers=headers)
    # response_json = json.loads(response.text)
    # pprint.pprint(response_json)
    get_all_comments(post_id)


    # query = f"SELECT id FROM push_shift_posts where created_utc < '2022-11-03 19:31:57' and created_utc > '2021-11-24 23:30:20' Limit 1000"
    # df = pd.read_sql_query(query, db)  # conn is your MySQL connection object
    #
    # post_ids = list(df['id'])
    #
    # count = 0
    # for postId in post_ids:
    #     count +=1
    #     get_all_comments(postId)
    #     print(count)

