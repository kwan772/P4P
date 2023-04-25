from datetime import datetime
from time import sleep

import praw

reddit = praw.Reddit(client_id='QkUsWUsdKxz5ZW0Cw_JmlQ', client_secret='dgPACtoDgLVOlnz7FkAxxFxAvHJVIg',
                         user_agent='kwan772@aucklanduni.ac.nz')
# Replace 'POST_ID' with the actual ID of the Reddit post you want to get the comments for
post = reddit.submission(id='12ip7u8')

# Get all comments for the post using the 'post.comments.list()' method
# comments = post.comments.list()
# print(post.author)
# data = []
comments = []

# Iterate through the comments and print the comment body and author name
print(post.num_comments)
count = 0

# while True:
    # try:
    #     moreComments = post.comments.replace_more()
    #     for moreComment in moreComments:
    #         c = moreComment.comments()
    #         print(c)
    #         for com in c:
    #             try:
    #                 print(datetime.fromtimestamp(com.created_utc))
    #             except Exception:
    #                 print("Not a comment")
    #         comments.extend(c)
    #     break
    # except Exception:
    #     print("Handling replace_more exception")
    #     sleep(1)

n_com = post.comments.list()
post.comments.replace_more(limit=None)
comments = post.comments.list()

# for comment in comments:
#     print(comment.body)
print("2@@@@@@@@@@@@@")
print(len(comments))
print(len(n_com))