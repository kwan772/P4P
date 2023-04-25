import threading
from datetime import datetime


class FetchThread(threading.Thread):
    def __init__(self, post_id, reddit, post_data):
        super().__init__()
        self.post_id = post_id
        self.reddit = reddit
        self.post_data = post_data

    def run(self):
        post = self.reddit.submission(id=self.post_id)
        print(post.title)
        edited = True if post.edited else False
        self.post_data.append(tuple({
        'id': post.id,
        'clicked': post.clicked, #bool
        'distinguished': post.distinguished, #bool
        'edited': edited, #bool
        'is_original_content': post.is_original_content, #bool
        'is_self': post.is_self, #bool
        'over_18': post.over_18, #bool
        'selftext': post.selftext, #big text
        'spoiler': post.spoiler, #bool
        'stickied': post.stickied, #bool
        'title': post.title, #medium text
        'upvote_ratio': post.upvote_ratio, #double
        'created_utc': datetime.utcfromtimestamp(post.created_utc),
        'num_comments': post.num_comments,
        'score': post.score
    }.values()))