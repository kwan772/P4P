import os
import threading
from datetime import datetime

import mysql


class FetchAuthorIdThread(threading.Thread):
    def __init__(self, post_id, reddit, update_data):
        super().__init__()
        self.post_id = post_id
        self.reddit = reddit
        self.update_data = update_data

    def run(self):
        post = self.reddit.submission(id=self.post_id)
        author = post.author
        self.update_data.append([str(author), self.post_id])