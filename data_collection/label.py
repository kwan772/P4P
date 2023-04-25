import os
import re
import threading
from datetime import datetime

# Set up a connection to the MySQL database

class Label():
    def __init__(self, post, symbols, db):
        super().__init__()
        self.post = post
        self.db = db
        self.symbols = symbols
        self.is_labeled = False
        self.id = self.post[0]
        self.count = 0
        self.labeled = set()

    def run(self):
        if "label" in self.id:
            return

        selftext = re.findall(r'\w+', self.post[7])
        title = re.findall(r'\w+', self.post[10])

        self.findMatches(selftext)
        self.findMatches(title)





    def findMatches(self, words):
        for word in words:
            if word in self.symbols:
                if not self.is_labeled:
                    self.labelPost(self.symbols[word])
                    self.is_labeled = True
                else:
                    self.insertPostLabel(self.symbols[word])

    def labelPost(self, symbol):
        cursor = self.db.cursor()
        query = "update reddit_posts set symbol = %s where id = %s"
        values = (symbol.upper(), self.id)
        cursor.execute(query, values)
        self.db.commit()
        cursor.close()
        self.labeled.add(symbol)
        print("labeled " + self.id + " with " + symbol)

    def insertPostLabel(self, symbol):
        if not self.labeled.__contains__(symbol):
            try:
                cursor = self.db.cursor()
                new_post = (self.post[0] + "_label" + str(self.count),) + self.post[1:-1] + (symbol.upper(),)
                self.count += 1
                query = "INSERT INTO reddit_posts (id, clicked, distinguished, edited, is_original_content, is_self, over_18, selftext, spoiler, stickied, title, upvote_ratio, created_utc, num_comments, score, symbol) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(query, new_post)
                cursor.close()
                self.labeled.add(symbol)
                print("inserted " + self.id + " with " + symbol + " count = " + str(self.count))
            except Exception as e:
                print(e)

