import praw


reddit = praw.Reddit(client_id='QkUsWUsdKxz5ZW0Cw_JmlQ', client_secret='dgPACtoDgLVOlnz7FkAxxFxAvHJVIg',
                         user_agent='kwan772@aucklanduni.ac.nz')
submission = reddit.submission("103w8ct")
submission.comments.replace_more(limit=None)
print(submission.num_comments)
count = 0
for comment in submission.comments:
    count+=1

print(count)