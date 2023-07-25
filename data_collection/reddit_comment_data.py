import ast

# Reading the data from the text file
with open('reddit_comment.txt', 'r') as f:
    data_str = f.read()

# Converting the data to a Python dictionary
data = ast.literal_eval(data_str)

# Now data is a Python dictionary that you can work with.
# For example, you could print it:
for row in data[1]['data']['children']:
    print(row)

print(len(data[1]['data']['children'][-1]["data"]["children"]))