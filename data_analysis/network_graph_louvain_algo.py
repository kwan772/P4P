import community as community_louvain
import random
import networkx as nx
import matplotlib.pyplot as plt
from data_collection.db_connection import db
import matplotlib.cm as cm

G = nx.Graph()

cursor = db.cursor()
# query = f"""
# SELECT parent_author_id, author_id
# FROM comments_for_certain_symbols
# WHERE symbol = "tsla"
#   AND parent_author_id != "None"
#   AND author_id != "None"
# ORDER BY RAND()
# limit 10000;
# """

query = f"""
SELECT parent_author_id, author_id 
FROM wsb_comments
WHERE parent_author_id != "None" 
  AND author_id != "None"
ORDER BY RAND()
limit 5000;
"""
# query = f"""select parent_author_id, author_id from wsb_comments where parent_author_id in (select t1.author_id from (select author_id, sum(num_comments) snc from reddit_posts where author_id is not null and author_id != 'None' group by author_id order by snc desc limit 10) t1)"""
# query = f"""CREATE TEMPORARY TABLE IF NOT EXISTS top_authors AS (
# SELECT t1.author_id AS author
# FROM (
#   SELECT author_id, SUM(num_comments) AS snc
#   FROM reddit_posts
#   GROUP BY author_id
#   ORDER BY snc DESC
#   LIMIT 100
# ) t1
# LEFT JOIN (
#   SELECT DISTINCT(author_id)
#   FROM reddit_posts
#   WHERE distinguished = "moderator"
# ) t2 ON t1.author_id = t2.author_id
# WHERE t2.author_id IS NULL and t1.author_id is not null limit 10
# );"""
# cursor.execute(query)
# db.commit()

# query = """SELECT parent_author_id, author_id
# FROM wsb_comments
# WHERE parent_author_id IN (SELECT author FROM top_authors);"""


cursor = db.cursor()
cursor.execute(query)
result = cursor.fetchall()

print(result)
print(len(result))
for edge in result:
    G.add_edge(edge[0], edge[1])

# compute the best partition
partition = community_louvain.best_partition(G)

# draw the graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()