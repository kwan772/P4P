import networkx as nx
import matplotlib.pyplot as plt
from data_collection.db_connection import db

G = nx.Graph()

cursor = db.cursor()
# query = f"""select parent_author_id, author_id from wsb_comments where parent_author_id in (select t1.author_id from (select author_id, sum(num_comments) snc from reddit_posts group by author_id order by snc desc limit 10) t1)"""
query = f"""CREATE TEMPORARY TABLE IF NOT EXISTS top_authors AS (
SELECT t1.author_id AS author
FROM (
  SELECT author_id, SUM(num_comments) AS snc
  FROM reddit_posts
  GROUP BY author_id
  ORDER BY snc DESC
  LIMIT 100
) t1
LEFT JOIN (
  SELECT DISTINCT(author_id)
  FROM reddit_posts
  WHERE distinguished = "moderator"
) t2 ON t1.author_id = t2.author_id
WHERE t2.author_id IS NULL and t1.author_id is not null limit 10
);"""
cursor.execute(query)
db.commit()

query = """SELECT parent_author_id, author_id 
FROM wsb_comments 
WHERE parent_author_id IN (SELECT author FROM top_authors);"""
cursor = db.cursor()
cursor.execute(query)
result = cursor.fetchall()

print(result)
print(len(result))

for edge in result:
    G.add_edge(edge[0], edge[1])

# Node sizes based on degree
sizes = [G.degree(node) * 1.01 for node in G.nodes()]

pos = nx.spring_layout(G)  # Layout algorithm for positioning nodes

nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='lightblue')
# nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='#045993')

labels = {node: node for node in G.nodes() if G.degree(node) > 100}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight=700, font_color='black')
# nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight=700, font_color='#ff7c0c')

plt.savefig('graph.png')  # Save the graph to a file
plt.show()