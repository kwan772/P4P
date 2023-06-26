import networkx as nx
import matplotlib.pyplot as plt
from data_collection.db_connection import db

G = nx.Graph()

cursor = db.cursor()
query = f"""select parent_author_id, author_id from wsb_comments where parent_author_id in (select t1.author_id from (select author_id, sum(num_comments) snc from reddit_posts group by author_id order by snc desc limit 10) t1)"""
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

labels = {node: node for node in G.nodes() if G.degree(node) > 100}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight=700, font_color='black')

plt.savefig('graph.png')  # Save the graph to a file
plt.show()