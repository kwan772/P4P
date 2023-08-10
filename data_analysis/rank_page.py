import networkx as nx
import community as community_louvain
import random
import networkx as nx
import matplotlib.pyplot as plt
from data_collection.db_connection import db
import matplotlib.cm as cm


cursor = db.cursor()

query = f"""
SELECT parent_author_id, author_id 
FROM wsb_comments
WHERE parent_author_id != "None" 
  AND author_id != "None"
ORDER BY RAND()
limit 10000
"""

cursor.execute(query)
result = cursor.fetchall()
num_edges = len(result)

# Create a directed graph
G = nx.Graph()

for edge in result:
    G.add_edge(edge[1],edge[0])

# Compute PageRank
pr = nx.pagerank(G, alpha=0.85)  # alpha is the damping factor

trustScores = []
for node, score in pr.items():
    trustScores.append((node,score))
trustScores.sort(key=lambda x:-x[1])
print(trustScores)

# # Print PageRank scores
# for node, score in pr.items():
#     print(f"Node {node} has PageRank score: {score}")
#
# # Determine the node size based on PageRank scores
# labels = {node: node for node in G.nodes() if G.degree(node) > 500}
#
# # Rest of your code ...
#
# # 1. Scale Nodes by PageRank Scores
# node_sizes = [pr[node]*50000 for node in G.nodes()]  # adjust the multiplier for appropriate scaling
#
# # 2. Layout Algorithm
# pos = nx.fruchterman_reingold_layout(G)

# largest_cc = max(nx.connected_components(G), key=len)
# H = G.subgraph(largest_cc)
#
# # 3. Subgraph Visualization
# # Get the largest connected component. This will work only if you're using an undirected graph.
#
# plt.figure(figsize=(12, 12))
# nx.draw(H, pos, labels=labels, node_size=node_sizes, alpha=0.5, edge_color='gray')
#
# plt.show()
