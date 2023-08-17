import networkx as nx
import community as community_louvain
import random
import networkx as nx
import matplotlib.pyplot as plt
from data_collection.db_connection import db
import matplotlib.cm as cm


cursor = db.cursor()
symbol = 'gme'

query = f"""
SELECT parent_author_id, author_id 
FROM comments_for_certain_symbols
WHERE parent_author_id != "None" 
  AND author_id != "None"
  AND symbol = "{symbol}"
"""

cursor.execute(query)
result = cursor.fetchall()
num_edges = len(result)

# Create a directed graph
G = nx.DiGraph()

for edge in result:
    G.add_edge(edge[0],edge[1])

# Compute PageRank
pr = nx.pagerank(G, alpha=0.85)  # alpha is the damping factor

trustScores = []
for node, score in pr.items():
    trustScores.append((node,score))
trustScores.sort(key=lambda x:-x[1])

# Extract the second elements
second_elements = [t[1] for t in trustScores]

# Find the minimum and maximum values of the second elements
min_val = min(second_elements)
max_val = max(second_elements)

# Function to normalize a value
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Create a new list of tuples with the second element normalized
normalized_tuples_list = [(t[0], normalize(t[1], min_val, max_val)) for t in trustScores]

# print(normalized_tuples_list)

cursor = db.cursor()

# SQL query with placeholders
query = """
    UPDATE comments_for_certain_symbols
    SET author_weight = %s
    WHERE author_id = %s AND symbol = %s
"""

# Prepare the list of tuples
data_to_update = [(row[1], row[0], symbol) for row in normalized_tuples_list]

# Use executemany to update data
cursor.executemany(query, data_to_update)

# Commit the changes
db.commit()

print(f"{cursor.rowcount} rows updated.")
# print(trustScores)

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
