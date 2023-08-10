import os

from igraph import *
import random
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from sqlalchemy import create_engine, text
from data_collection.db_connection import db

query = f"""
SELECT parent_author_id, author_id, FROM_UNIXTIME(created_utc) as time 
FROM comments_for_certain_symbols 
WHERE symbol = "tsla" 
  AND parent_author_id != "None" 
  AND author_id != "None"
having time < "2023-01-01"
ORDER BY RAND();
"""

db_connection_str = 'mysql+pymysql://root:'+ os.getenv('DB_PASSWORD') +'@localhost/p4p'
db_connection = create_engine(db_connection_str)
df = []

with db_connection.connect() as conn:
    df = pd.read_sql(text(query), conn)
    print(df)


print(len(df))

# convert unix timestamp to datetime
df['time'] = pd.to_datetime(df['time'], unit='s')

# set the datetime object to UTC
df['time'] = df['time'].dt.tz_localize('UTC')

# convert the UTC datetime to 'America/New_York' timezone
df['time'] = df['time'].dt.tz_convert('America/New_York')
df = df.sort_values('time')




# Define your time intervals. Here we use monthly frequency
time_intervals = pd.date_range(start=df['time'].min(), end=df['time'].max(), freq='M')

graphs = []
for start, end in zip(time_intervals[:-1], time_intervals[1:]):
    mask = (df['time'] >= start) & (df['time'] < end)
    interval_df = df.loc[mask]

    G = nx.from_pandas_edgelist(interval_df, 'parent_author_id', 'author_id')
    graphs.append(G)

# Create empty dataframes for storing graph stats
node_df = pd.DataFrame(columns=['time', 'nodes'])
edge_df = pd.DataFrame(columns=['time', 'edges'])
degree_df = pd.DataFrame(columns=['time', 'average_degree'])

for i, G in enumerate(graphs):
    num_nodes = G.order()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values())/num_nodes if num_nodes != 0 else 0

    # Add the stats to their respective dataframes
    node_df = node_df.append({'time': time_intervals[i], 'nodes': num_nodes}, ignore_index=True)
    edge_df = edge_df.append({'time': time_intervals[i], 'edges': num_edges}, ignore_index=True)
    degree_df = degree_df.append({'time': time_intervals[i], 'average_degree': avg_degree}, ignore_index=True)

# Now you can plot the dataframes
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(node_df['time'], node_df['nodes'], marker='o')
plt.title('Number of Nodes Over Time')
plt.ylabel('Number of Nodes')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(edge_df['time'], edge_df['edges'], marker='o', color='red')
plt.title('Number of Edges Over Time')
plt.ylabel('Number of Edges')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(degree_df['time'], degree_df['average_degree'], marker='o', color='green')
plt.title('Average Degree Over Time')
plt.xlabel('Time')
plt.ylabel('Average Degree')
plt.grid(True)

plt.tight_layout()
plt.show()
