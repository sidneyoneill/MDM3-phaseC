#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:55:41 2025

@author: lewisvaughan
"""

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np

# Load the CSV files
df_locations = pd.read_csv("library_locations.csv")  # Library locations
df_times = pd.read_csv("library_times.csv", index_col=0)  # time matrix

df_locations.columns = df_locations.columns.str.strip()


# make sure these are column headings in the loaded dataframe
expected_columns = {"ID", "Latitude", "Longitude", "LibraryName"} 
if not expected_columns.issubset(set(df_locations.columns)):
    raise ValueError(f"CSV file has incorrect column names! Found columns: {df_locations.columns}")


G = nx.Graph() # create a graph object 


# Add weighted edges from time matrix
for i in df_times.index:
    for j in df_times.columns:
        if not np.isnan(df_times.loc[i, j]) and i != j:  # if there's a valid connection, add edge to graph
            G.add_edge(int(i), int(j), weight=df_times.loc[i, j])

# Normalize lat/lon for visualization - origin (0,0) of graph is at minimum latitude and minimum longitude
min_lon, min_lat = df_locations["Longitude"].min(), df_locations["Latitude"].min()
df_locations["x"] = df_locations["Longitude"] - min_lon  
df_locations["y"] = df_locations["Latitude"] - min_lat   


# Dictionary with ID as key and normalised location as values
pos = {row["ID"]: (row["x"], row["y"]) for _, row in df_locations.iterrows()}

# Create edges for Plotly (with weights being walk times between libraries)
edge_x, edge_y, edge_labels = [], [], []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_labels.append(((x0 + x1) / 2, (y0 + y1) / 2, f"{edge[2]['weight']}")) #place time at midpoint of edge line

edge_trace = go.Scatter(x=edge_x, y=edge_y, 
                        line=dict(width=2, color='black'),
                        mode='lines') 

# Create nodes for Plotly
node_x = [pos[node][0] for node in G.nodes()] # each node has x and y normalised position
node_y = [pos[node][1] for node in G.nodes()]
id_to_name = {row["ID"]: row["LibraryName"] for _, row in df_locations.iterrows()} # dict (key:values) as (ID: library name)
node_labels = [f"{node}: {id_to_name[node]}" for node in G.nodes()] 

node_trace = go.Scatter(x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_labels,
                        textposition="top center",
                        marker=dict(size=25, color='blue'))

# Add edge weight labels
edge_label_trace = go.Scatter(
    x=[label[0] for label in edge_labels],
    y=[label[1] for label in edge_labels],
    mode="text",
    text=[label[2] for label in edge_labels],
    textposition="top center"
)

# Create Plotly figure
fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace])
fig.show(renderer="browser")  # Opens in web browser



