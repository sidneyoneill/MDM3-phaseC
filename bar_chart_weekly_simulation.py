#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 19:00:43 2025

@author: lewisvaughan
"""

import pandas as pd
import ast
import plotly.graph_objs as go

# Load the csv containing library names and capacities
library_data = pd.read_csv("library_locations.csv")
library_names = library_data['LibraryName '].tolist()
library_capacities = library_data.set_index("LibraryName ")["Capacity"].to_dict()

# Load the simulation results
results = pd.read_csv("all_simulation_runs.csv")['Library_Occupancy']

# Initialize the new dataframe
occupancy_df = pd.DataFrame(index=range(240), columns=library_names)

# Fill the dataframe with occupancy percentages
for i in range(240):
    library_dict = ast.literal_eval(results.iloc[i])  # Convert string to dictionary
    for library in library_names:
        capacity = library_capacities.get(library, 1)  # Use 1 to avoid division by zero
        occupancy = library_dict.get(library, 0)
        occupancy_df.loc[i, library] = round((occupancy / capacity) * 100, 2) # 2 d.p

#---------------------------------------------------------------------------------------------------

# Generate time labels
def generate_time_labels():
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    times = ['8:00 AM', '8:15 AM', '8:30 AM', '8:45 AM', '9:00 AM', 
             '9:15 AM', '9:30 AM', '9:45 AM', '10:00 AM', '10:15 AM', 
             '10:30 AM', '10:45 AM', '11:00 AM', '11:15 AM', '11:30 AM', 
             '11:45 AM', '12:00 PM', '12:15 PM', '12:30 PM', '12:45 PM', 
             '1:00 PM', '1:15 PM', '1:30 PM', '1:45 PM', '2:00 PM', 
             '2:15 PM', '2:30 PM', '2:45 PM', '3:00 PM', '3:15 PM', 
             '3:30 PM', '3:45 PM', '4:00 PM', '4:15 PM', '4:30 PM', 
             '4:45 PM', '5:00 PM', '5:15 PM', '5:30 PM', '5:45 PM', 
             '6:00 PM', '6:15 PM', '6:30 PM', '6:45 PM', '7:00 PM', 
             '7:15 PM', '7:30 PM', '7:45 PM', '8:00 PM']
    
    full_labels = []
    for day in days:
        for time in times:
            full_labels.append(f"{day} {time}")
    
    return full_labels

# Create animated bar chart
def create_library_occupancy_animation(occupancy_df):
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
              '#bcbd22', '#17becf']
    
    # Prepare frames for animation
    frames = []
    for i in range(len(occupancy_df)):
        frame_data = []
        for j, library in enumerate(occupancy_df.columns):
            frame_data.append(
                go.Bar(
                    x=[library], 
                    y=[occupancy_df.iloc[i][library]], 
                    text=[f"{occupancy_df.iloc[i][library]:.1f}%"],
                    textposition='auto',
                    marker_color=colors[j % len(colors)]
                )
            )
        
        frames.append(go.Frame(
            data=frame_data,
            name=f'frame{i}',
            layout=go.Layout(title=f'Library Occupancy: {generate_time_labels()[i]}')
        ))
    
    # Initial plot
    initial_data = []
    for j, library in enumerate(occupancy_df.columns):
        initial_data.append(
            go.Bar(
                x=[library], 
                y=[occupancy_df.iloc[0][library]], 
                text=[f"{occupancy_df.iloc[0][library]:.1f}%"],
                textposition='auto',
                marker_color=colors[j % len(colors)]
            )
        )
    
    # Create figure with animation
    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title='Library Occupancy: Monday 8:00 AM',
            yaxis=dict(range=[0, 100], title='Occupancy (%)'),
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 50, 'redraw': True},
                                        'fromcurrent': True, 
                                        'transition': {'duration': 0}}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                          'mode': 'immediate',
                                          'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        ),
        frames=frames
    )
    
    # Adjust layout
    fig.update_layout(
        height=800, 
        width=1200, 
        title_x=0.5,
        xaxis_title='Libraries',
        yaxis_title='Occupancy (%)',
        yaxis=dict(range=[0, 110]),  # Give 10% extra space at the top
        margin=dict(t=100)  # Add more top margin
    )
    
    return fig

fig = create_library_occupancy_animation(occupancy_df)
fig.show(renderer="browser")