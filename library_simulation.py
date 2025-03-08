#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:20:30 2025

@author: lewisvaughan
"""

import mesa
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
# from collections import defaultdict
import time
import math

# Set default renderer to single browser window
pio.renderers.default = "browser"
#---------------------------------------------------------------------------------------------------------------
class Student(mesa.Agent):
    """
    A student agent that moves between libraries based on a schedule and preferences.
    """
    def __init__(self, unique_id, model, home_library_id, schedule=None):
        super().__init__(unique_id, model)
        self.current_library_id = home_library_id
        self.target_library_id = None
        self.traveling = False
        self.travel_time_remaining = 0
        
        # If no schedule provided, create a random one
        if schedule is None:
            self.schedule = self._generate_random_schedule()
        else:
            self.schedule = schedule
            
    def _generate_random_schedule(self):
        """Generate a random daily schedule for when student visits libraries"""
        schedule = {}
        # Example: 30% chance of going to a library at each hour between 9am-5pm
        for hour in range(9, 18):
            if random.random() < 0.3:  # 30% chance of scheduling a library visit
                # Randomly choose a library and duration
                target_lib = random.choice(list(self.model.libraries.keys()))
                duration = random.randint(1, 3)  # Stay 1-3 hours
                schedule[hour] = {
                    'library_id': target_lib, 
                    'duration': duration
                }
        return schedule
    
    def choose_next_library(self, current_hour):
        """Determine if the student needs to move to a different library based on schedule"""
        # Check if there's a scheduled activity for this hour
        if current_hour in self.schedule:
            return self.schedule[current_hour]['library_id']
        
        # If student is at capacity library, consider moving
        current_library = self.model.libraries[self.current_library_id]
        if current_library.is_overcrowded() and random.random() < 0.7:
            # 70% chance to leave if library is overcrowded
            possible_libraries = [
                lib_id for lib_id, library in self.model.libraries.items() 
                if not library.is_overcrowded() and lib_id != self.current_library_id
            ]
            if possible_libraries:
                return random.choice(possible_libraries)
        
        # 10% random chance to move libraries if nothing scheduled
        if random.random() < 0.1 and not self.traveling:
            connected_libraries = list(self.model.graph.neighbors(self.current_library_id))
            if connected_libraries:
                return random.choice(connected_libraries)
                
        # Stay at current library
        return self.current_library_id
    
    def step(self):
        """Perform a step in the simulation"""
        current_hour = self.model.get_hour()
        
        # If travelling, continue journey
        if self.traveling:
            self.travel_time_remaining -= 1
            if self.travel_time_remaining <= 0:
                # Arrived at destination
                self.traveling = False
                old_library = self.model.libraries[self.current_library_id]
                old_library.remove_student()
                
                self.current_library_id = self.target_library_id
                new_library = self.model.libraries[self.current_library_id]
                new_library.add_student()
                self.target_library_id = None
            return
            
        # If not travelling, decide whether to move
        next_library_id = self.choose_next_library(current_hour)
        
        # If need to move
        if next_library_id != self.current_library_id:
            # Check if there's a direct path
            if self.model.graph.has_edge(self.current_library_id, next_library_id):
                # Get travel time in minutes from the graph - under weights
                travel_time_minutes = self.model.graph[self.current_library_id][next_library_id]['weight']
                
                # Convert minutes to simulation steps (rounded up)
                # Each step is 15 minutes (0.25 hours)
                travel_time_steps = math.ceil(travel_time_minutes / 15)
                
                self.traveling = True
                self.travel_time_remaining = travel_time_steps
                self.target_library_id = next_library_id
#---------------------------------------------------------------------------------------------------------------

class Library(object):
    """
    A library with capacity and current occupancy.
    """
    def __init__(self, library_id, name, capacity):
        self.id = library_id
        self.name = name
        self.capacity = capacity
        self.occupancy = 0
        
    def add_student(self):
        self.occupancy += 1
        
    def remove_student(self):
        if self.occupancy > 0:
            self.occupancy -= 1
            
    def is_overcrowded(self):
        return self.occupancy >= self.capacity * 0.9  # 90% of capacity is "overcrowded"
        
    def get_occupancy_percentage(self):
        return (self.occupancy / self.capacity) * 100 if self.capacity > 0 else 0
#---------------------------------------------------------------------------------------------------------------

class LibraryNetworkModel(mesa.Model):
    """
    Model to simulate student movement between libraries.
    """
    def __init__(self, 
                 student_count=500, 
                 library_locations_file="library_locations.csv", 
                 library_times_file="library_times.csv"):
        
        super().__init__()
        self.student_count = student_count
        self.current_step = 0
        self.hours_per_step = 0.25  # 15 minutes per step
        self.steps_per_day = int(24 / self.hours_per_step)
        self.day = 0
        
        # Load library data
        self.df_locations = pd.read_csv(library_locations_file)
        self.df_times = pd.read_csv(library_times_file, index_col=0)
        self.df_locations.columns = self.df_locations.columns.str.strip()
        
        # Create network graph
        self.graph = self._create_graph()
        
        # Create libraries (with estimated capacities for now)
        self.libraries = {}
        for _, row in self.df_locations.iterrows():
            # Example capacity calculation (replace with real data when available)
            capacity = random.randint(50, 200)  # Random capacity between 50-200
            self.libraries[row['ID']] = Library(row['ID'], row['LibraryName'], capacity)
            
        # Create student scheduler
        self.schedule = RandomActivation(self)
        
        # Create students
        for i in range(self.student_count):
            # Randomly assign a home library
            home_library_id = random.choice(list(self.libraries.keys()))
            student = Student(i, self, home_library_id)
            self.schedule.add(student)
            # Add student to their initial library
            self.libraries[home_library_id].add_student()
            
        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Library_Occupancy": self.get_library_occupancy,
                "Students_Traveling": self.get_traveling_students
            }
        )
        
    def _create_graph(self):
        """Create network graph from the loaded data"""
        G = nx.Graph()
        
        # Add weighted edges from time matrix
        for i in self.df_times.index:
            for j in self.df_times.columns:
                if not np.isnan(self.df_times.loc[i, j]) and i != j:
                    G.add_edge(int(i), int(j), weight=self.df_times.loc[i, j])
                    
        return G
    
    def get_hour(self):
        """Get the current hour of the day (0-23)"""
        return int((self.current_step % self.steps_per_day) * self.hours_per_step)
    
    def get_library_occupancy(self):
        """Return dictionary of library occupancies"""
        return {lib.name: lib.occupancy for lib in self.libraries.values()}
    
    def get_traveling_students(self):
        """Return count of students who are traveling"""
        return sum(1 for agent in self.schedule.agents if agent.traveling)
    
    def step(self):
        """Advance the model by one step"""
        self.schedule.step()
        self.current_step += 1
        
        # New day when we reach 24 hours
        if self.current_step % self.steps_per_day == 0:
            self.day += 1
            
        self.datacollector.collect(self)
        
    def get_network_visualization_data(self):
        """Get data for network visualization"""
        # Normalize lat/lon for visualization
        min_lon, min_lat = self.df_locations["Longitude"].min(), self.df_locations["Latitude"].min()
        self.df_locations["x"] = self.df_locations["Longitude"] - min_lon  
        self.df_locations["y"] = self.df_locations["Latitude"] - min_lat   
        
        # Dictionary with ID as key and normalized location as values
        pos = {row["ID"]: (row["x"], row["y"]) for _, row in self.df_locations.iterrows()}
        
        # Create edges for Plotly
        edge_x, edge_y, edge_labels = [], [], []
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_labels.append(((x0 + x1) / 2, (y0 + y1) / 2, f"{edge[2]['weight']}"))
            
        # Create nodes data
        node_x = []
        node_y = []
        node_labels = []
        node_colors = []
        
        for node in self.graph.nodes():
            node_x.append(pos[node][0])
            node_y.append(pos[node][1])
            
            library = self.libraries[node]
            occupancy_pct = library.get_occupancy_percentage()
            node_labels.append(f"{library.name}<br>{library.occupancy}/{library.capacity}")
            
            # Color node based on occupancy percentage
            if occupancy_pct < 50:
                color = 'green'  # Low occupancy
            elif occupancy_pct < 90:
                color = 'orange'  # Medium occupancy
            else:
                color = 'red'  # High occupancy
                
            node_colors.append(color)
            
        # Get current time info
        current_hour = self.get_hour()
        
        return {
            'edge_x': edge_x,
            'edge_y': edge_y,
            'edge_labels': edge_labels,
            'node_x': node_x,
            'node_y': node_y,
            'node_labels': node_labels,
            'node_colors': node_colors,
            'day': self.day,
            'hour': current_hour
        }
#---------------------------------------------------------------------------------------------------------------

def run_library_simulation_with_frames(steps=96, student_count=500, update_interval=4):
    """
    Run the library simulation and create an animation with frames
    
    Parameters:
    -----------
    steps: int
        Number of simulation steps to run
    student_count: int
        Number of students in the simulation
    update_interval: int
        How many steps to include in each frame
    """
    # Create the model
    model = LibraryNetworkModel(student_count=student_count)
    
    # Run simulation and collect frames
    frames = []
    
    for i in range(steps):
        if i % update_interval == 0:
            # Get network data for this step
            vis_data = model.get_network_visualization_data()
            
            # Get current hour for frame name
            current_hour = vis_data['hour']
            frame_day = vis_data['day']
            
            # Create frame
            frame = {
                "name": f"hour_{frame_day * 24 + current_hour}",  # Use absolute hour count for name
                "data": [
                    # Edges (don't change)
                    {
                        "type": "scatter",
                        "x": vis_data['edge_x'],
                        "y": vis_data['edge_y']
                    },
                    # Edge labels (don't change)
                    {
                        "type": "scatter",
                        "x": [label[0] for label in vis_data['edge_labels']],
                        "y": [label[1] for label in vis_data['edge_labels']],
                        "text": [label[2] for label in vis_data['edge_labels']]
                    },
                    # Nodes (update colors and labels)
                    {
                        "type": "scatter",
                        "x": vis_data['node_x'],
                        "y": vis_data['node_y'],
                        "text": vis_data['node_labels'],
                        "marker": {"color": vis_data['node_colors']}
                    }
                ],
                "layout": {
                    "title": f"Library Network - Day {vis_data['day']}, Hour {vis_data['hour']}:00"
                }
            }
            frames.append(frame)
        
        # Run simulation step
        model.step()
    
    # Get initial data for the base figure
    initial_data = model.get_network_visualization_data()
    
    # Create a dictionary to map hours to frame indices
    hour_to_frame = {}
    for i, frame in enumerate(frames):
        hour = int(frame["name"].split("_")[1])
        hour_to_frame[hour] = i
    
    # Get all unique hours in the simulation
    unique_hours = sorted(hour_to_frame.keys())
    
    # Create figure with animation
    fig = go.Figure(
        data=[
            # Edges
            go.Scatter(
                x=initial_data['edge_x'],
                y=initial_data['edge_y'],
                mode='lines',
                line=dict(width=2, color='black'),
                hoverinfo='none'
            ),
            # Edge labels
            go.Scatter(
                x=[label[0] for label in initial_data['edge_labels']],
                y=[label[1] for label in initial_data['edge_labels']],
                mode="text",
                text=[label[2] for label in initial_data['edge_labels']],
                textposition="top center",
                hoverinfo='none'
            ),
            # Nodes
            go.Scatter(
                x=initial_data['node_x'],
                y=initial_data['node_y'],
                mode='markers+text',
                text=initial_data['node_labels'],
                textposition="top center",
                marker=dict(
                    size=25,
                    color=initial_data['node_colors'],
                    line=dict(width=2, color='black')
                )
            )
        ],
        layout=go.Layout(
            title="Library Network Simulation",
            showlegend=False,
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [
                            [f"hour_{hour}"],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
                        ],
                        "label": f"Hour {hour % 24}", # Show time of day (0-23)
                        "method": "animate"
                    }
                    for hour in unique_hours  # Use all available hours
                ],
                "x": 0.1,
                "y": 0,
                "pad": {"b": 10, "t": 50},
                "len": 0.9
            }],
            height=1000,
            width=1500
        ),
        frames=frames
    )
    
    # Show animation
    fig.show(renderer="browser")
    
    return model