#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated Library Simulation with fixes for student tracking
"""

import mesa
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
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
        self.home_library_id = home_library_id
        self.current_library_id = 'not_in_library'
        self.target_library_id = None
        self.travel_time_remaining = 0
        self.status = "off_campus"  # Can be "in_library", "traveling", or "off_campus"
        
        # If no schedule provided, create a random one
        if schedule is None:
            self.schedule = self._generate_random_schedule()
        else:
            self.schedule = schedule
            
    def _generate_random_schedule(self):
        """Generate a random daily schedule for when student visits libraries"""
        schedule = {}
        # Example: 30% chance of going to a library at each hour between 9am-5pm
        for hour in range(8, 21): # 8am to 8pm
            if random.random() < 0.3:  # 30% chance of scheduling a library visit
                # Randomly choose a library and duration
                target_lib = random.choice(list(self.model.libraries.keys()))
                duration = random.randint(1, 3)  # Stay 1-3 hours
                schedule[hour] = {
                    'library_id': target_lib, 
                    'duration': duration
                }
            elif random.random() < 0.4: # 40% chance of scheduling time outside any library
                schedule[hour] = {
                    'library_id': 'not_in_library',
                    'duration': random.randint(1,3) # stay away for 1-3 hours
                }
        return schedule
    
    def _choose_next_library(self, current_hour):
        """Determine if the student needs to move to a different library or leave entirely"""
        # Students leave libraries and go off campus outside of 8am-8pm
        if current_hour < 8 or current_hour >= 20:
            if self.current_library_id != 'not_in_library':
                return 'not_in_library'
            else:
                return self.current_library_id
            
        # Check if there's a scheduled activity for this hour
        if current_hour in self.schedule:
            target_library = self.schedule[current_hour]['library_id']
            # If target is a real library and current is a real library, check if they're connected
            if (target_library != 'not_in_library' and 
                self.current_library_id != 'not_in_library' and
                not self.model.graph.has_edge(self.current_library_id, target_library)):
                # No direct connection, consider going to 'not_in_library' first
                if random.random() < 0.5:  # 50% chance to go to "not_in_library" instead
                    return 'not_in_library'
            return target_library
        
        # If not in library already, consider returning to one (30% chance each hour)
        if self.current_library_id == 'not_in_library':
            if random.random() < 0.3:
                return random.choice(list(self.model.libraries.keys()))
            return 'not_in_library'
        
        # If student is at capacity library, consider moving
        current_library = self.model.libraries[self.current_library_id]
        if current_library.is_overcrowded() and random.random() < 0.7:
            # 70% chance to leave if library is overcrowded
            # Get only libraries that are connected to current library
            connected_libraries = list(self.model.graph.neighbors(self.current_library_id))
            possible_libraries = [
                lib_id for lib_id in connected_libraries
                if not self.model.libraries[lib_id].is_overcrowded() and lib_id != self.current_library_id
            ]
            
            # 40% chance to leave campus entirely if overcrowded 
            if random.random() < 0.4:
                return 'not_in_library'
            
            if possible_libraries:
                return random.choice(possible_libraries)
        
        # 10% random chance to move libraries if nothing scheduled
        if random.random() < 0.1 and self.status != "traveling":
            # 30% chance to leave campus entirely when moving randomly 
            if random.random() < 0.3:
                return 'not_in_library'
            
            connected_libraries = list(self.model.graph.neighbors(self.current_library_id))
            if connected_libraries:
                return random.choice(connected_libraries)
                
        # Stay at current library
        return self.current_library_id
    
    def step(self):
        """Perform a step in the simulation"""
        current_hour = self.model.get_hour()
        
        # If travelling, continue journey
        if self.status == "traveling":
            self.travel_time_remaining -= 1
            if self.travel_time_remaining <= 0:
                # Journey complete - update status and location
                if self.target_library_id == 'not_in_library':
                    # Student is going off campus
                    self.current_library_id = 'not_in_library'
                    self.status = "off_campus"
                else:
                    # Student is arriving at a library
                    self.current_library_id = self.target_library_id
                    self.model.libraries[self.current_library_id].add_student()
                    self.status = "in_library"
                
                self.target_library_id = None
            return
                    
        # If not travelling, decide whether to move
        next_library_id = self._choose_next_library(current_hour)
        
        # If need to move
        if next_library_id != self.current_library_id:
            self.target_library_id = next_library_id
            
            # Handle movement from library to anywhere (including not_in_library)
            if self.current_library_id != 'not_in_library':
                # Currently in a library, need to leave
                self.model.libraries[self.current_library_id].remove_student()
                self.status = "traveling"
                
                # Set travel time based on destination
                if next_library_id == 'not_in_library':
                    self.travel_time_remaining = random.randint(1, 2)  # 15-30 min to leave
                else:
                    # Check if there's a direct path between libraries
                    if self.model.graph.has_edge(self.current_library_id, next_library_id):
                        # Get travel time from graph
                        travel_time_minutes = self.model.graph[self.current_library_id][next_library_id]['weight']
                        travel_time_steps = math.ceil(travel_time_minutes / 15)
                        self.travel_time_remaining = travel_time_steps
                    else: 
                        # No direct path, set a default travel time
                        self.travel_time_remaining = random.randint(2, 5)  # 30-75 min for indirect route
            else:
                # Handle movement from 'not_in_library' to a real library
                self.status = "traveling"
                self.travel_time_remaining = random.randint(1, 4)  # 15-60 min to arrive

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
        else:
            print(f"Warning: Attempted to remove student from {self.name} but occupancy is already 0!")
            
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
                 student_count, 
                 library_locations_file="library_locations.csv", 
                 library_times_file="library_times.csv",
                 start_hour=8,    # Start at 8am
                 end_hour=20):    # End at 8pm
        
        super().__init__()
        self.student_count = student_count
        self.current_step = 0
        self.hours_per_step = 0.25  # 15 minutes per step
        self.steps_per_day = int(24 / self.hours_per_step)
        self.day = 0
        
        # Add operating hours
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.hours_per_day = end_hour - start_hour
        self.steps_per_operating_day = int(self.hours_per_day / self.hours_per_step)
        
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
            
        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Library_Occupancy": self.get_library_occupancy,
                "Students_Traveling": self.get_traveling_students,
                "Students_Off_Campus": self.get_students_off_campus
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
        return sum(1 for agent in self.schedule.agents if agent.status == "traveling")
    
    def get_students_in_library(self):
        """Return count of students who are in a library"""
        return sum(1 for agent in self.schedule.agents if agent.status == "in_library")

    def get_students_off_campus(self):
        """Return count of students who are not in any library"""
        return sum(1 for agent in self.schedule.agents if agent.status == "off_campus")
    
    def step(self):
        """Advance the model by one step (15-minute interval)."""
        self.schedule.step()
        self.current_step += 1
    
        # If we reach the next hour, collect data
        if self.current_step % 4 == 0:  # Since each step is 15 minutes, 4 steps = 1 hour
            self.datacollector.collect(self)
            if self.current_step % self.steps_per_day == 0:
                self.day += 1
    
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
            
        # Get counts for different student states
        students_in_libraries = self.get_students_in_library()
        students_traveling = self.get_traveling_students()
        students_off_campus = self.get_students_off_campus()
        
        # Verify total matches
        total_students = students_in_libraries + students_traveling + students_off_campus
            
        # Add this information to the title
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
            'hour': current_hour,
            'students_in_libraries': students_in_libraries,
            'students_traveling': students_traveling, 
            'students_off_campus': students_off_campus,
            'total_students': total_students
        }
#---------------------------------------------------------------------------------------------------------------

def run_library_simulation_with_frames(steps=48, student_count=10, update_interval=4, start_hour=8, end_hour=20):
    """
    Run the library simulation and create an animation with frames
    """ 
    # Create the model with specified operating hours
    model = LibraryNetworkModel(
        student_count=student_count,
        start_hour=start_hour,
        end_hour=end_hour
    )
    
    # Force the model's initial hour to match start_hour
    # This ensures we start at the beginning of our desired time range
    model.current_step = start_hour * 4  # 4 steps per hour
    
    # Create a frame for each hour we want to display - include the end hour
    frames = []
    
    # First, run the simulation for each hour we want to display (including end_hour)
    for hour in range(start_hour, end_hour + 1):
        # Set the model's time to this hour
        model.current_step = hour * 4  # 4 steps per hour
        
        # Get network data for this hour
        vis_data = model.get_network_visualization_data()
        
        # Create a frame for this hour
        frame = {
            "name": f"hour_{hour}",
            "data": [
                # Edges
                {
                    "type": "scatter",
                    "x": vis_data['edge_x'],
                    "y": vis_data['edge_y'],
                    "mode": "lines",
                    "line": {"width": 2, "color": "black"},
                    "hoverinfo": "none"
                },
                # Edge labels
                {
                    "type": "scatter",
                    "x": [label[0] for label in vis_data['edge_labels']],
                    "y": [label[1] for label in vis_data['edge_labels']],
                    "mode": "text",
                    "text": [label[2] for label in vis_data['edge_labels']],
                    "textposition": "top center",
                    "hoverinfo": "none"
                },
                # Nodes
                {
                    "type": "scatter",
                    "x": vis_data['node_x'],
                    "y": vis_data['node_y'],
                    "mode": "markers+text",
                    "text": vis_data['node_labels'],
                    "textposition": "top center",
                    "marker": {
                        "size": 25,
                        "color": vis_data['node_colors'],
                        "line": {"width": 2, "color": "black"}
                    }
                }
            ],
            "layout": {
                "title": (f"Library Network - Day {vis_data['day']}, Hour {hour}:00<br>"
                          f"Students: {vis_data['students_in_libraries']} in Libraries, "
                          f"{vis_data['students_traveling']} Traveling, "
                          f"{vis_data['students_off_campus']} Off Campus "
                          f"(Total: {vis_data['total_students']}/{student_count})")
            }
        }
        
        # Add the frame
        frames.append(frame)
        
        # Now run the simulation for update_interval steps to get to the next hour
        for _ in range(update_interval):
            model.step()
    
    # Now create the slider steps - one for each hour, including end_hour
    slider_steps = []
    for i, hour in enumerate(range(start_hour, end_hour + 1)):
        slider_steps.append({
            "args": [
                [f"hour_{hour}"],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
            ],
            "label": f"{hour}:00",
            "method": "animate"
        })
    
    # Create the initial figure using the first frame
    initial_frame = frames[0] if frames else {"data": [], "layout": {"title": "No data"}}
    
    # Create figure with animation
    fig = go.Figure(
        data=initial_frame["data"],
        layout=go.Layout(
            title=f"Library Network Simulation ({start_hour}:00-{end_hour}:00)",
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
                "active": 0,  # Set the initial active frame to 0 (8am)
                "steps": slider_steps,
                "x": 0.1,
                "y": 0,
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "currentvalue": {
                    "visible": True,
                    "prefix": "Hour: ",
                    "xanchor": "right",
                    "font": {"size": 16, "color": "#666"}
                }
            }],
            height=1000,
            width=1500
        ),
        frames=frames
    )
    
    # Show animation
    fig.show(renderer="browser")
    
    return model


