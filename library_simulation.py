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
    def __init__(self, unique_id, model, student_data=None):
        super().__init__(unique_id, model)
        
        # Set default values
        self.faculty = None
        self.year = None
        self.preferred_library_id = None
        self.schedule = {}
        
        # Load data if provided
        if student_data:
            self.faculty = student_data.get("faculty")
            self.year = student_data.get("year")
            self.preferred_library_id = student_data.get("preferred_library_id")
            self.schedule = student_data.get("schedule", {})
        else:
            print(f"Error: No student data provided for student {unique_id}")
            # Note if student data not in place, the simulation will not run - could put dummy values in here
        
        # Initialise location and status
        self.current_library_id = 'not_in_library'
        self.target_library_id = None
        self.travel_time_remaining = 0
        self.status = "off_campus"  # Can be "in_library", "traveling", or "off_campus"
        self.attempted_libraries = []  # Track libraries that were attempted but full
                
    def should_be_studying(self, current_hour):
        """Check if the student should be studying at the current hour"""
        return self.schedule.get(current_hour, False)
    
    def _find_closest_library(self):
        """Find the closest library by travel time that hasn't been attempted"""
        # Check if preferred library exists
        if self.preferred_library_id not in self.model.libraries:
            # If preferred library doesn't exist, choose a random one
            return random.choice(list(self.model.libraries.keys()))
            
        # First check if preferred library is the target
        if not self.preferred_library_id in self.attempted_libraries:
            return self.preferred_library_id
        
        # Use Dijkstra's algorithm to find shortest paths by travel time
        distances = {}  # Dictionary to store shortest distances
        for library_id in self.model.libraries.keys():
            if library_id != self.preferred_library_id:
                try:
                    # Get shortest path length using travel time weights
                    dist = nx.shortest_path_length(
                        self.model.graph, 
                        self.preferred_library_id, 
                        library_id, 
                        weight='weight'
                    )
                    distances[library_id] = dist
                except nx.NetworkXNoPath:
                    # No path exists between these libraries
                    pass
        
        # Sort libraries by distance (travel time)
        sorted_libraries = sorted(distances.keys(), key=lambda x: distances[x])
        
        # Find the first library that hasn't been attempted yet
        for library_id in sorted_libraries:
            if library_id not in self.attempted_libraries:
                return library_id
        
        # If all libraries have been attempted, reset attempted list and return preferred
        self.attempted_libraries = []
        return self.preferred_library_id
    
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
                    self.attempted_libraries = []  # Reset attempted libraries list
                else:
                    # Student is arriving at a library - NOW check if it's overcrowded
                    if self.model.libraries[self.target_library_id].is_overcrowded():
                        # Library is full, add to attempted libraries
                        self.attempted_libraries.append(self.target_library_id)
                        
                        # Find another library to go to
                        next_library = self._find_closest_library()
                        
                        current_library = self.target_library_id # Store the current target before updating it
                        
                        # Set the new target and continue traveling
                        self.target_library_id = next_library
                        self.status = "traveling"
                        
                        # Set travel time based on graph
                        if self.model.graph.has_edge(current_library, next_library):
                            travel_time_minutes = self.model.graph[current_library][next_library]['weight']
                            travel_time_steps = math.ceil(travel_time_minutes / 15)
                            self.travel_time_remaining = travel_time_steps
                        else:
                            # No direct path, set default travel time
                            self.travel_time_remaining = random.randint(2, 5)  # 30-75 min
                    else:
                        # Library has space, enter it
                        self.current_library_id = self.target_library_id
                        self.model.libraries[self.current_library_id].add_student()
                        self.status = "in_library"
                        self.attempted_libraries = []  # Reset attempted libraries list
                
                self.target_library_id = None
            return
        
        # Check if student should be studying at current hour
        should_study = self.should_be_studying(current_hour)
        
        # Logic for students currently in a library
        if self.status == "in_library":
            if not should_study:
                # Student needs to leave library (not in their schedule)
                self.target_library_id = 'not_in_library'
                self.model.libraries[self.current_library_id].remove_student()
                self.status = "traveling"
                self.travel_time_remaining = random.randint(1, 2)  # 15-30 min to leave
            # Removed the logic for students leaving an overcrowded library since
            # we want them to stay once they have a seat
        
        # Logic for students off campus
        elif self.status == "off_campus":
            if should_study:
                # Student needs to go to a library - choose the preferred one first
                # They don't know it's full until they get there
                target_library = self._find_closest_library()
                self.target_library_id = target_library
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
                 student_count=10, 
                 library_locations_file="library_locations.csv", 
                 library_times_file="library_times.csv",
                 start_hour=8,    # Start at 8am
                 end_hour=20,     # End at 8pm
                 student_data=None):    
        
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
            # Get student data if available
            student_info = None
            if student_data and i in student_data:
                student_info = student_data[i]
                
            # Create the student with data
            student = Student(i, self, student_info)
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
        self.schedule.step() # self.schedule is an instance of RandomActivation, which is a scheduler that calls the step() method for each agent
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

def run_library_simulation_with_frames(steps=48, student_count=10, update_interval=4, 
                                        start_hour=8, end_hour=20, student_data=None):
    """
    Run the library simulation and create an animation with frames
    """ 
    # Create the model with specified operating hours and student data
    model = LibraryNetworkModel(
        student_count=student_count,
        start_hour=start_hour,
        end_hour=end_hour,
        student_data=student_data
    )
    
    # Force the model's initial hour to match start_hour - ensure start at beginning of desired time range
    # model.current_step = start_hour * 4  # 4 steps per hour
    
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
