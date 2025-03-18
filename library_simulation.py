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
        self.acceptable_libraries = []
        self.avoided_libraries = []
        self.schedule = {}  # Schedule with values: "library", "lecture", or None (off campus)
        
        # Load data if provided
        if student_data:
            self.faculty = student_data.get("faculty")
            self.year = student_data.get("year")
            self.preferred_library_id = student_data.get("preferred_library_id")
            
            # Get acceptable and avoided libraries based on faculty
            if self.faculty in self.model.faculty_library_mapping:
                self.acceptable_libraries = self.model.faculty_library_mapping[self.faculty].get("acceptable", [])
                self.avoided_libraries = self.model.faculty_library_mapping[self.faculty].get("avoided", [])
                
            self.schedule = student_data.get("schedule", {})
        else:
            print(f"Error: No student data provided for student {unique_id}")
            
        # Initialize location and status
        self.current_library_id = 'not_in_library'
        self.target_library_id = None
        self.travel_time_remaining = 0
        self.status = "off_campus"  # Can be "in_library", "in_lecture", "traveling", or "off_campus"
        self.attempted_libraries = []  # Track libraries that were attempted but full
                
    def get_schedule_activity(self, current_hour):
        """Get the student's scheduled activity for the current hour"""
        return self.schedule.get(current_hour)
    
    def _find_closest_library(self):
        """Find the closest library by travel time that respects faculty preferences"""
        # First check if preferred library is available
        if self.preferred_library_id not in self.attempted_libraries:
            return self.preferred_library_id
        
        # Get preferred libraries from faculty mapping
        preferred_libraries = []
        if self.faculty in self.model.faculty_library_mapping:
            preferred_libraries = self.model.faculty_library_mapping[self.faculty].get("preferred", [])
        
        # Try other preferred libraries that haven't been attempted yet
        available_preferred = [lib for lib in preferred_libraries 
                             if lib in self.model.libraries 
                             and lib not in self.attempted_libraries 
                             and lib != self.preferred_library_id]
        
        if available_preferred:
            # Find closest preferred library
            preferred_distances = {}
            for library_id in available_preferred:
                try:
                    dist = nx.shortest_path_length(
                        self.model.graph, 
                        self.preferred_library_id, 
                        library_id, 
                        weight='weight'
                    )
                    preferred_distances[library_id] = dist
                except nx.NetworkXNoPath:
                    pass
            
            if preferred_distances:
                return min(preferred_distances, key=preferred_distances.get)
        
        # If no preferred libraries available, try acceptable libraries
        acceptable_distances = {}
        for library_id in self.acceptable_libraries:
            if (library_id in self.model.libraries 
                and library_id not in self.attempted_libraries 
                and library_id != self.preferred_library_id
                and library_id not in preferred_libraries):
                try:
                    # Get shortest path length using travel time weights
                    dist = nx.shortest_path_length(
                        self.model.graph, 
                        self.preferred_library_id, 
                        library_id, 
                        weight='weight'
                    )
                    acceptable_distances[library_id] = dist
                except nx.NetworkXNoPath:
                    # No path exists between these libraries
                    pass
        
        # If we have acceptable options, return the closest one
        if acceptable_distances:
            return min(acceptable_distances, key=acceptable_distances.get)
        
        # If all acceptable libraries are full, only then consider avoided libraries
        avoided_distances = {}
        for library_id in self.model.libraries.keys():
            if (library_id not in self.attempted_libraries and 
                library_id != self.preferred_library_id and
                library_id not in preferred_libraries and
                library_id not in self.acceptable_libraries):
                
                try:
                    dist = nx.shortest_path_length(
                        self.model.graph, 
                        self.preferred_library_id, 
                        library_id, 
                        weight='weight'
                    )
                    avoided_distances[library_id] = dist
                except nx.NetworkXNoPath:
                    pass
        
        # If we have avoided options, return the closest one
        if avoided_distances:
            return min(avoided_distances, key=avoided_distances.get)
        
        # If all libraries have been attempted, reset attempted list and return to preferred
        self.attempted_libraries = []
        return self.preferred_library_id
    
    def step(self):
        """Perform a step in the simulation"""
        current_hour = self.model.get_hour()
        scheduled_activity = self.get_schedule_activity(current_hour)
        
        # Check if traveling
        if self.status == "traveling":
            # Skip travel time if going to/from off campus
            if self.target_library_id == 'not_in_library' or self.current_library_id == 'not_in_library':
                self.travel_time_remaining = 0
                
            # Continue journey
            self.travel_time_remaining -= 1
            if self.travel_time_remaining <= 0:
                # Journey complete - update status and location
                if self.target_library_id == 'not_in_library':
                    # Student is going off campus
                    self.current_library_id = 'not_in_library'
                    self.status = "off_campus"
                    self.attempted_libraries = []  # Reset attempted libraries list
                else:
                    # Student is arriving at a library
                    # Check if they're going to a lecture or library study session
                    if scheduled_activity == "lecture":
                        # Ensure target_library_id is valid before setting current_library_id
                        if self.target_library_id is not None and self.target_library_id in self.model.libraries:
                            self.current_library_id = self.target_library_id
                            self.status = "in_lecture"
                        else:
                            # Invalid target, go off campus instead
                            self.current_library_id = 'not_in_library'
                            self.status = "off_campus"
                    elif scheduled_activity == "library":
                        # Ensure target_library_id is valid
                        if self.target_library_id is None or self.target_library_id not in self.model.libraries:
                            # Try to find another valid library
                            self.attempted_libraries = []  # Reset to try all libraries
                            next_library = self._find_closest_library()
                            
                            # If still no valid library, go off campus
                            if next_library is None:
                                self.current_library_id = 'not_in_library'
                                self.status = "off_campus"
                                return
                            
                            self.target_library_id = next_library
                        
                        # Now check if library is overcrowded
                        if self.model.libraries[self.target_library_id].is_overcrowded():
                            # Library is full, add to attempted libraries
                            self.attempted_libraries.append(self.target_library_id)
                            
                            # Find another library to go to
                            next_library = self._find_closest_library()
                            
                            # If no valid library found, go off campus
                            if next_library is None:
                                self.current_library_id = 'not_in_library'
                                self.status = "off_campus"
                                return
                            
                            current_library = self.target_library_id # Store the current target
                            
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
                
                # Only set target_library_id to None once student has completed their journey
                # and the new status has been set
                self.target_library_id = None
            
            return    
        
        # Logic for students currently in a library
        if self.status == "in_library":
            if scheduled_activity == "lecture":
                self.model.libraries[self.current_library_id].remove_student()
                
                # Safety check for preferred_library_id
                if self.preferred_library_id is None or self.preferred_library_id not in self.model.libraries:
                    if self.model.libraries:
                        self.preferred_library_id = random.choice(list(self.model.libraries.keys()))
                        print(f"Warning: Student {self.unique_id} had invalid preferred library. Reassigned to {self.preferred_library_id}")
                    else:
                        # No libraries available
                        self.status = "off_campus"
                        self.current_library_id = 'not_in_library'
                        return
                
                # check if the student is already in their preferred library
                if self.current_library_id == self.preferred_library_id:
                    # already in preferred library, no travel time required
                    self.status = "in_lecture"
                else:
                    # Not in preferred library, so travel is required
                    self.target_library_id = self.preferred_library_id
                    self.status = "traveling"
                    
                    # set travel time based on graph
                    if self.model.graph.has_edge(self.current_library_id, self.preferred_library_id):
                        travel_time_minutes = self.model.graph[self.current_library_id][self.preferred_library_id]['weight']
                        self.travel_time_remaining = math.ceil(travel_time_minutes / 15)
                    else:
                        # No direct path, set default travel time
                        self.travel_time_remaining = random.randint(2, 5)  # 30-75 min
                    
            elif scheduled_activity != "library":
                # Student needs to leave library (not in their schedule)
                self.target_library_id = 'not_in_library'
                self.model.libraries[self.current_library_id].remove_student()
                self.status = "traveling"
                # No travel time for going off campus
                self.travel_time_remaining = 0
                
        # Logic for students currently in a lecture
        elif self.status == "in_lecture":
            # Safety check - ensure current_library_id is valid
            if self.current_library_id is None or self.current_library_id not in self.model.libraries:
                # Fix the inconsistent state
                if self.preferred_library_id and self.preferred_library_id in self.model.libraries:
                    self.current_library_id = self.preferred_library_id
                    print(f"Warning: Student {self.unique_id} in lecture had invalid library. Reassigned to {self.current_library_id}")
                elif self.model.libraries:
                    self.current_library_id = random.choice(list(self.model.libraries.keys()))
                    print(f"Warning: Student {self.unique_id} in lecture had invalid library. Reassigned to {self.current_library_id}")
                else:
                    # No libraries available at all, send student off campus
                    self.status = "off_campus"
                    self.current_library_id = 'not_in_library'
                    print(f"Warning: No libraries available for Student {self.unique_id} in lecture. Sent off campus.")
                    return
                    
            
            if scheduled_activity == "library":
                # Lecture ended, student wants to study in the library
                if self.model.libraries[self.current_library_id].is_overcrowded():
                    # Preferred library is full, find another one
                    self.attempted_libraries.append(self.current_library_id)
                    next_library = self._find_closest_library()
                    self.target_library_id = next_library
                    self.status = "traveling"
                    
                    # Set travel time based on graph
                    if self.model.graph.has_edge(self.current_library_id, next_library):
                        travel_time_minutes = self.model.graph[self.current_library_id][next_library]['weight']
                        travel_time_steps = math.ceil(travel_time_minutes / 15)
                        self.travel_time_remaining = travel_time_steps
                    else:
                        # No direct path, set default travel time
                        self.travel_time_remaining = random.randint(2, 5)  # 30-75 min
                else:
                    # Preferred library has space, enter it
                    self.model.libraries[self.current_library_id].add_student()
                    self.status = "in_library"
            elif scheduled_activity != "lecture":
                # Lecture ended, student wants to go off campus
                self.target_library_id = 'not_in_library'
                self.status = "traveling"
                # No travel time for going off campus
                self.travel_time_remaining = 0
        
        # Logic for students off campus
        elif self.status == "off_campus":
            if scheduled_activity == "lecture":
                # Student needs to go to a lecture
                self.target_library_id = self.preferred_library_id
                self.status = "traveling"
                # No travel time for coming from off campus
                self.travel_time_remaining = 0
            elif scheduled_activity == "library":
                # Student needs to go to a library - choose the preferred one first
                target_library = self._find_closest_library()
                self.target_library_id = target_library
                self.status = "traveling"
                # No travel time for coming from off campus
                self.travel_time_remaining = 0

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
                 student_data=None,
                 faculty_library_mapping=None):    
        
        super().__init__()
        self.student_count = student_count
        self.current_step = 0
        self.hours_per_step = 0.25  # 15 minutes per step
        self.steps_per_day = int(24 / self.hours_per_step)
        self.day = 0
        
        # Use the provided mapping or a default empty one
        self.faculty_library_mapping = faculty_library_mapping # or {}
        
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
        
            
        # Create libraries (with specific capacities from the CSV)
        self.libraries = {}
        for _, row in self.df_locations.iterrows():
            # Use the capacity from the CSV file instead of random values
            capacity = row['Capacity']  # Get capacity from the CSV
            self.libraries[row['ID']] = Library(row['ID'], row['LibraryName'], capacity)
            
        # Create student scheduler
        self.schedule = RandomActivation(self) # self.schedule is an instance of RandomActivation, which is a scheduler provided by mesa
        
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
                "Students_In_Library": self.get_students_in_library,
                "Students_In_Lecture": self.get_students_in_lecture,
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
        # Calculate hour based on step and start_hour
        return self.start_hour + int((self.current_step % self.steps_per_operating_day) * self.hours_per_step)
    
    def get_library_occupancy(self):
        """Return dictionary of library occupancies"""
        return {lib.name: lib.occupancy for lib in self.libraries.values()}
    
    def get_traveling_students(self):
        """Return count of students who are traveling"""
        return sum(1 for agent in self.schedule.agents if agent.status == "traveling")
    
    def get_students_in_library(self):
        """Return count of students who are in a library"""
        return sum(1 for agent in self.schedule.agents if agent.status == "in_library")

    def get_students_in_lecture(self):
        """Return count of students who are in lecture"""
        return sum(1 for agent in self.schedule.agents if agent.status == "in_lecture")

    def get_students_off_campus(self):
        """Return count of students who are not in any library"""
        return sum(1 for agent in self.schedule.agents if agent.status == "off_campus")

    def step(self):
        """Advance the model by one step (15-minute interval)."""
        self.schedule.step() # Call RandomActivation, which randomly iterates through all the student agents and calls student.step() for each one
        self.current_step += 1
    
        # Collect data on every step (15 minutes)
        self.datacollector.collect(self)
        
        # Still track day changes
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
        
        # Calculate y-axis range for visualization buffer - keep all nodes from being cut off the visualisation
        node_y_values = [pos[node][1] for node in pos]  # Extract all y-values
        min_y = min(node_y_values)
        max_y = max(node_y_values)
        buffer = 0.1 * (max_y - min_y)  # Add 10% padding above the highest node

        
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
        lecture_counts = {}  # Track students in lecture at each building
        
        # Count students in lectures at each library
        for agent in self.schedule.agents:
            if agent.status == "in_lecture":
                if agent.current_library_id in lecture_counts:
                    lecture_counts[agent.current_library_id] += 1
                else:
                    lecture_counts[agent.current_library_id] = 1
        
        for node in self.graph.nodes():
            node_x.append(pos[node][0])
            node_y.append(pos[node][1])
            
            library = self.libraries[node]
            lecture_count = lecture_counts.get(node, 0)
            
            # Include lecture count in node label
            node_labels.append(f"{library.name}<br>Library: {library.occupancy}/{library.capacity}<br>Lecture: {lecture_count}")
            
            occupancy_pct = library.get_occupancy_percentage()
            
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
        students_in_lecture = self.get_students_in_lecture()
        students_traveling = self.get_traveling_students()
        students_off_campus = self.get_students_off_campus()
        
        # Verify total matches
        total_students = students_in_libraries + students_in_lecture + students_traveling + students_off_campus
            
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
            'students_in_lecture': students_in_lecture,
            'students_traveling': students_traveling, 
            'students_off_campus': students_off_campus,
            'total_students': total_students,
            'min_y': min_y, 
            'max_y': max_y,
            'buffer': buffer
        }
    
#---------------------------------------------------------------------------------------------------------------

def run_library_simulation_with_frames(steps=48, student_count=10, update_interval=1, 
                                      start_hour=8, end_hour=20, student_data=None, faculty_library_mapping=None):
    """
    Run the library simulation and create an animation with frames
    Update interval of 1 means create a frame for every step (15 minutes)
    """ 
    # Create the model with specified operating hours and student data
    model = LibraryNetworkModel(
        student_count=student_count,
        start_hour=start_hour,
        end_hour=end_hour,
        student_data=student_data,
        faculty_library_mapping=faculty_library_mapping
    )
    
    # Create a frame for each 15-minute interval
    frames = []
    
    # Calculate total number of 15-minute intervals (4 per hour)
    total_intervals = (end_hour - start_hour) * 4 + 1
    
    # Run simulation for each 15-minute interval
    for step in range(total_intervals):
        # Set the model's time to this step
        model.current_step = step
        
        # Calculate current hour and minutes
        current_hour = start_hour + (step // 4)
        current_minute = (step % 4) * 15
        
        # Get network data for this step
        vis_data = model.get_network_visualization_data()
        
        # Create a frame for this step
        frame = {
            "name": f"step_{step}",
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
                "title": (f"Library Network - Day {vis_data['day']}, {current_hour}:{current_minute:02d}<br>"
                          f"Students: {vis_data['students_in_libraries']} in Libraries, "
                          f"{vis_data['students_in_lecture']} in Lectures, "
                          f"{vis_data['students_traveling']} Traveling, "
                          f"{vis_data['students_off_campus']} Off Campus "
                          f"(Total: {vis_data['total_students']}/{student_count})")
            }
        }
        
        # Add the frame
        frames.append(frame)
        
        # Run the simulation for one step (15 minutes)
        if step < total_intervals - 1:  # Don't step after the last frame
            model.step()
    
    # Create slider steps for all 15-minute intervals
    slider_steps = []
    
    for step in range(total_intervals):
        current_hour = start_hour + (step // 4)
        current_minute = (step % 4) * 15
        
        # Only add hour marks as labeled steps
        if current_minute == 0:
            label = f"{current_hour}:00"
        else:
            # For 15-min intervals, use a small mark "|" instead of a label
            label = "|"
            
        slider_steps.append({
            "args": [
                [f"step_{step}"],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
            ],
            "label": label,
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
                "active": 0,
                "steps": slider_steps,
                "x": 0.1,
                "y": 0,
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "currentvalue": {
                    "visible": True,
                    "prefix": "Time: ",
                    "xanchor": "right",
                    "font": {"size": 16, "color": "#666"}
                }
            }],
            height=1000,
            width=1500, 
            yaxis=dict(range=[vis_data['min_y'] - vis_data['buffer'], 
                      vis_data['max_y'] + vis_data['buffer']])
        ),
        frames=frames
    )
    
    # Show animation
    fig.show(renderer="browser")
    
    return model

