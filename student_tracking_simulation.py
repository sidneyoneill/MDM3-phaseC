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
        
        # Determine the starting point for distance calculations
        origin = self.current_library_id if self.current_library_id != 'not_in_library' else self.preferred_library_id
        
        # Get preferred libraries from faculty mapping
        preferred_libraries = self.model.faculty_library_mapping[self.faculty].get("preferred", [])
        
        # Try other preferred libraries that haven't been attempted yet
        available_preferred = [lib for lib in preferred_libraries 
                             if lib in self.model.libraries 
                             and lib not in self.attempted_libraries 
                             and lib != self.preferred_library_id]
        
        if available_preferred and origin != 'not_in_library':
            # Find closest preferred library
            preferred_distances = {}
            for library_id in available_preferred:
                try:
                    dist = nx.shortest_path_length(
                        self.model.graph, 
                        origin, 
                        library_id, 
                        weight='weight'
                    )
                    preferred_distances[library_id] = dist
                except nx.NetworkXNoPath:
                    pass
            
            if preferred_distances:
                return min(preferred_distances, key=preferred_distances.get)
        elif available_preferred: # should not come in here - if off campus, their attempted library list should be empty
            # If student is off-campus, just choose a random preferred library
            return random.choice(available_preferred)
        
        # If no preferred libraries available, try acceptable libraries
        acceptable_distances = {}
        for library_id in self.acceptable_libraries:
            if (library_id in self.model.libraries 
                and library_id not in self.attempted_libraries 
                and library_id != self.preferred_library_id
                and library_id not in preferred_libraries):
                
                if origin != 'not_in_library':
                    try:
                        # Get shortest path length using travel time weights
                        dist = nx.shortest_path_length(
                            self.model.graph, 
                            origin, 
                            library_id, 
                            weight='weight'
                        )
                        acceptable_distances[library_id] = dist
                    except nx.NetworkXNoPath:
                        # No path exists between these libraries
                        pass
                else:
                    # If student is off-campus, add library with a placeholder distance
                    acceptable_distances[library_id] = 0
        
        # If we have acceptable options, return the closest one
        if acceptable_distances:
            if origin != 'not_in_library':
                return min(acceptable_distances, key=acceptable_distances.get)
            else:
                # If student is off-campus, just choose a random acceptable library
                return random.choice(list(acceptable_distances.keys()))
        
        # If all acceptable libraries are full, only then consider avoided libraries
        avoided_distances = {}
        for library_id in self.model.libraries.keys():
            if (library_id not in self.attempted_libraries and 
                library_id != self.preferred_library_id and
                library_id not in preferred_libraries and
                library_id not in self.acceptable_libraries):
                
                if origin != 'not_in_library':
                    try:
                        dist = nx.shortest_path_length(
                            self.model.graph, 
                            origin, 
                            library_id, 
                            weight='weight'
                        )
                        avoided_distances[library_id] = dist
                    except nx.NetworkXNoPath:
                        pass
                else:
                    # If student is off-campus, add library with a placeholder distance
                    avoided_distances[library_id] = 0
        
        # If we have avoided options, return the closest one
        if avoided_distances:
            if origin != 'not_in_library':
                return min(avoided_distances, key=avoided_distances.get)
            else:
                # If student is off-campus, just choose a random avoided library
                return random.choice(list(avoided_distances.keys()))
        
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
                    self.target_library_id = None
                else:
                    # Student is arriving at a library
                    # Check if they're going to a lecture or library study session
                    if scheduled_activity == "lecture":
                        # Ensure target_library_id is valid before setting current_library_id
                        if self.target_library_id is not None and self.target_library_id in self.model.libraries:
                            self.current_library_id = self.target_library_id
                            self.status = "in_lecture"
                            self.target_library_id = None
                        else:
                            # Invalid target, go off campus instead
                            self.current_library_id = 'not_in_library'
                            self.status = "off_campus"
                            self.target_library_id = None
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
                            
                            current_library = self.target_library_id
                            
                            # Set the new target and continue travelling
                            self.target_library_id = next_library
                            self.status = "traveling"
                            
                            # Set travel time based on graph
                            if self.model.graph.has_edge(current_library, next_library):
                                travel_time_minutes = self.model.graph[current_library][next_library]['weight']
                                travel_time_steps = math.ceil(travel_time_minutes / 5)
                                self.travel_time_remaining = travel_time_steps
                            else:
                                # No direct path, set default travel time
                                self.travel_time_remaining = random.randint(6, 15)  # 30-75 min
                        else:
                            # Library has space, enter it
                            self.current_library_id = self.target_library_id
                            self.model.libraries[self.current_library_id].add_student()
                            self.status = "in_library"
                            self.attempted_libraries = []  # Reset attempted libraries list
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
                        self.travel_time_remaining = math.ceil(travel_time_minutes / 5)
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
                        travel_time_steps = math.ceil(travel_time_minutes / 5)
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
        self.hours_per_step = 5/60  # 5 minutes
        self.steps_per_day = int(24 / self.hours_per_step)
        self.day = 0
        
        # Use the provided mapping or a default empty one
        self.faculty_library_mapping = faculty_library_mapping 
        
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
        step_hours = self.current_step * self.hours_per_step
        return self.start_hour + int(step_hours)
    
    def get_minute(self):
        """Get the current minute (0-59)"""
        step_hours = self.current_step * self.hours_per_step  # Convert steps to hours
        full_hours = int(step_hours)  # Integer part is hours
        minutes = (step_hours - full_hours) * 60  # Fractional part converted to minutes
        return int(minutes)
    
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
    
    def get_network_visualization_data(self, tracked_student_id=0):
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
            
        # Get info about the tracked student
        tracked_student_info = {}
        tracked_student_pos = None
        
        # Find the tracked student
        for agent in self.schedule.agents:
            if agent.unique_id == tracked_student_id:
                # Get the current library name
                current_library_name = "Not in a library"
                if agent.current_library_id != 'not_in_library' and agent.current_library_id in self.libraries:
                    current_library_name = self.libraries[agent.current_library_id].name
                    
                # Get the target library name
                target_library_name = "None"
                if agent.target_library_id and agent.target_library_id != 'not_in_library' and agent.target_library_id in self.libraries:
                    target_library_name = self.libraries[agent.target_library_id].name
                    
                # Get the preferred library name
                preferred_library_name = "None"
                if agent.preferred_library_id and agent.preferred_library_id in self.libraries:
                    preferred_library_name = self.libraries[agent.preferred_library_id].name
                    
                # Calculate the tracked student position - handle all statuses
                if agent.status == "in_library" or agent.status == "in_lecture":
                    # Student is in a library or lecture - position them at the library
                    if agent.current_library_id in pos:
                        tracked_student_pos = pos[agent.current_library_id]
                elif agent.status == "traveling":
                    if (agent.current_library_id != 'not_in_library' and 
                        agent.target_library_id != 'not_in_library' and
                        agent.current_library_id in pos and 
                        agent.target_library_id in pos):
                    
                        # Student is traveling - need to calculate their position on the edge
                        if agent.current_library_id != 'not_in_library' and agent.target_library_id != 'not_in_library':
                            # Going between two libraries
                            if agent.current_library_id in pos and agent.target_library_id in pos:
                                start_pos = pos[agent.current_library_id]
                                end_pos = pos[agent.target_library_id]
                                
                                # Get original travel time from graph (or calculate a default)
                                if self.graph.has_edge(agent.current_library_id, agent.target_library_id):
                                    total_travel_time = math.ceil(self.graph[agent.current_library_id][agent.target_library_id]['weight'] / 5)
                                else:
                                    # Default if no direct path
                                    total_travel_time = 5
                                
                                # Calculate progress along path (0 to 1)
                                if total_travel_time > 0:
                                    progress = 1 - (agent.travel_time_remaining / total_travel_time)
                                else:
                                    progress = 0.5
                                
                                # Ensure progress is between 0 and 1
                                progress = max(0, min(1, progress))
                                
                                # Interpolate position
                                x = start_pos[0] + progress * (end_pos[0] - start_pos[0])
                                y = start_pos[1] + progress * (end_pos[1] - start_pos[1])
                                
                                tracked_student_pos = (x, y)
                                
                    # Handle case where student just attempted a library but it was full
                    elif agent.attempted_libraries and agent.target_library_id != 'not_in_library':
                        # Student tried a library that was full and is now going to another one
                        # Use the last attempted library as the starting point
                        last_attempted = agent.attempted_libraries[-1]
                        if last_attempted in pos and agent.target_library_id in pos:
                            start_pos = pos[last_attempted]
                            end_pos = pos[agent.target_library_id]
                            
                            # Calculate progress (similar to existing code)
                            if agent.travel_time_remaining > 0:
                                progress = 1 - (agent.travel_time_remaining / 5)  # Simplified for clarity
                            else:
                                progress = 0.5
                                
                            progress = max(0, min(1, progress))
                            
                            # Interpolate position
                            x = start_pos[0] + progress * (end_pos[0] - start_pos[0])
                            y = start_pos[1] + progress * (end_pos[1] - start_pos[1])
                            
                            tracked_student_pos = (x, y)        
                            
                    elif agent.current_library_id == 'not_in_library' and agent.target_library_id != 'not_in_library':
                        # Coming from off-campus to a library - immediately place them at target library
                        if agent.target_library_id in pos:
                            tracked_student_pos = pos[agent.target_library_id]

                    elif agent.current_library_id != 'not_in_library' and agent.target_library_id == 'not_in_library':
                        # Going from a library to off-campus - not show the student at all - set tracked_student_pos to None
                        tracked_student_pos = None
                        
                # Save student info
                tracked_student_info = {
                    'faculty': agent.faculty,
                    'year': agent.year,
                    'status': agent.status,
                    'current_library': current_library_name,
                    'target_library': target_library_name,
                    'preferred_library': preferred_library_name,
                    'schedule': agent.schedule,
                    'attempted_libraries': agent.attempted_libraries,  # Add this to show libraries that were full
                    'travel_time_remaining': agent.travel_time_remaining  # Add travel time for context
                }
        # Get counts for different student states
        students_in_libraries = self.get_students_in_library()
        students_in_lecture = self.get_students_in_lecture()
        students_traveling = self.get_traveling_students()
        students_off_campus = self.get_students_off_campus()
        
        # Verify total matches
        total_students = students_in_libraries + students_in_lecture + students_traveling + students_off_campus
            
        return {
        'edge_x': edge_x,
        'edge_y': edge_y,
        'edge_labels': edge_labels,
        'node_x': node_x,
        'node_y': node_y,
        'node_labels': node_labels,
        'node_colors': node_colors,
        'day': self.day,
        'hour': self.get_hour(),
        'minute': self.get_minute(),
        'students_in_libraries': students_in_libraries,
        'students_in_lecture': students_in_lecture,
        'students_traveling': students_traveling, 
        'students_off_campus': students_off_campus,
        'total_students': total_students,
        'min_y': min_y, 
        'max_y': max_y,
        'buffer': buffer, 
        'tracked_student': tracked_student_info,
        'tracked_student_pos': tracked_student_pos
    }
    
#---------------------------------------------------------------------------------------------------------------

def run_library_simulation_with_frames(steps=144, student_count=10, update_interval=3, 
                                      start_hour=8, end_hour=20, student_data=None, 
                                      faculty_library_mapping=None, tracked_student_id=0):
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
    total_5min_intervals = (end_hour - start_hour) * 12 + 1
    
    # Run simulation for each 5-minute interval
    for step in range(total_5min_intervals):
        # Set the model's time to this step
        model.current_step = step
        
        # Calculate current hour and minutes
        current_hour = start_hour + (step // 12)
        current_minute = (step % 12) * 5
    
        # Get network data for this step
        vis_data = model.get_network_visualization_data(tracked_student_id)
        
        tracked_student = vis_data.get('tracked_student', {})
        if tracked_student:
            current_hour_str = f"{current_hour}:{current_minute:02d}"
            current_schedule = tracked_student.get('schedule', {}).get(current_hour, "Off campus")
            
            # Add attempted libraries and travel time info
            attempted_libraries_text = ""
            if tracked_student.get('status') == "traveling" and tracked_student.get('attempted_libraries'):
                # Convert library IDs to names for better readability
                attempted_library_names = []
                for lib_id in tracked_student.get('attempted_libraries', []):
                    if lib_id in model.libraries:
                        attempted_library_names.append(model.libraries[lib_id].name)
                    else:
                        attempted_library_names.append(str(lib_id))
                
                if attempted_library_names:
                    attempted_libraries_text = f"Attempted Libraries: {', '.join(attempted_library_names)}<br>"
            
            # Add travel time info if traveling
            travel_time_text = ""
            if tracked_student.get('status') == "traveling" and 'travel_time_remaining' in tracked_student:
                travel_time_text = f"Travel Time Remaining: {tracked_student['travel_time_remaining']} steps<br>"
            
            student_info_text = (
                f"<b>Student #{tracked_student_id} Info:</b><br>"
                f"Faculty: {tracked_student.get('faculty', 'Unknown')}<br>"
                f"Year: {tracked_student.get('year', 'Unknown')}<br>"
                f"Status: {tracked_student.get('status', 'Unknown')}<br>"
                f"Current Location: {tracked_student.get('current_library', 'Unknown')}<br>"
                f"Target library: {tracked_student.get('target_library', 'Unknown')}<br>"
                f"Preferred Library: {tracked_student.get('preferred_library', 'Unknown')}<br>"
                f"{travel_time_text}"
                f"{attempted_libraries_text}<br>"
                f"<b>Current Schedule ({current_hour_str}):</b> {current_schedule}<br><br>"
                f"<b>Today's Schedule:</b><br>"
            )
            
            # Add schedule timeline
            for hour in range(start_hour, end_hour + 1):
                activity = tracked_student.get('schedule', {}).get(hour, "Off campus")
                time_str = f"{hour}:00"
                
                # Highlight current hour
                if hour == current_hour:
                    student_info_text += f"<b>{time_str}: {activity}</b><br>"
                else:
                    student_info_text += f"{time_str}: {activity}<br>"
        else:
            student_info_text = f"Student #{tracked_student_id} not found"
    
        # Create a frame for this step
        frame_data = [
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
            }
        ]
        
        # Nodes (Libraries & Tracked Student)
        node_x = vis_data['node_x'][:]  # Copy existing library x-coordinates
        node_y = vis_data['node_y'][:]  # Copy existing library y-coordinates
        node_labels = vis_data['node_labels'][:]  # Copy existing labels
        node_colors = vis_data['node_colors'][:]  # Copy existing colors
        
        # Add tracked student position to the main nodes if they are on campus
        tracked_student_pos = vis_data.get('tracked_student_pos', None)
        
        if tracked_student_pos:
            node_x.append(tracked_student_pos[0])
            node_y.append(tracked_student_pos[1])
            node_colors.append("white")  # Make it stand out
            
        # Adjust markers for non-library nodes
        node_marker_size = []
        node_marker_symbol = []
        for i, color in enumerate(node_colors):
            if color == "white":  # Non-library node (tracked student or similar)
                node_marker_size.append(10)  # Smaller size for non-library node
                node_marker_symbol.append("star")  # Star symbol for non-library node
            else:  # Library node
                node_marker_size.append(25)  # Larger size for library nodes
                node_marker_symbol.append("circle")  # Default circle for library node
        
        # Create the main node trace including libraries and the tracked student
        frame_data.append({
            "type": "scatter",
            "x": node_x,
            "y": node_y,
            "mode": "markers+text",
            "text": node_labels,
            "textposition": "top center",
            "marker": {
            "size": node_marker_size,  # Set sizes according to the type of node
            "color": node_colors,
            "symbol": node_marker_symbol,  # Use different symbols
            "line": {"width": 2, "color": "black"}
        }
        })

        # Create layout with student info annotation
        frame_layout = {
            "title": (f"Library Network - Day {vis_data['day']}, {current_hour}:{current_minute:02d}<br>"
                      f"Students: {vis_data['students_in_libraries']} in Libraries, "
                      f"{vis_data['students_in_lecture']} in Lectures, "
                      f"{vis_data['students_traveling']} Traveling, "
                      f"{vis_data['students_off_campus']} Off Campus "
                      f"(Total: {vis_data['total_students']}/{student_count})"),
            "annotations": [
                {
                    "x": 1.1,
                    "y": 0.5,
                    "xref": "paper",
                    "yref": "paper",
                    "text": student_info_text,
                    "showarrow": False,
                    "align": "left",
                    "bgcolor": "rgba(255, 255, 255, 0.8)",
                    "bordercolor": "gray",
                    "borderwidth": 1,
                    "borderpad": 10,
                    "font": {"size": 12}
                }
            ],
            "margin": {"r": 300}  # Add right margin to make space for the annotation
        }
            
        # Create the frame
        frame = {
            "name": f"step_{step//update_interval}",
            "data": frame_data,
            "layout": frame_layout
        }
                
        # Add the frame
        frames.append(frame)
        
        # Run the simulation for one step (5 minutes)
        if step < total_5min_intervals - 1:  # Don't step after the last frame
            model.step()
    
    # Create slider steps for all 15-minute intervals
    slider_steps = []
    
    total_5min_intervals = ((end_hour - start_hour) * 12) + 1
    for step in range(total_5min_intervals):
        current_hour = start_hour + (step // 12)
        current_minute = (step % 12) * 5
        
        # Create appropriately formatted labels
        if current_minute == 0:
            # Full hour marks
            label = f"{current_hour}:00"
        elif current_minute % 15 == 0:
            # 15-min marks get abbreviated time
            label = f"{current_minute}"
        else:
            # 5-min marks just get a tick
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
            width=1800,
            margin={"r": 350},
            yaxis=dict(range=[vis_data['min_y'] - vis_data['buffer'], 
                      vis_data['max_y'] + vis_data['buffer']], scaleanchor="x", automargin=True),
            xaxis=dict(domain=[0, 0.9])
        ),
        frames=frames
    )
    fig.show(renderer="browser")
    return model
