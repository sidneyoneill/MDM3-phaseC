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
        self.known_library_full = []
        self.has_occupancy_knowledge = False
        self.current_physical_location = 'not_in_library'
        
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
            
            if "has_occupancy_knowledge" in student_data:
                self.has_occupancy_knowledge = student_data.get("has_occupancy_knowledge")
            else:
                # Randomly determine if this student has occupancy knowledge based on model parameter
                self.has_occupancy_knowledge = random.random() < self.model.occupancy_knowledge_proportion
        else:
            print(f"Error: No student data provided for student {unique_id}")
            
        # Initialize location and status
        self.current_library_id = 'not_in_library'
        self.target_library_id = None
        self.travel_time_remaining = 0
        self.status = "off_campus"  # Can be "in_library", "in_lecture", "traveling", or "off_campus"
        self.attempted_libraries = []  # Track libraries that were attempted but full
                    
    def get_schedule_activity(self, day, hour):
        """Get the student's scheduled activity for the current day and hour"""
        return self.schedule.get((day, hour))
    
    def _find_closest_library(self):
        """Find the best library considering preferences, distance, and occupancy"""
        # Track libraries that are known to be full or have been rejected based on occupancy probability
        if not hasattr(self, 'known_library_full'):
            self.known_library_full = []
        
        # First check if preferred library is available and not known to be full
        if self.preferred_library_id not in self.attempted_libraries and self.preferred_library_id not in self.known_library_full:
            # Check the occupancy probability
            if self._should_choose_based_on_occupancy(self.preferred_library_id):
                return self.preferred_library_id
            else:
                if self.has_occupancy_knowledge: # Only knowledgeable students track full libraries
                    # Reject based on occupancy, add to known full list
                    self.known_library_full.append(self.preferred_library_id)
                    
        # Determine the starting point for distance calculations
        # Use current_physical_location instead of current_library_id when not in a library
        if self.current_library_id != 'not_in_library':
            origin = self.current_library_id
        elif self.current_physical_location != 'not_in_library':
            origin = self.current_physical_location
        else:
            origin = self.preferred_library_id            
    
        # Get preferred libraries from faculty mapping
        preferred_libraries = self.model.faculty_library_mapping[self.faculty].get("preferred", [])
        
        # Try other preferred libraries that haven't been attempted yet or known to be too full
        available_preferred = [lib for lib in preferred_libraries 
                             if lib in self.model.libraries 
                             and lib not in self.attempted_libraries 
                             and lib not in self.known_library_full
                             and lib != self.preferred_library_id]
        
        if available_preferred and origin != 'not_in_library': 
            # Find closest preferred library considering travel time
            preferred_candidates = {}
            for library_id in available_preferred:
                try:
                    dist = nx.shortest_path_length(
                        self.model.graph, 
                        origin, 
                        library_id, 
                        weight='weight'
                    )
                    preferred_candidates[library_id] = dist
                except nx.NetworkXNoPath:
                    pass
            
            if preferred_candidates:
                # Sort by distance
                sorted_candidates = sorted(preferred_candidates.items(), key=lambda x: x[1])
                
                # Try candidates in order of increasing distance
                for library_id, _ in sorted_candidates:
                    if self._should_choose_based_on_occupancy(library_id):
                        return library_id
                    else:
                        if self.has_occupancy_knowledge: # Only knowledgeable students track full libraries
                            # Reject based on occupancy
                            self.known_library_full.append(library_id)
                            
                # For knowledgeable students: if all were rejected based on occupancy, go to next tier
                # For non-knowledgeable students: this block wouldn't happen, they'd always choose the first option
                if self.has_occupancy_knowledge:
                    if len(self.known_library_full) == len(available_preferred) + (1 if self.preferred_library_id not in self.known_library_full else 0):
                        # All preferred libraries rejected - move to acceptable ones
                        pass
                    else:
                        # Try again with the closest preferred (fallback if all have poor occupancy)
                        return sorted_candidates[0][0]
             
        elif available_preferred:  # off campus 
            # If student is off-campus, for knowledgeable students sort by expected occupancy
            # For non-knowledgeable students, just pick the first one in the list
            if not self.has_occupancy_knowledge and available_preferred:
                return random.choice(available_preferred)
        
            # Only for knowledgeable students - evaluate occupancy
            occupancy_scores = {}
            for library_id in available_preferred:
                occupancy_pct = self.model.libraries[library_id].get_occupancy_percentage() / 100.0
                # Higher score = better choice (lower occupnancy)
                occupancy_scores[library_id] = self._calculate_occupancy_probability(occupancy_pct)
            
            if occupancy_scores:
                best_library = max(occupancy_scores, key=occupancy_scores.get)
                if self._should_choose_based_on_occupancy(best_library):
                    return best_library 
                else:
                    # Only track for knowledgeable students
                    if self.has_occupancy_knowledge:
                        self.known_library_full.append(best_library)
        
        
        # If no preferred libraries available or all rejected, try acceptable libraries
        acceptable_candidates = {}
        for library_id in self.acceptable_libraries:
            if (library_id in self.model.libraries 
                and library_id not in self.attempted_libraries 
                and library_id not in self.known_library_full
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
                        acceptable_candidates[library_id] = dist
                    except nx.NetworkXNoPath:
                        # No path exists between these libraries
                        pass
                else:
                    
                    # If student is off-campus
                    if self.has_occupancy_knowledge:
                        # For knowledgeable students, add library based on occupancy
                        occupancy_pct = self.model.libraries[library_id].get_occupancy_percentage() / 100.0
                        # Invert the value so lower occupancy = lower "distance" score
                        acceptable_candidates[library_id] = 1.0 - self._calculate_occupancy_probability(occupancy_pct)
                    else:
                        # For non-knowledgeable students, just use a constant value
                        acceptable_candidates[library_id] = 1.0  # All equal priority

        # If we have acceptable options, try them in order of distance/occupancy
        if acceptable_candidates:
            sorted_candidates = sorted(acceptable_candidates.items(), key=lambda x: x[1])
        
            
            # For non-knowledgeable students, simply take the first one
            if not self.has_occupancy_knowledge and sorted_candidates:
                return random.choice(sorted_candidates)[0]
            
            # For knowledgeable students, evaluate occupancy
            for library_id, _ in sorted_candidates:
                if self._should_choose_based_on_occupancy(library_id):
                    return library_id
                else:
                    # Only track for knowledgeable students
                    if self.has_occupancy_knowledge:
                        # Reject based on occupancy
                        self.known_library_full.append(library_id)
            
            # If all were rejected or for non-knowledgeable students, use the closest one (or lowest occupancy one if from off campus)
            return sorted_candidates[0][0]
               
        # If all acceptable libraries are full/rejected, only then consider avoided libraries
        avoided_candidates = {}
        for library_id in self.model.libraries.keys():
            if (library_id not in self.attempted_libraries and 
                library_id not in self.known_library_full and
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
                        avoided_candidates[library_id] = dist
                    except nx.NetworkXNoPath:
                        pass
                else:
                    # If student is off-campus
                    if self.has_occupancy_knowledge:
                        # For knowledgeable students, add library based on occupancy
                        occupancy_pct = self.model.libraries[library_id].get_occupancy_percentage() / 100.0
                        # Invert the value so lower occupancy = lower "distance" score
                        avoided_candidates[library_id] = 1.0 - self._calculate_occupancy_probability(occupancy_pct)
                    else:
                        # For non-knowledgeable students, just use a constant value
                        avoided_candidates[library_id] = 1.0  # All equal priority
        
        # Try avoided libraries if available
        if avoided_candidates:
            sorted_candidates = sorted(avoided_candidates.items(), key=lambda x: x[1])
            
            # For non-knowledgeable students, just take the first one
            if not self.has_occupancy_knowledge and sorted_candidates:
                return random.choice(sorted_candidates)[0]
            
            # # For knowledgeable students, evaluate occupancy
            for library_id, _ in sorted_candidates:
                if self._should_choose_based_on_occupancy(library_id):
                    return library_id
                else:
                    # Only track for knowledgeable students
                    if self.has_occupancy_knowledge:
                        # Reject based on occupancy
                        self.known_library_full.append(library_id)
                    
            # If all were rejected or for non-knowledgeable students, use the closest one (or lowest occupancy one if from off campus)
            return sorted_candidates[0][0]
        
        # If all libraries have been attempted or rejected, reset lists and return to preferred
        self.attempted_libraries = []
        self.known_library_full = []
        return self.preferred_library_id
    
    def _calculate_occupancy_probability(self, occupancy):
        """
        Calculate probability of choosing a library based on its occupancy.
        Uses the function y = e^(-0.1*(x^5 / (1-x))), where x is occupancy (0-1) and y is probability (0-1).
        """
        # Apply the function y = e^(-0.2*(x^3 / (1-x)))
        try:
            exponent = -0.2 * ((occupancy**3) / (1 - occupancy))
            probability = math.exp(exponent)
            return probability
        except (OverflowError, ZeroDivisionError):
            # Handle numerical issues near occupancy=1
            return 0.0

    def _should_choose_based_on_occupancy(self, library_id):
        """
        Decide whether to choose a library based on its occupancy.
        Returns True if the library should be chosen, False otherwise.
        """
        if library_id not in self.model.libraries:
            return False
        
        # If student doesn't have occupancy knowledge, always select the library
        if not self.has_occupancy_knowledge:
            return True
        
        # Get current occupancy percentage
        occupancy_pct = self.model.libraries[library_id].get_occupancy_percentage() / 100.0
        
        # Calculate probability of choosing this library
        probability = self._calculate_occupancy_probability(occupancy_pct)
        
        # Make random decision based on probability
        return random.random() < probability
    
    def step(self):
        """Perform a step in the simulation"""  
        day, current_hour, _ = self.model.get_day_and_time()
        scheduled_activity = self.get_schedule_activity(day, current_hour)
        
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
                    self.current_physical_location = 'not_in_library'
                    self.status = "off_campus"
                    self.attempted_libraries = []  # Reset attempted libraries list
                    self.known_library_full = []
                    self.target_library_id = None
                else:
                    # Student is arriving at a library
                    # Check if they're going to a lecture or to a library
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
                            
                            # Track this rejection for metrics
                            self.model.library_rejection_counts[self.target_library_id] = self.model.library_rejection_counts.get(self.target_library_id, 0) + 1
                            self.model.library_entry_attempts[self.target_library_id] = self.model.library_entry_attempts.get(self.target_library_id, 0) + 1
                            
                            # Update physical location to where the student actually is
                            self.current_physical_location = self.target_library_id
                            
                            # Find another library to go to
                            next_library = self._find_closest_library()
                            
                            # If no valid library found, go off campus
                            if next_library is None:
                                self.current_library_id = 'not_in_library'
                                self.current_physical_location = 'not_in_library'
                                self.status = "off_campus"
                                return
                            
                            current_library = self.target_library_id
                            
                            # Set the new target and continue travelling
                            self.target_library_id = next_library
                            self.status = "traveling"
                            
                            # Set travel time based on graph
                            if self.model.graph.has_edge(current_library, next_library):
                                travel_time_minutes = self.model.graph[current_library][next_library]['weight']
                                travel_time_steps = math.ceil(travel_time_minutes / 15) 
                                self.travel_time_remaining = travel_time_steps -1 
                            else:
                                # No direct path, set default travel time
                                self.travel_time_remaining = 1  # 15 min 
                        else:
                            # Library has space, enter it
                            self.current_library_id = self.target_library_id
                            self.current_physical_location = self.target_library_id
                            self.model.libraries[self.current_library_id].add_student()
                            self.model.library_entry_attempts[self.current_library_id] = self.model.library_entry_attempts.get(self.current_library_id, 0) + 1
                            self.status = "in_library"
                            self.attempted_libraries = []  # Reset attempted libraries list
                            self.known_library_full = []
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
                        self.travel_time_remaining = 1 # 15 min
                    
            elif scheduled_activity != "library":
                # Student needs to leave library (not in their schedule)
                self.target_library_id = 'not_in_library'
                self.current_physical_location = 'not_in_library'
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
                    # Track this rejection for metrics
                    self.model.library_rejection_counts[self.target_library_id] = self.model.library_rejection_counts.get(self.target_library_id, 0) + 1
                    self.model.library_entry_attempts[self.target_library_id] = self.model.library_entry_attempts.get(self.target_library_id, 0) + 1
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
                        self.travel_time_remaining = 1  # 15 min
                else:
                    # Preferred library has space, enter it
                    self.model.libraries[self.current_library_id].add_student()
                    self.model.library_entry_attempts[self.current_library_id] = self.model.library_entry_attempts.get(self.current_library_id, 0) + 1
                    self.status = "in_library"
            elif scheduled_activity != "lecture":
                # Lecture ended, student wants to go off campus
                self.target_library_id = 'not_in_library'
                self.current_physical_location = 'not_in_library'
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
                self.travel_time_remaining = 1
            elif scheduled_activity == "library":
                # Student needs to go to a library - choose the preferred one first
                target_library = self._find_closest_library()
                self.target_library_id = target_library
                self.status = "traveling"
                # No travel time for coming from off campus
                self.travel_time_remaining = 1

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
        # Always consider fully occupied libraries as overcrowded
        if self.occupancy >= self.capacity:
            return True
        # For libraries between 90% and 100% capacity, there's a probabilistic chance
        elif self.occupancy >= self.capacity * 0.9:
            # 10% chance of allowing the student (90% chance of returning True)
            return random.random() > 0.1
        # Libraries below 90% capacity are not overcrowded
        return False
        
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
                 faculty_library_mapping=None,
                 occupancy_knowledge_proportion=1.0):    
        
        super().__init__()
        self.student_count = student_count
        self.current_step = 0
        self.hours_per_step = 0.25  # 15 minutes per step
        self.steps_per_day = int(24 / self.hours_per_step)
        self.day = 0
        
        self.occupancy_knowledge_proportion = occupancy_knowledge_proportion
        
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
            
        self.library_rejection_counts = {lib_id: 0 for lib_id in self.libraries.keys()}
        self.library_entry_attempts = {lib_id: 0 for lib_id in self.libraries.keys()}
            
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
        _, hour, _ = self.get_day_and_time()
        return hour
    
    def get_day_and_time(self):
        """Get the urrent day (1-5) and time (hour:minute)."""
        # calculate which day we're on (1-5 for Monday-Friday)
        day = (self.current_step // self.steps_per_operating_day) + 1
        
        # calculate current hour and minute
        hour = self.start_hour + int((self.current_step % self.steps_per_operating_day) * self.hours_per_step)
        minute = int(((self.current_step % self.steps_per_operating_day) * self.hours_per_step % 1) * 60)
        
        return day, hour, minute
    
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
        # check if at the start of a new day (8am) 
        is_day_start = (self.current_step % self.steps_per_operating_day == 0)
        
        # reset all students to be off campus at day start
        if is_day_start:
            self.reset_students_for_new_day()
        
        self.schedule.step() # Call RandomActivation, which randomly iterates through all the student agents and calls student.step() for each one
        self.current_step += 1
    
        # Collect data on every step (15 minutes)
        self.datacollector.collect(self)
        
        # Still track day changes
        if self.current_step % self.steps_per_day == 0:
            self.day += 1
            
    def reset_students_for_new_day(self):
        """Reset all students to be off campus at the start of a new day """
        for student in self.schedule.agents:
            #only reset if not already off campus
            if student.status != "off_campus":
                # if they're in a library, make sure to decrement the occupancy
                if student.status == "in_library" and student.current_library_id in self.libraries:
                    self.libraries[student.current_library_id].remove_student()
                
                # Reset student location and status
                student.current_library_id = 'not_in_library'
                student.current_physical_location = 'not_in_library'
                student.target_library_id = None
                student.status = "off_campus"
                student.travel_time_remaining = 0
                student.attempted_libraries = []
                student.known_library_full = []
             
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
            
            # Only add lecture count for Queens library (node ID 6)
            if node == 6:
                node_label = f"{library.name}<br>Library: {library.occupancy}/{library.capacity}<br>Lecture: {lecture_count}"
            else:
                node_label = f"{library.name}<br>Library: {library.occupancy}/{library.capacity}"
            
            node_labels.append(node_label)
     
            occupancy_pct = library.get_occupancy_percentage()
            
            # Color node based on occupancy percentage
            if occupancy_pct < 50:
                color = 'green'  # Low occupancy
            elif occupancy_pct < 90:
                color = 'orange'  # Medium occupancy
            else:
                color = 'red'  # High occupancy
                
            node_colors.append(color)
        
        day, current_hour, current_minute = self.get_day_and_time()
            
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
            'day': day,
            'hour': current_hour,
            'minute': current_minute,
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

def run_library_simulation_with_frames(days=5, student_count=10, update_interval=1, 
                                      start_hour=8, end_hour=20, student_data=None, faculty_library_mapping=None, occupancy_knowledge_proportion=1.0):
    """
    Run the library simulation for a specified number of days (default 5 days = one week)
    Update interval of 1 means create a frame for every step (15 minutes)
    """ 
    # Create the model with specified operating hours and student data
    model = LibraryNetworkModel(
        student_count=student_count,
        start_hour=start_hour,
        end_hour=end_hour,
        student_data=student_data,
        faculty_library_mapping=faculty_library_mapping,
        occupancy_knowledge_proportion=occupancy_knowledge_proportion
    )
    
    # Create a frame for each 15-minute interval
    frames = []
    
    intervals_per_day = (end_hour - start_hour) * 4
    total_intervals = intervals_per_day * days
    
    day_names = ["Mon", "Tues", "Wed", "Thur", "Fri"]
    
    # Run simulation for each 15-minute interval
    for step in range(total_intervals):
        # Calculate current day, hour and minute
        current_day = (step // intervals_per_day) % 5  # 0-4 for Monday-Friday
        current_hour = start_hour + ((step % intervals_per_day) // 4)
        current_minute = (step % 4) * 15
        
        model.current_step = step
        model.step()
        
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
                "title": (f"Library Network - {day_names[current_day]}, {current_hour}:{current_minute:02d}<br>"
                          f"Students: {vis_data['students_in_libraries']} in Libraries, "
                          f"{vis_data['students_in_lecture']} in Lectures, "
                          f"{vis_data['students_traveling']} Traveling, "
                          f"{vis_data['students_off_campus']} Off Campus "
                          f"(Total: {vis_data['total_students']}/{student_count})")
            }
        }
        
        # Add the frame
        frames.append(frame)
        
    # Create slider steps with clearer day/time labels
    slider_steps = []
    
    for step in range(total_intervals):
        # Calculate current day, hour and minute
        current_day = step // intervals_per_day
        current_hour = start_hour + ((step % intervals_per_day) // 4)
        current_minute = (step % 4) * 15
        
        # Format the slider labels
        if current_minute == 0:
            # For full hours, show the day and hour
            if current_hour == start_hour:
                # For the start of each day, show the day name
                label = f"{day_names[current_day]} {current_hour}:00"
            else:
                # For other hours, just show the hour
                label = f"{current_hour}:00"
        else:
            # For 15-min intervals, use a small mark instead of a label
            label = ""
        
        # Only add visible markers at hours or significant times
        if current_minute == 0:
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
            title=f"Weekly Library Network Simulation (Mon-Fri, {start_hour}:00-{end_hour}:00)",
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

    # fig.show(renderer="browser")
    return model

