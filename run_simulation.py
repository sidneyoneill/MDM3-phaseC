#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:20:56 2025

@author: lewisvaughan
"""

# Import the simulation module
from library_simulation import run_library_simulation_with_frames

# Import the student data
from student_data import student_data

# Define operating hours
START_HOUR = 8    # 8am
END_HOUR = 20     # 8pm
HOURS_TO_SIMULATE = END_HOUR - START_HOUR
STEPS_PER_HOUR = 4  # 15-minute steps

# Use animation with playback controls and student data
model = run_library_simulation_with_frames(
    steps=HOURS_TO_SIMULATE * STEPS_PER_HOUR,  # Steps for 12 hours (8am-8pm)
    student_count=10,
    update_interval=4,  # Create a frame every hour
    start_hour=START_HOUR,
    end_hour=END_HOUR,
    student_data=student_data
)

# After simulation completes, analyze the data
results = model.datacollector.get_model_vars_dataframe()
print(results.tail())

"""
# Print verification of student counts
students_in_libraries = model.get_students_in_library()
students_traveling = model.get_traveling_students()
students_off_campus = model.get_students_off_campus()
total = students_in_libraries + students_traveling + students_off_campus

print("\nFinal Status Check:")
print(f"Students in libraries: {students_in_libraries}")
print(f"Students traveling: {students_traveling}")
print(f"Students off-campus: {students_off_campus}")
print(f"Total students: {total} (Expected: {model.student_count})")
"""

# Save the results to CSV for further analysis
# results.to_csv("library_simulation_results.csv")