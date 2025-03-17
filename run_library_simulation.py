#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:20:56 2025

@author: lewisvaughan
"""

# Import the simulation module
from library_simulation import run_library_simulation_with_frames
from student_data import generate_student_population

# Define operating hours
START_HOUR = 8    # 8am
END_HOUR = 20     # 8pm
HOURS_TO_SIMULATE = END_HOUR - START_HOUR
STEPS_PER_HOUR = 4  # 15-minute steps
STUDENT_COUNT = 1000  # Now simulating 100 students

# Generate Monte Carlo student data
student_data = generate_student_population(
    STUDENT_COUNT,
    start_hour=START_HOUR,
    end_hour=END_HOUR
)

# Use animation with playback controls and student data
model = run_library_simulation_with_frames(
    steps=HOURS_TO_SIMULATE * STEPS_PER_HOUR,  # Steps for 12 hours (8am-8pm)
    student_count=STUDENT_COUNT,
    update_interval=1,  # Create a frame every 15 minutes (every step)
    start_hour=START_HOUR,
    end_hour=END_HOUR,
    student_data=student_data
)

# After simulation completes, analyze the data
results = model.datacollector.get_model_vars_dataframe()
print(results.tail())

# Save the results to CSV for further analysis
# results.to_csv("library_simulation_results.csv")

