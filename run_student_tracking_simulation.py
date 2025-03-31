#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:20:56 2025

@author: lewisvaughan
"""

from student_tracking_simulation import run_library_simulation_with_frames
from daily_student_data import generate_student_population, faculty_library_mapping

# Define operating hours
START_HOUR = 8    # 8am
END_HOUR = 20     # 8pm
HOURS_TO_SIMULATE = END_HOUR - START_HOUR
STEPS_PER_HOUR = 12  
STUDENT_COUNT = 4000 
TRACKED_STUDENT_ID = 0 # comment out if want no tracking

# Generate Monte Carlo student data
student_data = generate_student_population(
    STUDENT_COUNT,
    start_hour=START_HOUR,
    end_hour=END_HOUR
)

model = run_library_simulation_with_frames(
    steps=HOURS_TO_SIMULATE * STEPS_PER_HOUR,  # Steps for 12 hours (8am-8pm)
    student_count=STUDENT_COUNT,
    update_interval=1,  # 1 so show simulation every 5 minutes
    start_hour=START_HOUR,
    end_hour=END_HOUR,
    student_data=student_data,
    faculty_library_mapping=faculty_library_mapping,
    tracked_student_id = TRACKED_STUDENT_ID
)

# After simulation completes, analyze the data
results = model.datacollector.get_model_vars_dataframe()
print(results.tail())

# Save the results to CSV for further analysis
# results.to_csv("library_simulation_results.csv")

