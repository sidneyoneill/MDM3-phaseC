#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:20:56 2025

@author: lewisvaughan
"""

from weekly_library_simulation import run_library_simulation_with_frames
from weekly_student_data import generate_student_population, faculty_library_mapping
from weekly_library_metrics import analyze_library_simulation

# Define operating hours
START_HOUR = 8    # 8am
END_HOUR = 20     # 8pm
HOURS_TO_SIMULATE = END_HOUR - START_HOUR
STEPS_PER_HOUR = 4  # 15-minute steps
STUDENT_COUNT = 4000  # Now simulating 100 students
NUM_DAYS = 5

# Generate Monte Carlo student data
student_data = generate_student_population(
    STUDENT_COUNT,
    start_hour=START_HOUR,
    end_hour=END_HOUR
)

# RUN FOR WEEKLY SIMULATION WITHOUT KNOWLEDGE
model = run_library_simulation_with_frames(
    days=5,  # 5 days, 8am-8pm each day
    student_count=STUDENT_COUNT,
    update_interval=1,  # Create a frame every 15 minutes (every step)
    start_hour=START_HOUR,
    end_hour=END_HOUR,
    student_data=student_data,
    faculty_library_mapping=faculty_library_mapping
)

analyze_library_simulation(model, days=5)

# After simulation completes, analyze the data
results = model.datacollector.get_model_vars_dataframe()
print(results.tail())

# Save the results to CSV for further analysis
# results.to_csv("weekly_library_simulation_results.csv")

