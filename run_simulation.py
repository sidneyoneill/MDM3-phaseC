#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:20:56 2025

@author: lewisvaughan
"""

# Import the simulation module
from library_simulation import run_library_simulation_with_frames

# Use animation with playback controls
model = run_library_simulation_with_frames(
    steps=96,          # Simulate for 24 hours - a step per 15 minutes
    student_count=500,
    update_interval=4  # Create a frame every hour
)

# After simulation completes, analyse the data
results = model.datacollector.get_model_vars_dataframe()
print(results.tail())

# Save the results to CSV for further analysis
# results.to_csv("library_simulation_results.csv")