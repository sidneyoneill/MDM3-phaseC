#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 12:04:14 2025

@author: lewisvaughan
"""
import pandas as pd

from weekly_library_simulation_knowledge import run_library_simulation_with_frames
from weekly_student_data import generate_student_population, faculty_library_mapping
from weekly_library_metrics import calculate_library_metrics, print_aggregated_metrics, aggregate_metrics

# Define operating hours
START_HOUR = 8    # 8am
END_HOUR = 20     # 8pm
HOURS_TO_SIMULATE = END_HOUR - START_HOUR
STEPS_PER_HOUR = 4  # 15-minute steps
STUDENT_COUNT = 4000  # Now simulating 100 students
NUM_DAYS = 5
NUM_SIMULATIONS = 5

def run_multiple_simulations(num_simulations=5, days=5, student_count=4000, 
                            update_interval=1, start_hour=8, end_hour=20,
                            faculty_library_mapping=None, occupancy_knowledge_proportion=1.0):
    """
    Run multiple library simulations and aggregate the results.
    """
    # Initialize containers for aggregated metrics
    all_metrics = []
    all_results = pd.DataFrame()
    
    for i in range(num_simulations):
        print(f"\nRunning simulation {i+1} of {num_simulations}...")
        
        # Generate new student data for each run
        student_data = generate_student_population(
            student_count,
            start_hour=start_hour,
            end_hour=end_hour
        )
        
        # Run the simulation
        model = run_library_simulation_with_frames(
            days=days,
            student_count=student_count,
            update_interval=update_interval,
            start_hour=start_hour,
            end_hour=end_hour,
            student_data=student_data,
            faculty_library_mapping=faculty_library_mapping,
            occupancy_knowledge_proportion=occupancy_knowledge_proportion
        )
        
        # After simulation completes, analyze the data
        results = model.datacollector.get_model_vars_dataframe()
        
        # Add a column to identify which simulation run this is
        results['simulation_run'] = i + 1
        
        # Append to the combined results DataFrame
        all_results = pd.concat([all_results, results], ignore_index=False)
        
        # Get metrics for this run
        metrics = calculate_library_metrics(model, simulation_days=days)
        all_metrics.append(metrics)
    
    # Save the combined results to Excel
    all_results.to_csv("all_simulation_runs_knowledge.csv")
    
    # Aggregate the metrics
    return aggregate_metrics(all_metrics)

# Run multiple simulations and get aggregated metrics
aggregated_metrics = run_multiple_simulations(
    num_simulations=NUM_SIMULATIONS,
    days=NUM_DAYS,
    student_count=STUDENT_COUNT,
    update_interval=1,
    start_hour=START_HOUR,
    end_hour=END_HOUR,
    faculty_library_mapping=faculty_library_mapping
)

# Print the aggregated metrics
print_aggregated_metrics(aggregated_metrics, NUM_SIMULATIONS)

