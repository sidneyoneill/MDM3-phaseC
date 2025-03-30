#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 14:09:00 2025

@author: charliedavies
"""

def calculate_library_metrics(model, simulation_days=5):
    """
    Calculate comprehensive metrics for library usage based on a completed simulation.
    
    Parameters:
    model (LibraryNetworkModel): The completed library simulation model
    simulation_days (int): Number of days the simulation was run for
    
    Returns:
    dict: Dictionary containing all calculated metrics
    """
    # Step 1: Extract all library data
    libraries = model.libraries
    library_names = [lib.name for lib in libraries.values()]
    library_capacities = [lib.capacity for lib in libraries.values()]
    
    # Get occupancy data from the model's datacollector
    occupancy_data = model.datacollector.get_model_vars_dataframe()["Library_Occupancy"]
    
    # Calculate hours per day and total simulation hours
    hours_per_day = model.end_hour - model.start_hour
    total_hours = hours_per_day * simulation_days
    steps_per_hour = int(1 / model.hours_per_step)  # Number of 15-min steps per hour
    total_steps = total_hours * steps_per_hour
    
    # Extract all occupancy values for each library
    library_occupancies = {}
    for i, row in occupancy_data.items():
        for lib_name, occupancy in row.items():
            if lib_name not in library_occupancies:
                library_occupancies[lib_name] = []
            library_occupancies[lib_name].append(occupancy)
    
    # 1. Calculate average occupancy percentage for each library
    avg_occupancy_pct = {}
    library_id_map = {lib.name: lib_id for lib_id, lib in libraries.items()}
    
    for lib_name, occupancies in library_occupancies.items():
        lib_id = library_id_map.get(lib_name)
        if lib_id is not None:
            capacity = libraries[lib_id].capacity
            avg_occ = sum(occupancies) / len(occupancies)
            avg_occupancy_pct[lib_name] = (avg_occ / capacity) * 100 if capacity > 0 else 0
    
    # 2. Mean occupancy across all libraries
    mean_occupancy = sum(avg_occupancy_pct.values()) / len(avg_occupancy_pct) if avg_occupancy_pct else 0
    
    # 3. Deviation from mean for each library
    deviation_from_mean = {lib_name: occ_pct - mean_occupancy for lib_name, occ_pct in avg_occupancy_pct.items()}
    
    # 4. Standard deviation across libraries
    import math
    squared_deviations = sum(dev**2 for dev in deviation_from_mean.values())
    std_deviation = math.sqrt(squared_deviations / len(deviation_from_mean)) if deviation_from_mean else 0
    
  # 5 & 6. Count failed entries and calculate rejection probabilities
    # We need to modify this part to use model-wide tracking instead of agent.attempted_libraries
    
    # If the model doesn't have these attributes, we need to add them
    if not hasattr(model, 'library_rejection_counts'):
        # This means metrics were calculated without tracking rejections during simulation
        print("Warning: No rejection tracking found in model. Metrics 5-6 will not be accurate.")
        failed_entries = {lib_name: 0 for lib_name in library_names}
        entry_attempts = {lib_name: 0 for lib_name in library_names}
    else:
        # Use the tracked data from the model
        failed_entries = model.library_rejection_counts
        entry_attempts = model.library_entry_attempts
    
    # 6. Calculate rejection probabilities
    rejection_probability = {}
    for lib_name in library_names:
        lib_id = library_id_map.get(lib_name)
        if lib_id is not None:
            attempts = entry_attempts.get(lib_id, 0)
            rejections = failed_entries.get(lib_id, 0)
            rejection_probability[lib_name] = (rejections / attempts * 100) if attempts > 0 else 0
    
    # 5. Failed library entries percentage (total)
    total_attempts = sum(entry_attempts.values())
    total_rejections = sum(failed_entries.values())
    failed_entries_percentage = (total_rejections / total_attempts * 100) if total_attempts > 0 else 0
    
    # 7. Congestion score - hours exceeding 80% capacity
    congestion_scores = {}
    for lib_name, occupancies in library_occupancies.items():
        lib_id = library_id_map.get(lib_name)
        if lib_id is not None:
            capacity = libraries[lib_id].capacity
            # Count steps where occupancy exceeds 80% of capacity
            threshold = capacity * 0.8
            congested_steps = sum(1 for occ in occupancies if occ >= threshold)
            # Convert steps to hours (considering each step is 15 minutes)
            congested_hours = congested_steps / steps_per_hour
            congestion_scores[lib_name] = congested_hours
    
    # Compile all metrics into a dictionary
    metrics = {
        "average_occupancy_percentage": avg_occupancy_pct,
        "mean_occupancy_all_libraries": mean_occupancy,
        "deviation_from_mean": deviation_from_mean,
        "standard_deviation": std_deviation,
        "failed_entries_percentage": failed_entries_percentage,
        "rejection_probability": rejection_probability,
        "congestion_score_hours": congestion_scores
    }
    
    return metrics


def print_library_metrics(metrics):
    """
    Print formatted metrics for better readability.
    
    Parameters:
    metrics (dict): Dictionary of calculated metrics from calculate_library_metrics()
    """
    print("\n===== LIBRARY SIMULATION METRICS =====\n")
    
    # 1. Print average occupancy percentages
    print("1. AVERAGE OCCUPANCY (% of capacity)")
    print("-" * 40)
    for lib_name, percentage in metrics["average_occupancy_percentage"].items():
        print(f"  {lib_name}: {percentage:.2f}%")
    print()
    
    # 2. Print mean occupancy
    print(f"2. MEAN OCCUPANCY ACROSS ALL LIBRARIES: {metrics['mean_occupancy_all_libraries']:.2f}%")
    print()
    
    # 3. Print deviation from mean
    print("3. DEVIATION FROM MEAN OCCUPANCY")
    print("-" * 40)
    for lib_name, deviation in metrics["deviation_from_mean"].items():
        print(f"  {lib_name}: {deviation:+.2f}%")
    print()
    
    # 4. Print standard deviation
    print(f"4. STANDARD DEVIATION OF OCCUPANCY: {metrics['standard_deviation']:.2f}%")
    print()
    
    # 5. Print failed entries percentage
    print(f"5. FAILED LIBRARY ENTRIES: {metrics['failed_entries_percentage']:.2f}% of all attempts")
    print()
    
    # 6. Print rejection probabilities
    print("6. REJECTION PROBABILITY BY LIBRARY")
    print("-" * 40)
    for lib_name, probability in metrics["rejection_probability"].items():
        print(f"  {lib_name}: {probability:.2f}%")
    print()
    
    # 7. Print congestion scores
    print("7. CONGESTION SCORE (hours above 80% capacity)")
    print("-" * 40)
    for lib_name, hours in metrics["congestion_score_hours"].items():
        print(f"  {lib_name}: {hours:.1f} hours")
    print()


# Example usage:
def analyze_library_simulation(model, days=5):
    """
    Run metrics analysis on a completed library simulation.
    
    Parameters:
    model (LibraryNetworkModel): The completed simulation model
    days (int): Number of days the simulation was run for
    """
    metrics = calculate_library_metrics(model, simulation_days=days)
    print_library_metrics(metrics)
    return metrics