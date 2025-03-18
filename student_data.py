#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo student schedule generator for library simulation
"""

import random
import numpy as np

# Faculty to library mapping - each faculty has one preferred library
faculty_library_mapping = {
    "Engineering": 6,
    "Arts": [1,7],
    "Science": [2,5],
    "Medical": 4,
    "Social_science_law": [1, 7]
}

# Statistical profiles for each faculty
faculty_characteristics = {
    "Engineering": {
        "lectures_per_day": {"mean": 3.5, "std_dev": 0.8},  # Normal distribution
        "lecture_duration_hours": {"mean": 1.5, "std_dev": 0.5},
        "library_visit_probability": 0.75,  # Probability of visiting library when free
        "library_duration_hours": {"mean": 2.5, "std_dev": 1.0},  # How long they stay
        "library_transition_probability": 0.8,  # Probability to stay in library next hour
        "chronotype_distribution": {"mean": 0.55, "std_dev": 0.2}  # Engineers tend to be morning people
    },
    "Arts": {
        "lectures_per_day": {"mean": 2.5, "std_dev": 0.7},
        "lecture_duration_hours": {"mean": 1.0, "std_dev": 0.3},
        "library_visit_probability": 0.65,
        "library_duration_hours": {"mean": 2.0, "std_dev": 0.8},
        "library_transition_probability": 0.7,
        "chronotype_distribution": {"mean": 0.4, "std_dev": 0.25}  # Arts students tend to be night owls
    },
    "Science": {
        "lectures_per_day": {"mean": 3.0, "std_dev": 0.6},
        "lecture_duration_hours": {"mean": 1.2, "std_dev": 0.4},
        "library_visit_probability": 0.7,
        "library_duration_hours": {"mean": 2.8, "std_dev": 1.0},
        "library_transition_probability": 0.75,
        "chronotype_distribution": {"mean": 0.5, "std_dev": 0.2}  # Evenly distributed
    },
    "Medical": { # combines, medicine, vet science and dentistry
        "lectures_per_day": {"mean": 6.0, "std_dev": 0.9},  # More structured programs
        "lecture_duration_hours": {"mean": 1.5, "std_dev": 0.5},
        "library_visit_probability": 0.8,
        "library_duration_hours": {"mean": 3.0, "std_dev": 1.2},
        "library_transition_probability": 0.85,  # More likely to have long study sessions
        "chronotype_distribution": {"mean": 0.7, "std_dev": 0.15}  # Medicine students tend to be morning people
    },
    "Social_science_law": {
        "lectures_per_day": {"mean": 3.0, "std_dev": 0.7},
        "lecture_duration_hours": {"mean": 1.3, "std_dev": 0.4},
        "library_visit_probability": 0.6,
        "library_duration_hours": {"mean": 2.0, "std_dev": 0.9},
        "library_transition_probability": 0.7,
        "chronotype_distribution": {"mean": 0.5, "std_dev": 0.2}  # Evenly distributed
    }
}

def adjust_by_year(characteristics, year):
    """Adjust faculty characteristics based on student's year of study"""
    adjusted = characteristics.copy()
    
    if year == 1:
        # First years have more structured time, more lectures, less library use
        adjusted["lectures_per_day"]["mean"] += 0.8
        adjusted["library_visit_probability"] -= 0.1
        adjusted["library_duration_hours"]["mean"] -= 0.5
    elif year == 2:
        # Second years - small adjustment
        adjusted["lectures_per_day"]["mean"] += 0.3
    elif year == 3:
        # Third years - more independent study
        adjusted["lectures_per_day"]["mean"] -= 0.3
        adjusted["library_visit_probability"] += 0.05
        adjusted["library_duration_hours"]["mean"] += 0.3
    elif year == 4:
        # Fourth years - much more independent study, fewer lectures
        adjusted["lectures_per_day"]["mean"] -= 0.8
        adjusted["library_visit_probability"] += 0.15
        adjusted["library_duration_hours"]["mean"] += 0.7
        
    return adjusted

def generate_lectures(characteristics, start_hour=9, end_hour=18): # lectures can start at 9am and finish at 6pm at the latest
    """Generate lecture schedule based on faculty characteristics"""
    # Determine number of lectures
    num_lectures = max(1, int(round(np.random.normal(
        characteristics["lectures_per_day"]["mean"],
        characteristics["lectures_per_day"]["std_dev"]))))
    
    # All possible lecture hours
    all_hours = list(range(start_hour, end_hour))
    
    # Randomly select lecture start times
    if num_lectures > 0 and all_hours:
        lecture_starts = random.sample(all_hours, min(len(all_hours), num_lectures))
        lecture_starts.sort()  # Sort chronologically
    else:
        lecture_starts = []
    
    # Generate lecture durations
    lecture_schedule = {}
    for start_hour in lecture_starts:
        # Generate duration for this lecture
        duration = max(1, int(round(np.random.normal(
            characteristics["lecture_duration_hours"]["mean"],
            characteristics["lecture_duration_hours"]["std_dev"]
        ))))
        
        # Add lecture hours to schedule
        for hour in range(start_hour, min(start_hour + duration, end_hour)):
            lecture_schedule[hour] = "lecture"
    
    return lecture_schedule

def generate_library_visits(characteristics, lecture_schedule, chronotype, start_hour=8, end_hour=20):
    """Generate library visits during non-lecture hours, influenced by chronotype"""
    schedule = lecture_schedule.copy()
    
    # Find blocks of free time
    free_blocks = []
    current_block = []
    
    for hour in range(start_hour, end_hour):
        if hour not in schedule:
            current_block.append(hour)
        else:
            if current_block:
                free_blocks.append(current_block)
                current_block = []
    
    # Add the last block if it exists
    if current_block:
        free_blocks.append(current_block)
    
    # For each free block, decide if student visits library
    for block in free_blocks:
        
        # Base probability of visiting library
        base_probability = characteristics["library_visit_probability"]
        
        # Adjust probability based on chronotype and time of day
        block_start_hour = block[0]
        
        # Morning hours (8-12)
        if block_start_hour < 12:
            # Morning people more likely to use library in morning
            time_factor = chronotype * 0.3  # +30% for perfect morning people
        # Afternoon hours (12-17)
        elif block_start_hour < 17:
            time_factor = 0  # Neutral for everyone
        # Evening hours (17-20)
        else:
            # Night owls more likely to use library in evening
            time_factor = (1 - chronotype) * 0.3  # +30% for perfect night owls
        
        # Adjust probability
        adjusted_probability = min(1.0, base_probability + time_factor)
            
        # Decide if student visits library during this block
        if random.random() < adjusted_probability:
            # Determine duration of library visit
            target_duration = max(1, int(round(np.random.normal(
                characteristics["library_duration_hours"]["mean"],
                characteristics["library_duration_hours"]["std_dev"]
            ))))
            
            # Limit duration to block length
            actual_duration = min(target_duration, len(block))
            
            # Randomly choose start time within block
            if len(block) > actual_duration:
                start_index = random.randint(0, len(block) - actual_duration)
            else:
                start_index = 0
            
            # Add library hours to schedule
            for i in range(start_index, start_index + actual_duration):
                schedule[block[i]] = "library"
    
    return schedule

def generate_student_schedule(faculty, year, chronotype, start_hour=8, end_hour=20):
    """Generate a complete daily schedule for a student based on faculty, year and chronotype"""
    # Get base characteristics for this faculty
    base_characteristics = faculty_characteristics[faculty]
    
    # Adjust characteristics based on year
    adjusted_characteristics = adjust_by_year(base_characteristics, year)
    
    # Generate lecture schedule
    schedule = generate_lectures(adjusted_characteristics, 9, 18)
    
    # Add library visits
    schedule = generate_library_visits(adjusted_characteristics, schedule, chronotype, start_hour, end_hour)
    
    return schedule

def assign_preferred_library(faculty):
    """Assigns a preferred library based on the faculty's mapping."""
    preferred_libraries = faculty_library_mapping[faculty]
    
    # If there's only one preferred library, assign it directly
    if isinstance(preferred_libraries, int):
        return preferred_libraries
    
    # If there are multiple preferred libraries, randomly pick one
    return random.choice(preferred_libraries)

def generate_student_population(num_students, faculty_distribution=None, start_hour=8, end_hour=20):
    """Generate data for the specified number of students"""
    if faculty_distribution is None:
        # Default faculty distribution
        faculty_distribution = {
            "Engineering": 0.134,
            "Arts": 0.1843,
            "Science": 0.1934, 
            "Medical": 0.226,
            "Social_science_law": 0.2623
        }
    
    faculties = list(faculty_distribution.keys())
    probabilities = list(faculty_distribution.values())
    
    student_data = {}
    for i in range(num_students):
        # 1. Randomly assign faculty based on distribution
        faculty = np.random.choice(faculties, p=probabilities)
        
        # 2. Randomly assign year (1-4)
        year = random.randint(1, 4)
        
        # 3. Set preferred library based on faculty
        preferred_library_id = assign_preferred_library(faculty)
        
        # 4. Generate chronotype (0 = night owl, 1 = morning person)
        chronotype_params = faculty_characteristics[faculty]["chronotype_distribution"]
        chronotype = np.random.normal(chronotype_params["mean"], chronotype_params["std_dev"])
        chronotype = max(0, min(1, chronotype))  # Clamp between 0 and 1
        
        # 5. Generate schedule
        schedule = generate_student_schedule(faculty, year, chronotype, start_hour, end_hour)
        
        student_data[i] = {
            "faculty": faculty,
            "year": year,
            "preferred_library_id": preferred_library_id,
            "chronotype": chronotype,  # Store chronotype for reference
            "schedule": schedule
        }
    
    return student_data
"""
# Example usage - only do when run this script directly (not imported in from other modules)
if __name__ == "__main__":
    # Generate schedules for 20 students
    students = generate_student_population(20)
    
    # Print first student's schedule as an example
    student = students[0]
    print(f"Student 0: Faculty={student['faculty']}, Year={student['year']}, Chronotype={student['chronotype']:.2f}")
    print(f"Preferred Library: {student['preferred_library_id']}")
    print("Schedule:")
    for hour in range(8, 20):
        activity = student['schedule'].get(hour, None)
        print(f"{hour}:00 - {activity if activity else 'off campus'}")
"""