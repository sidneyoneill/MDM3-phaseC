# Define student data with faculty, year, preferred library, and schedule
# Schedule can have values: "library", "lecture", or None (for off campus)
student_data = {
    0: {
        "faculty": "Engineering",
        "year": 2,
        "preferred_library_id": 1,  # Engineering Library
        "schedule": {
            9: "lecture",    # 9am
            10: "lecture",   # 10am
            11: "library",   # 11am
            12: None,        # Lunch break
            13: None,        # Lunch break  
            14: "library",   # 2pm
            15: "library",   # 3pm
            16: "library",   # 4pm
        }
    },
    1: {
        "faculty": "Arts",
        "year": 3,
        "preferred_library_id": 2,  # Arts Library
        "schedule": {
            10: "lecture",   # 10am
            11: "lecture",   # 11am
            12: "library",   # 12pm
            13: None,        # Break
            14: None,        # Break
            15: "library",   # 3pm
            16: "library",   # 4pm
            17: "library",   # 5pm
        }
    },
    2: {
        "faculty": "Science",
        "year": 1,
        "preferred_library_id": 3,  # Science Library
        "schedule": {
            8: "library",    # 8am (early bird)
            9: "library",    # 9am
            10: "library",   # 10am
            11: None,        # Break
            12: None,        # Lunch
            13: "lecture",   # 1pm
            14: "lecture",   # 2pm
        }
    },
    3: {
        "faculty": "Medicine",
        "year": 4,
        "preferred_library_id": 4,  # Medical Library
        "schedule": {
            11: "lecture",   # 11am
            12: "lecture",   # 12pm
            13: "lecture",   # 1pm
            14: "library",   # 2pm
            15: "library",   # 3pm
            16: None,        # Break
            17: "library",   # 5pm
            18: "library",   # 6pm
            19: "library",   # 7pm
        }
    },
    4: {
        "faculty": "Business",
        "year": 2,
        "preferred_library_id": 5,  # Business Library
        "schedule": {
            9: "lecture",    # 9am
            10: "lecture",   # 10am
            11: None,        # Break
            12: None,        # Lunch
            13: "library",   # 1pm
            14: "library",   # 2pm
            15: "library",   # 3pm
            16: "library",   # 4pm
        }
    },
    5: {
        "faculty": "Engineering",
        "year": 3,
        "preferred_library_id": 1,  # Engineering Library
        "schedule": {
            13: "lecture",   # 1pm
            14: "lecture",   # 2pm
            15: "library",   # 3pm
            16: "library",   # 4pm
            17: "library",   # 5pm
            18: "library",   # 6pm
        }
    },
    6: {
        "faculty": "Arts",
        "year": 1,
        "preferred_library_id": 2,  # Arts Library
        "schedule": {
            8: "library",    # 8am
            9: "library",    # 9am
            10: "library",   # 10am
            11: None,        # Break
            12: None,        # Lunch
            13: None,        # Break
            14: "lecture",   # 2pm
            15: "lecture",   # 3pm
        }
    },
    7: {
        "faculty": "Science",
        "year": 4,
        "preferred_library_id": 3,  # Science Library
        "schedule": {
            10: "lecture",   # 10am
            11: "lecture",   # 11am
            12: "library",   # 12pm
            13: None,        # Lunch
            14: "library",   # 2pm
            15: "library",   # 3pm
            16: "library",   # 4pm
            17: "library",   # 5pm
        }
    },
    8: {
        "faculty": "Medicine",
        "year": 2,
        "preferred_library_id": 4,  # Medical Library
        "schedule": {
            8: "lecture",    # 8am
            9: "lecture",    # 9am
            10: "lecture",   # 10am
            11: "library",   # 11am
            12: None,        # Lunch
            13: None,        # Break
            14: None,        # Break
            15: "library",   # 3pm
            16: "library",   # 4pm
        }
    },
    9: {
        "faculty": "Business",
        "year": 3,
        "preferred_library_id": 5,  # Business Library
        "schedule": {
            11: "library",   # 11am
            12: "library",   # 12pm
            13: "lecture",   # 1pm
            14: "lecture",   # 2pm
            15: None,        # Break
            16: "library",   # 4pm
            17: "library",   # 5pm
            18: "library",   # 6pm
        }
    }
}