# Define student data with faculty, year, preferred library, and study hours
student_data = {
    0: {
        "faculty": "Engineering",
        "year": 2,
        "preferred_library_id": 1,  # Engineering Library
        "schedule": {
            9: True,   # 9am
            10: True,  # 10am
            11: True,  # 11am
            12: False, # Lunch break
            13: False, # Lunch break
            14: True,  # 2pm
            15: True,  # 3pm
            16: True,  # 4pm
        }
    },
    1: {
        "faculty": "Arts",
        "year": 3,
        "preferred_library_id": 2,  # Arts Library
        "schedule": {
            10: True,  # 10am
            11: True,  # 11am
            12: True,  # 12pm
            13: False, # Break
            14: False, # Break
            15: True,  # 3pm
            16: True,  # 4pm
            17: True,  # 5pm
        }
    },
    2: {
        "faculty": "Science",
        "year": 1,
        "preferred_library_id": 3,  # Science Library
        "schedule": {
            8: True,   # 8am (early bird)
            9: True,   # 9am
            10: True,  # 10am
            11: False, # Break
            12: False, # Lunch
            13: True,  # 1pm
            14: True,  # 2pm
        }
    },
    3: {
        "faculty": "Medicine",
        "year": 4,
        "preferred_library_id": 4,  # Medical Library
        "schedule": {
            11: True,  # 11am
            12: True,  # 12pm
            13: True,  # 1pm
            14: True,  # 2pm
            15: True,  # 3pm
            16: False, # Break
            17: True,  # 5pm
            18: True,  # 6pm
            19: True,  # 7pm
        }
    },
    4: {
        "faculty": "Business",
        "year": 2,
        "preferred_library_id": 5,  # Business Library
        "schedule": {
            9: True,   # 9am
            10: True,  # 10am
            11: False, # Break
            12: False, # Lunch
            13: True,  # 1pm
            14: True,  # 2pm
            15: True,  # 3pm
            16: True,  # 4pm
        }
    },
    5: {
        "faculty": "Engineering",
        "year": 3,
        "preferred_library_id": 1,  # Engineering Library
        "schedule": {
            13: True,  # 1pm
            14: True,  # 2pm
            15: True,  # 3pm
            16: True,  # 4pm
            17: True,  # 5pm
            18: True,  # 6pm
        }
    },
    6: {
        "faculty": "Arts",
        "year": 1,
        "preferred_library_id": 2,  # Arts Library
        "schedule": {
            8: True,   # 8am
            9: True,   # 9am
            10: True,  # 10am
            11: False, # Break
            12: False, # Lunch
            13: False, # Break
            14: True,  # 2pm
            15: True,  # 3pm
        }
    },
    7: {
        "faculty": "Science",
        "year": 4,
        "preferred_library_id": 3,  # Science Library
        "schedule": {
            10: True,  # 10am
            11: True,  # 11am
            12: True,  # 12pm
            13: False, # Lunch
            14: True,  # 2pm
            15: True,  # 3pm
            16: True,  # 4pm
            17: True,  # 5pm
        }
    },
    8: {
        "faculty": "Medicine",
        "year": 2,
        "preferred_library_id": 4,  # Medical Library
        "schedule": {
            8: True,   # 8am
            9: True,   # 9am
            10: True,  # 10am
            11: True,  # 11am
            12: False, # Lunch
            13: False, # Break
            14: False, # Break
            15: True,  # 3pm
            16: True,  # 4pm
        }
    },
    9: {
        "faculty": "Business",
        "year": 3,
        "preferred_library_id": 5,  # Business Library
        "schedule": {
            11: True,  # 11am
            12: True,  # 12pm
            13: True,  # 1pm
            14: True,  # 2pm
            15: False, # Break
            16: True,  # 4pm
            17: True,  # 5pm
            18: True,  # 6pm
        }
    }
}
