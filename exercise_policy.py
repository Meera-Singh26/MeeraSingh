# exercise_policy.py

def get_exercise(problem):

    mapping = {
        1: "knee",
        2: "joint",
        3: "muscle",
        4: "hip"
    }

    return mapping.get(problem, "knee")