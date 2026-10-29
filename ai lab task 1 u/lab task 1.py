def check_result(marks, attendance, disciplinary_issue):
    """
    Evaluates whether a student passes or fails based on marks, attendance, and disciplinary status.
    
    Parameters:
    - marks (int or float): Student's marks percentage.
    - attendance (int or float): Student's attendance percentage.
    - disciplinary_issue (bool): True if the student has a disciplinary record, False otherwise.
    
    Returns:
    - tuple: (status [str], reason [str])
    """
    if marks < 70:
        return "FAIL", "Marks are below 70."

    if attendance < 75:
        return "FAIL", "Attendance is below 75%."

    if disciplinary_issue:
        return "FAIL", "Student has a disciplinary issue."

    return "PASS", "All conditions satisfied."


# Test Cases
test_cases = [
    {
        "name": "Case 1",
        "marks": 85,
        "attendance": 90,
        "disciplinary_issue": False
    },
    {
        "name": "Case 2",
        "marks": 65,
        "attendance": 90,
        "disciplinary_issue": False
    },
    {
        "name": "Case 3",
        "marks": 85,
        "attendance": 70,
        "disciplinary_issue": False
    },
    {
        "name": "Case 4",
        "marks": 85,
        "attendance": 90,
        "disciplinary_issue": True
    }
]


# Main Execution Loop
if __name__ == "__main__":
    for case in test_cases:
        result, reason = check_result(
            case["marks"],
            case["attendance"],
            case["disciplinary_issue"]
        )

        print(f"--- {case['name']} ---")
        print(f"Input: {case}")
        print(f"Expected Result: {'PASS' if case['name'] == 'Case 1' else 'FAIL'}")
        print(f"Actual Result: {result}")
        print(f"Reason: {reason}")
        print("-" * 50)
