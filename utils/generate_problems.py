import json

# Define JSON file path
json_filename = "/opt/CFD-Benchmark/data/cfd_problems.json"  # Update with your actual file path


def update_methods(filename, tokens):
    """
    Updates the "method" field for all problems to "finite difference, fully-discrete schemes".
    """
    with open(filename, "r") as file:
        data = json.load(file)

    # Update the "method" field
    for problem in data["problems"]:
        problem["method"] = tokens

    # Save changes
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f'Updated "method" field for all problems in {filename}')
    return data  # Return updated data for further processing


def add_problems(filename, tokens):
    """
    Creates a copy of problems with "finite difference, semi-discrete methods"
    for those that have "finite difference, fully-discrete schemes" and appends them to the JSON.
    """
    with open(filename, "r") as file:
        data = json.load(file)

    # Create modified problems
    new_problems = []
    for problem in data["problems"]:
        if problem["method"] == "finite difference, fully-discrete schemes":
            new_problem = problem.copy()  # Copy original problem
            new_problem["method"] = tokens
            new_problems.append(new_problem)

    # Append new problems
    data["problems"].extend(new_problems)

    # Save updated JSON
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f'Added new problems with "finite difference, semi-discrete methods" to {filename}')


# Usage
# update_methods(json_filename, tokens="finite difference, fully-discrete schemes")  # Step 1: Update methods
# add_problems(json_filename, tokens="finite difference, semi-discrete methods")  # Step 2: Append new problems

