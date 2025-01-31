import json


def generate_pde_problems_json(N, filename="pde_problems.json"):
    """
    Generate a JSON file with N PDE problems, where all fields are blank.

    Parameters:
    - N (int): Number of PDE problems to generate.
    - filename (str): Name of the JSON file to save the problems.
    """
    problems = []

    for i in range(1, N + 1):
        problem = {
            "key": i,
            "name": "",
            "equation": "",
            "physical_properties": "",
            "boundary conditions": {},
            "initial conditions": {},
            "spatial domain": "",
            "temporal domain": "",
            "mesh": "",
            "numerical method": "",
            "other information": ""
        }
        problems.append(problem)

    # Save to JSON file
    with open(filename, "w") as json_file:
        json.dump({"problems": problems}, json_file, indent=4)

    print(f"JSON file '{filename}' with {N} PDE problems created successfully!")


def autofill_cfd_problems(json_filename="cfd_problem.json"):
    """
    Auto-fills missing fields in cfd_problem.json with default values.

    Parameters:
    - json_filename (str): The name of the JSON file to update.
    """
    # Default values
    default_values = {
        "spatial domain": "x: [0, 2]",
        "temporal domain": "T: [0, 2]",
        "mesh": "structured",
        "numerical method": "finite difference, fully-discrete schemes",
        "other information": "write python code, do not call external packages"
    }

    # Load existing JSON data
    with open(json_filename, "r") as file:
        data = json.load(file)

    # Loop through each problem and add missing fields
    for problem in data["problems"]:
        for key, default_value in default_values.items():
            if key in problem and (problem[key] == "" or problem[key] is None):
                problem[key] = default_value  # Fill missing value

    # Save updated JSON file
    with open(json_filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Successfully updated '{json_filename}' with default values!")


# Example Usage:
autofill_cfd_problems("/opt/CFD-Benchmark/MMS/data/cfd_problem.json")  # Update cfd_problem.json

# Example Usage:
# generate_pde_problems_json(12, "/opt/CFD-Benchmark/MMS/data/cfd_problem.json")  # Generate JSON file with 10 empty
# PDE problems
