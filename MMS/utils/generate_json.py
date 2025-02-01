import json
import itertools


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


def update_multiple_problems(json_filename, problem_keys, updates):
    """
    Updates specific key values for multiple problems in the JSON file.

    Parameters:
    - json_filename (str): Path to the JSON file containing problems.
    - problem_keys (list): List of key numbers of problems to update.
    - updates (dict): Dictionary where keys are JSON fields to update and values are new values.

    Example:
    update_multiple_problems("cfd_problems.json", [1, 3, 5], {
        "boundary conditions": "u = 1 at x = 0, 2",
        "numerical method": "finite volume method"
    })
    """
    # Load the JSON file
    with open(json_filename, "r") as file:
        data = json.load(file)

    # Iterate through each problem and update the ones matching the given keys
    for problem in data["problems"]:
        if problem["key"] in problem_keys:
            for field, new_value in updates.items():
                if field in problem:
                    problem[field] = new_value
                else:
                    print(f" Warning: '{field}' not found in problem {problem['key']}")

    # Save the updated JSON file
    with open(json_filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Successfully updated problems {problem_keys} in '{json_filename}'")


def generate_new_prompts_for_keys(json_filename, selected_keys, key_modifications):
    """
    Generates new problem prompts by modifying certain keys for selected problems.

    Parameters:
    - json_filename (str): Path to the JSON file containing problems.
    - selected_keys (list): List of problem keys to modify.
    - key_modifications (dict): Dictionary where keys are JSON fields to modify,
      and values are lists of possible choices.

    Example:
    generate_new_prompts_for_keys("cfd_problems.json", [1, 2], {
        "numerical method": ["finite volume", "spectral method"],
        "physical_properties": ["wave speed c = 1", "wave speed c = 2"]
    })
    """
    # Load the existing JSON file
    with open(json_filename, "r") as file:
        data = json.load(file)

    # Extract problems
    existing_problems = data["problems"]

    # Dictionary to store new problems mapped to the original key index
    new_problems_map = {}

    # Generate all possible combinations of key modifications
    keys_to_modify = list(key_modifications.keys())
    possible_combinations = list(itertools.product(*key_modifications.values()))

    # Generate new problems for selected keys
    for idx, problem in enumerate(existing_problems):
        if problem["key"] in selected_keys:
            new_problems = []
            for combination in possible_combinations:
                new_problem = problem.copy()  # Copy original problem
                new_problem["key"] = None  # Will be assigned later

                # Apply modifications
                for i, key in enumerate(keys_to_modify):
                    new_problem[key] = combination[i]

                new_problems.append(new_problem)

            # Store new problems to be inserted after the original problem
            new_problems_map[idx] = new_problems

    # Reconstruct the list with correct key ordering
    new_problem_list = []
    new_key = 1  # Start key numbering from 1

    for idx, problem in enumerate(existing_problems):
        problem["key"] = new_key
        new_problem_list.append(problem)
        new_key += 1

        # If this problem has new variations, insert them after it
        if idx in new_problems_map:
            for new_problem in new_problems_map[idx]:
                new_problem["key"] = new_key
                new_problem_list.append(new_problem)
                new_key += 1

    # Save the updated JSON file
    data["problems"] = new_problem_list
    with open(json_filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Successfully generated new problems and updated key ordering in {json_filename}")


def delete_prompts_by_keys(json_filename, keys_to_delete):
    """
    Deletes prompts with specific key numbers and rearranges the remaining keys sequentially.

    Parameters:
    - json_filename (str): Path to the JSON file containing problems.
    - keys_to_delete (list): List of keys to be deleted.

    Example:
    delete_prompts_by_keys("cfd_problems.json", [1, 4, 6])
    """
    # Load JSON data
    with open(json_filename, "r") as file:
        data = json.load(file)

    # Remove problems with specified keys
    updated_problems = [problem for problem in data["problems"] if problem["key"] not in keys_to_delete]

    # Reassign keys sequentially starting from 1
    for i, problem in enumerate(updated_problems, start=1):
        problem["key"] = i

    # Save updated JSON file
    data["problems"] = updated_problems
    with open(json_filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Successfully deleted keys {keys_to_delete} and updated {json_filename}")


# Example Usage:
# delete_prompts_by_keys("/opt/CFD-Benchmark/MMS/data/cfd_problem.json", [1,7,13,19,3,9,15,21,5,11,17,23])

# Example Usage:
# generate_new_prompts_for_keys("/opt/CFD-Benchmark/MMS/data/cfd_problem.json", [1, 2, 3, 4], {
#     "numerical method": ["finite difference: Forward in Time, Centered in Space (FTCS)",
#                          "finite difference: First Order Upwind (FOU)",
#                          "finite difference: Leapfrog",
#                          "finite difference: Lax-Friedrichs",
#                          "finite difference: Lax-Wendroff",
#                          "finite difference: Beam-Warming"]
# })

# Example Usage:
# update_multiple_problems("/opt/CFD-Benchmark/MMS/data/cfd_problem.json", list(range(1, 29)), {
#     "other information": "Write a Python program to solve the given PDE using a numerical method. Do not use external "
#                          "packages. Ensure the solution is stable by applying von Neumann stability analysis. Plot "
#                          "the solution at key time steps: t = 0, t = T/4, t = T/2, and t = T in the same figure, "
#                          "figure title is equation name + numerical method."
# })

# Example Usage:
# autofill_cfd_problems("/opt/CFD-Benchmark/MMS/data/cfd_problem.json")  # Update cfd_problem.json

# Example Usage:
# generate_pde_problems_json(12, "/opt/CFD-Benchmark/MMS/data/cfd_problem.json")  # Generate JSON file with 10 empty
# PDE problems
