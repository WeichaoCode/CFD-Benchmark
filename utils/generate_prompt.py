# import json
#
#
# def generate_prompt(problem):
#     prompt = f"""Solve the {problem['name']} problem using Python.
#
# Equation:
# {problem['equation_format']}
#
# Physical Properties:
# {problem['physical_properties']}
#
# Boundary Conditions:
# {problem['boundary_conditions']}
#
# Initial Conditions:
# {problem['initial_conditions']}
#
# Domain: {problem['domain']}
# Grid: Nx={problem['grid']['Nx']}, Ny={problem['grid']['Ny']}
#
# Write Python code to numerically solve this equation using finite difference or finite volume method. Include necessary libraries and comments to explain the implementation."""
#     return prompt
#
#
# # Load CFD problems from JSON file
# with open("/opt/CFD-Benchmark/data/cfd_problems.json", "r") as file:
#     data = json.load(file)
#
# # Generate prompts and store them in a new JSON structure
# prompts_data = {"prompts": []}
#
# for problem in data["problems"]:
#     prompt_text = generate_prompt(problem)
#     prompts_data["prompts"].append({
#         "name": problem["name"],
#         "prompt": prompt_text
#     })
#
# # Save the generated prompts into a new JSON file
# output_filename = "/opt/CFD-Benchmark/data/cfd_prompts.json"
# with open(output_filename, "w") as output_file:
#     json.dump(prompts_data, output_file, indent=4)
#
# print(f"Prompts saved successfully to {output_filename}")
import json

input_json = "/opt/CFD-Benchmark/MMS/data/cfd_problem.json"  # Your original problems JSON file
output_json = "/opt/CFD-Benchmark/MMS/data/cfd_prompt.json"    # The new JSON file with "name" and "prompt"

# Load the existing problems JSON file
with open(input_json, "r") as file:
    data = json.load(file)

# Extract relevant fields dynamically
prompt_data = {"prompts": []}

for problem in data["problems"]:
    prompt_text = f"Simulate the problem: {problem['name']}. "

    # Iterate through all keys dynamically
    for key, value in problem.items():
        if key != "name":  # Skip the name field as it's already included
            prompt_text += f"{key.replace('_', ' ').capitalize()}: {value}. "

    # Append the structured prompt
    prompt_data["prompts"].append({
        "key": problem["key"],
        "name": problem["name"],
        "prompt": prompt_text.strip()
    })

# Save the extracted data to a new JSON file
with open(output_json, "w") as file:
    json.dump(prompt_data, file, indent=4)

print(f"Generated {output_json} successfully!")
