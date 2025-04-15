import json

# Function to remove unwanted keys from the problem
def remove_keys_from_problem(problem):
    keys_to_remove = ["numerical method", "expert instruction group 1", "expert instruction group 2"]
    for key in keys_to_remove:
        if key in problem:
            del problem[key]
    return problem

# Load the JSON file
input_file = "/opt/CFD-Benchmark/PDE_Benchmark_8/prompt/PDE_TASK_CONFIGURATION.json"  # Replace with your actual file path
output_file = "/opt/CFD-Benchmark/PDE_Benchmark_8/prompt/PDE_TASK_QUESTION_ONLY.json"  # Replace with the desired output file path

with open(input_file, 'r') as file:
    data = json.load(file)

# Remove specified keys for each problem
for problem_name, problem_data in data.items():
    data[problem_name] = remove_keys_from_problem(problem_data)

# Save the updated data back to a new JSON file
with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Updated JSON file saved to: {output_file}")


