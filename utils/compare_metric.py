import os
import json
import re

# Define paths
results_folder = "/opt/CFD-Benchmark/results/generate_code"
json_filename = "/opt/CFD-Benchmark/results/output_pred.json"

# Ensure the results folder exists
if not os.path.exists(results_folder):
    print(f"Folder '{results_folder}' does not exist.")
else:
    # Get all Python files in the results folder
    py_files = [f for f in os.listdir(results_folder) if f.endswith(".py")]

    for py_file in py_files:
        file_path = os.path.join(results_folder, py_file)

        # Read the Python file content
        with open(file_path, "r") as file:
            code_lines = file.readlines()

        # Ensure imports are present at the beginning
        new_code = ["import os\n", "import json\n"] + code_lines

        # Identify solution variables by looking for plt.plot or array assignments
        solution_vars = set()
        for line in code_lines:
            match = re.search(r"plt\.plot\s*\(\s*([^,]+)\s*,\s*([^,)\s]+)", line)
            if match:
                solution_vars.add(match.group(2))  # The second parameter is usually the solution

        # If no plt.plot is found, try detecting key variable assignments (e.g., `u = ...`)
        if not solution_vars:
            for line in code_lines:
                match = re.search(r"(\w+)\s*=\s*.*np\.zeros|\w+\s*=\s*.*np\.array", line)
                if match:
                    solution_vars.add(match.group(1))

        # Prepare the JSON-saving code snippet
        json_saving_code = """
##############################################
# The following lines are used to print output
##############################################

# Identify the filename of the running script
script_filename = os.path.basename(__file__)

# Define the JSON file
json_filename = "/opt/CFD-Benchmark/results/output_pred.json"

# Load existing JSON data if the file exists
if os.path.exists(json_filename):
    with open(json_filename, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = {}  # Handle empty or corrupted file
else:
    data = {}

# Save filename and output array in a structured format
data[script_filename] = {
    "filename": script_filename,
"""

        # Append solution variables dynamically
        for var in solution_vars:
            json_saving_code += f'    "{var}": {var}.tolist(),\n'

        json_saving_code += """}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")
"""

        # Append the JSON-saving code to the script
        new_code.append(json_saving_code)

        # Write back the modified script
        with open(file_path, "w") as file:
            file.writelines(new_code)

        print(f"Updated script: {py_file}")

    print("\nAll scripts updated successfully!")
