import os
import json
import subprocess

# Define directories
generated_dir = "generated_python_files"
solution_dir = "solution_python_files"

# Define JSON output files
output_generate_json = "output_generate.json"
output_true_json = "output_true.json"


def run_python_scripts(directory, output_json):
    """
    Runs all Python scripts in the given directory and saves their output to a JSON file.
    """
    data = {}

    # Load existing JSON data if available
    if os.path.exists(output_json):
        with open(output_json, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}  # Handle empty or corrupted file

    # Iterate through all Python files in the directory
    for script in os.listdir(directory):
        if script.endswith(".py"):
            script_path = os.path.join(directory, script)

            try:
                # Run the script and capture its output
                result = subprocess.run(
                    ["python", script_path], capture_output=True, text=True, check=True
                )

                # Process output (assuming script prints a NumPy array or JSON-like structure)
                output_data = result.stdout.strip()

                # Save the output in JSON format
                data[script] = {
                    "filename": script,
                    "output": output_data
                }

                print(f"Successfully executed {script}")

            except subprocess.CalledProcessError as e:
                print(f"Error executing {script}: {e}")

    # Save updated JSON data
    with open(output_json, "w") as file:
        json.dump(data, file, indent=4)

    print(f"All outputs saved to {output_json}")


# Run scripts from both folders
run_python_scripts(generated_dir, output_generate_json)
run_python_scripts(solution_dir, output_true_json)
