import os


def modify_python_files(folder_path, json_output_path="/opt/CFD-Benchmark/MMS/result/output.json"):
    # Code to append at the end of each file
    append_code = f"""
# Identify the filename of the running script
script_filename = os.path.basename(__file__)

# Define the JSON file
json_filename = "{json_output_path}"

# Load existing JSON data if the file exists
if os.path.exists(json_filename):
    with open(json_filename, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = {{}}  # Handle empty or corrupted file
else:
    data = {{}}

# Save filename and output array in a structured format
data[script_filename] = {{
    "filename": script_filename,
    "u": u.tolist()  # Convert NumPy array to list for JSON serialization
}}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {{script_filename}} to {{json_filename}}")
"""

    # Iterate through all Python files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".py"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as file:
                lines = file.readlines()

            # Check if imports are already present
            has_import_os = any("import os" in line for line in lines)
            has_import_json = any("import json" in line for line in lines)

            modified_lines = []

            # Add missing imports at the beginning of the file
            if not has_import_os:
                modified_lines.append("import os\n")
            if not has_import_json:
                modified_lines.append("import json\n")

            modified_lines += lines  # Add the existing code
            modified_lines.append(append_code)  # Append the new code

            # Write the modified content back to the file
            with open(file_path, "w") as file:
                file.writelines(modified_lines)

            print(f"Modified {filename} successfully!")


# Example usage
folder_path = "/opt/CFD-Benchmark/MMS/generated_code/sonnet-35/1"
modify_python_files(folder_path)

