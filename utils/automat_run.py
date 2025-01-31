import os
import subprocess

# Define directories
generated_dir = "/opt/CFD-Benchmark/results/generate_code"
solution_dir = "/opt/CFD-Benchmark/solution_python_files"


def run_python_scripts(directory):
    """
    Runs all Python scripts in the given directory.
    """
    for script in os.listdir(directory):
        if script.endswith(".py"):
            script_path = os.path.join(directory, script)

            try:
                # Run the script
                subprocess.run(["python", script_path], check=True)
                print(f"Successfully executed {script}")

            except subprocess.CalledProcessError as e:
                print(f"Error executing {script}: {e}")


# Run scripts from both folders
run_python_scripts(generated_dir)
run_python_scripts(solution_dir)
