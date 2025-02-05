import os
import subprocess
import json
import re

# Define directories
generated_dir = "/opt/CFD-Benchmark/MMS/generated_code/gpt-4/2"
solution_dir = "/opt/CFD-Benchmark/solution_python_files"

# JSON file to save the results
output_json = "/opt/CFD-Benchmark/MMS/generated_code/gpt-4/2/execution_summary.json"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_json), exist_ok=True)

# Regex pattern to detect Python warnings
warning_pattern = re.compile(r"(RuntimeWarning|UserWarning|DeprecationWarning|Warning)", re.IGNORECASE)

def run_python_scripts(directory):
    """
    Runs all Python scripts in the given directory.
    Categorizes results as Passed, Failed, or Warning based on output.
    """
    results = {
        "summary": {"passed": 0, "failed": 0, "warnings": 0},
        "details": {}
    }

    for script in os.listdir(directory):
        if script.endswith(".py"):
            script_path = os.path.join(directory, script)
            print(f"Running {script}...")

            try:
                # Run the script and capture both stdout and stderr
                process = subprocess.run(
                    ["python", script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Combine stdout and stderr to analyze for warnings/errors
                output_combined = process.stdout + process.stderr

                # Check for errors (non-zero exit code)
                if process.returncode != 0:
                    print(f"❌ Failed executing {script}")
                    results["details"][script] = {
                        "status": "failed",
                        "error_type": "ExecutionError",
                        "error_message": process.stderr.strip() or "Unknown error"
                    }
                    results["summary"]["failed"] += 1

                # Check for warnings (even if script succeeded)
                elif warning_pattern.search(output_combined):
                    warning_message = warning_pattern.findall(output_combined)
                    print(f"⚠️ Warning detected in {script}: {warning_message}")

                    results["details"][script] = {
                        "status": "warning",
                        "error_type": "RuntimeWarning",
                        "error_message": process.stderr.strip() or "Warning detected"
                    }
                    results["summary"]["warnings"] += 1

                # If no errors or warnings, mark as success
                else:
                    print(f"✅ Successfully executed {script}")
                    results["details"][script] = {
                        "status": "passed",
                        "error_type": None,
                        "error_message": None
                    }
                    results["summary"]["passed"] += 1

            except Exception as ex:
                # Handle unexpected errors
                print(f"❌ Error while running {script}: {ex}")
                results["details"][script] = {
                    "status": "failed",
                    "error_type": type(ex).__name__,
                    "error_message": str(ex)
                }
                results["summary"]["failed"] += 1

    # Save the results to a JSON file
    with open(output_json, "w") as json_file:
        json.dump(results, json_file, indent=4)

    # Final summary
    print("\nExecution Summary:")
    print(f"✅ Passed: {results['summary']['passed']}")
    print(f"⚠️ Warnings: {results['summary']['warnings']}")
    print(f"❌ Failed: {results['summary']['failed']}")
    print(f"Results saved to {output_json}")


# Run the scripts
run_python_scripts(generated_dir)

