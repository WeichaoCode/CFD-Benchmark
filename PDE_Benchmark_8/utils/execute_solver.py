import os
import subprocess
from datetime import datetime


def call_execute_solver(llm_model, prompt_json):
    # Get current time with microseconds
    timestamp = datetime.now().strftime("%H-%M-%S-%f")  # %f gives microseconds
    # Define the directory where generated solver scripts are stored
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
    GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, f"solver/{llm_model}/{prompt_json}")
    GENERATED_SOLVERS_SAVE_DIR = os.path.join(ROOT_DIR, "report")
    # Define the log file for execution results
    LOG_FILE = os.path.join(GENERATED_SOLVERS_SAVE_DIR, f"execution_results_{llm_model}_{prompt_json}_{timestamp}.log")

    # Get all Python files in the solvers directory
    python_files = [f for f in os.listdir(GENERATED_SOLVERS_DIR) if f.endswith(".py")]

    # Ensure there are solver scripts to run
    if not python_files:
        print("No Python solver scripts found in the directory.")
        exit()
    # Initialize counters for pass and fail
    pass_count = 0
    fail_count = 0
    # Open a log file to save execution results
    with open(LOG_FILE, "w") as log:
        log.write("====== Execution Results for Generated Solvers ======\n\n")

        for script in python_files:
            script_path = os.path.join(GENERATED_SOLVERS_DIR, script)
            print(f"🔹 Running: {script} ...")

            # Run the script and capture the output
            try:
                result = subprocess.run(
                    ["python3", script_path],
                    capture_output=True,
                    text=True,
                    timeout=300  # Set a timeout of 5 minutes per script
                )

                # Write execution results to log file
                log.write(f"--- Running: {script} ---\n")
                log.write(f"Exit Code: {result.returncode}\n")
                log.write("Output:\n")
                log.write(result.stdout + "\n")
                log.write("Errors:\n")
                log.write(result.stderr + "\n")
                log.write("-" * 50 + "\n\n")

                # Print execution status
                if result.returncode == 0:
                    print(f"✅ {script} executed successfully.")
                    pass_count += 1  # Increment pass count
                else:
                    print(f"❌ {script} encountered an error (check logs).")
                    fail_count += 1  # Increment fail count

            except subprocess.TimeoutExpired:
                log.write(f"--- Running: {script} ---\n")
                log.write("⚠️ Timeout Error: Script took too long to execute.\n")
                log.write("-" * 50 + "\n\n")
                print(f"⚠️ Timeout: {script} took too long and was skipped.")
                fail_count += 1  # Increment fail count for timeout

            # Log the summary of pass and fail counts
            log.write("\n\n====== Execution Summary ======\n")
            log.write(f"Total Scripts Passed: {pass_count}\n")
            log.write(f"Total Scripts Failed: {fail_count}\n")

    print(f"\n🎯 Execution completed. Results saved in: {LOG_FILE}")


# call_execute_solver('gpt-4o', 'prompts_both_instructions')
#
# call_execute_solver('gpt-4o', 'prompts_instruction_1')
#
# call_execute_solver('gpt-4o', 'prompts_instruction_2')

call_execute_solver('gpt-4o', 'prompts_no_instruction')
