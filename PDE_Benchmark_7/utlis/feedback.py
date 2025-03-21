"""
- Author: Weichao Li
- Date: 2025-03-20
- Brief Explanation:
########################################################################################################################
This script interacts with the OpenAI GPT-4o API to generate Python code for solving CFD (Computational Fluid Dynamics)
problems using the Finite Difference Method (FDM). It generates code, tracks token usage, and estimates the cost of the
API calls. Additionally, it executes the generated code, captures errors, and logs results to a file. It has feedback /
reviewer and the maximum number is 5.
# REF: this code generated with the help of GPT-4o
########################################################################################################################
"""
import os
import json
import re
import subprocess
from openai import OpenAI
import logging
from datetime import datetime

# Get current time with microseconds
timestamp = datetime.now().strftime("%H-%M-%S-%f")  # %f gives microseconds

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
PROMPTS_FILE = os.path.join(ROOT_DIR, "prompt", "PDE_TASK_PROMPT.json")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "solver/gpt-4o")
LOG_FILE = os.path.join(ROOT_DIR, f"report/execution_gpt-4o_results_{timestamp}.log")
# LOG_FILE = os.path.join(ROOT_DIR, f"report/execution_gpt-4o_results_{timestamp}.log")
# USAGE_FILE = os.path.join(ROOT_DIR, f"report/token_usage_summary_gpt-4o_{timestamp}.log")

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("####################################################################################################")
logging.info("Using the GPT-4o, change temperature to 0.0, run 2D_Burgers_Equation again since it fails")

# === OpenAI API Configuration ===
api_key = "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"  # Replace this with your OpenAI API key
client = OpenAI(api_key=api_key)

# === Token Usage Tracking ===
total_input_tokens = 0
total_output_tokens = 0
total_cost = 0  # To keep track of the total cost (optional, needs pricing rate)

# === Load PDE Prompts ===
with open(PROMPTS_FILE, "r") as file:
    pde_prompts = json.load(file)


# === Function to Execute Python Script and Capture Errors and Warnings ===
def execute_python_script(filepath):
    """ Runs the generated Python script and captures errors and warnings. """
    try:
        result = subprocess.run(["python3", filepath], capture_output=True, text=True, timeout=60)
        stderr_output = result.stderr.strip()

        if result.returncode == 0:
            if "warning" in stderr_output.lower():
                logging.warning(f"Execution completed with warnings:\n{stderr_output}")
                return f"⚠️ Execution completed with warnings:\n{stderr_output}"
            else:
                logging.info("Execution successful, no errors detected.")
                return "Execution successful, no errors detected."

        logging.error(f"Execution failed with errors:\n{stderr_output}")
        return stderr_output

    except subprocess.TimeoutExpired:
        logging.warning("⚠️ Timeout Error: Script took too long to execute.")
        return "⚠️ Timeout Error: Script took too long to execute."


# === Function to Generate Code from LLM ===
def generate_code(task_name, prompt, max_retries=5):
    """ Calls LLM API to generate Python code with feedback updates if errors occur. """
    if task_name not in {"2D_Burgers_Equation"}:
        return
    retries = 0
    original_prompt = prompt  # Keep the original prompt unchanged
    while retries < max_retries:
        print(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
        logging.info(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
        updated_prompt = original_prompt
        try:
            # Call OpenAI GPT-4o API
            response = client.chat.completions.create(
                model="gpt-4o",  # Specify the model
                messages=[
                    {"role": "system", "content": "You are an expert in Computational Fluid Dynamics (CFD) and Python "
                                                  "programming, specializing in solving problems using the Finite "
                                                  "Difference Method (FDM). Your task is to assist in developing "
                                                  "efficient Python code for solving CFD problems using FDM. You "
                                                  "should focus on writing clear, modular, and optimized code for "
                                                  "numerical simulations, including proper discretization of partial "
                                                  "differential equations (PDEs). Prioritize best practices in "
                                                  "scientific computing, ensuring that the code is well-commented and "
                                                  "easy to understand for implementation and future improvements."},
                    {"role": "user", "content": updated_prompt}
                ],
                max_tokens=7000,
                temperature=0.0
            )

            # Extract model response
            model_response = response.choices[0].message.content.strip()

            # Extract token usage from the response
            total_tokens = response.usage.total_tokens  # Get the total tokens used for this request
            input_tokens = len(updated_prompt.split())  # Get the number of tokens used by the prompt (input tokens)
            output_tokens = total_tokens - input_tokens  # Subtract input tokens from total tokens to get output tokens

            # Track the total token usage and cost
            global total_input_tokens, total_output_tokens, total_cost
            total_input_tokens += input_tokens  # Update total input tokens count
            total_output_tokens += output_tokens  # Update total output tokens count

            # Calculate cost (example, adjust as per pricing details)
            cost_per_input_token = 2.50 / 1_000_000  # Cost per input token (example, adjust accordingly)
            cost_per_output_token = 10.00 / 1_000_000  # Cost per output token (example, adjust accordingly)
            total_cost += (input_tokens * cost_per_input_token) + (output_tokens * cost_per_output_token)

            # Log the usage and estimated cost
            logging.info(f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")
            logging.info(
                f"Estimated cost for this request: ${input_tokens * cost_per_input_token + output_tokens * cost_per_output_token:.6f}")

            # Extract Python code using regex
            code_match = re.findall(r"```python(.*?)```", model_response, re.DOTALL)
            extracted_code = code_match[0].strip() if code_match else "# No valid Python code extracted"

            # Save the full model response
            response_file = os.path.join(OUTPUT_FOLDER, f"{task_name}.txt")
            with open(response_file, "w") as txt_file:
                txt_file.write(model_response)

            # Save the extracted Python code
            script_path = os.path.join(OUTPUT_FOLDER, f"{task_name}.py")
            with open(script_path, "w") as py_file:
                py_file.write(extracted_code)

            print(f"✅ Code saved: {script_path}")

            # Execute and check for errors
            execution_feedback = execute_python_script(script_path)

            if "no errors detected" in execution_feedback:
                print(f"🎯 {task_name} executed successfully without syntax errors.")
                logging.info(f"🎯 {task_name} executed successfully without syntax errors.")
                return  # Exit function if no errors

            else:
                print(f"❌ Error detected in {task_name}, refining prompt...")
                logging.info(f"❌ Error detected in {task_name}, refining prompt...")
                logging.info(
                    f"\n\n[Feedback]: The previous generated code had the following error:\n{execution_feedback}\nPlease correct it.")
                updated_prompt = f"{original_prompt}\n\n[Feedback]: The previous generated code had the following error:\n{execution_feedback}\nPlease correct it."

            retries += 1

        except Exception as e:
            print(f"❌ API Call Error for {task_name}: {str(e)}")
            logging.info(f"❌ API Call Error for {task_name}: {str(e)}")
            return  # Stop retrying if API call fails

    print(f"⚠️ Max retries reached for {task_name}. Check logs for remaining errors.")
    logging.info(f"⚠️ Max retries reached for {task_name}. Check logs for remaining errors.")


# === Process Each Prompt ===
for task_name, prompt in pde_prompts["prompts"].items():
    generate_code(task_name, prompt)

print("\n🎯 Execution completed. Check the solver directory for generated files.")
logging.info("\n🎯 Execution completed. Check the solver directory for generated files.")

# Log the token usage and cost summary
logging.info(f"Total Input Tokens: {total_input_tokens}")
logging.info(f"Total Output Tokens: {total_output_tokens}")
logging.info(f"Total Estimated Cost: ${total_cost:.6f}")
