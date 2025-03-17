import os
import json
import re
import subprocess
from openai import OpenAI
import logging
logging.basicConfig(
    filename="/opt/CFD-Benchmark/PDE_Benchmark_6/report/execution_results.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
# === OpenAI API Configuration ===
api_key = "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"
client = OpenAI(api_key=api_key)

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
PROMPTS_FILE = os.path.join(ROOT_DIR, "prompt", "PDE_TASK_PROMPT.json")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "solver")
LOG_FILE = os.path.join(ROOT_DIR, "report/execution_results.log")
USAGE_FILE = os.path.join(ROOT_DIR, "report/token_usage_summary.json")

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Token Usage Tracking ===
total_input_tokens = 0
total_output_tokens = 0

# === Load PDE Prompts ===
with open(PROMPTS_FILE, "r") as file:
    pde_prompts = json.load(file)


# === Function to Execute Python Script and Capture Errors ===
def execute_python_script(filepath):
    """ Runs the generated Python script and captures errors. """
    try:
        result = subprocess.run(["python3", filepath], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            logging.info("Execution successful, no errors detected.")
            return "Execution successful, no errors detected."
        return result.stderr.strip()
    except subprocess.TimeoutExpired:
        logging.info("⚠️ Timeout Error: Script took too long to execute.")
        return "⚠️ Timeout Error: Script took too long to execute."


# === Function to Generate Code from LLM ===
def generate_code(task_name, prompt, max_retries=10):
    """ Calls LLM API to generate Python code with feedback updates if errors occur. """
    retries = 0
    while retries < max_retries:
        print(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
        logging.info(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
        try:
            # Call OpenAI GPT-4 API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in CFD and Python."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=7000,
                temperature=1.0
            )

            # Extract model response
            model_response = response.choices[0].message.content.strip()

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
                logging.info(f"\n\n[Feedback]: The previous generated code had the following error:\n{execution_feedback}\nPlease correct it.")
                prompt += f"\n\n[Feedback]: The previous generated code had the following error:\n{execution_feedback}\nPlease correct it."

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
