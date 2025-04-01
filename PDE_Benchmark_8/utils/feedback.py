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

# === Token Usage Tracking ===
total_input_tokens = 0
total_output_tokens = 0
total_cost = 0  # To keep track of the total cost (optional, needs pricing rate)


def call_api(llm_model, prompt_json, temperature=0.0):
    # Get current time with microseconds
    timestamp = datetime.now().strftime("%H-%M-%S-%f")  # %f gives microseconds

    # === Paths ===
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
    PROMPTS_FILE = os.path.join(ROOT_DIR, "prompt", prompt_json)
    OUTPUT_FOLDER = os.path.join(ROOT_DIR, f"solver/{llm_model}/{prompt_json}")
    LOG_FILE = os.path.join(ROOT_DIR, f"report/{llm_model}_{prompt_json}_{timestamp}.log")
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
    logging.info(f"Using the {llm_model}, change temperature to {temperature}, use the prompt {prompt_json}")

    # === OpenAI API Configuration ===
    api_key = "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"  # Replace this with your OpenAI API key
    client = OpenAI(api_key=api_key)

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
        # if task_name not in {"2D_Steady_Heat_Equation_Gauss"}:
        #     return
        retries = 0
        original_prompt = prompt  # Keep the original prompt unchanged
        # Initialize an empty list to store the conversation history
        if llm_model == "o1-mini":
            conversation_history = [
                {"role": "user",
                 "content": "You are a highly skilled assistant capable of generating Python code to solve CFD problems "
                            "using appropriate numerical methods."
                            "Given the problem description, you should reason through the problem and determine the best "
                            "approach for discretizing and solving it,"
                            "while respecting the specified boundary conditions, initial conditions, and domain.\n"
                            "For unsteady problems, save only the solution at the final time step. For 1D problems, "
                            "save a 1D array; for 2D problems, save a 2D array.\n"
                            "Ensure the code follows the user's specifications and saves the requested variables exactly "
                            "as named in `save_values`.\n"
                            "Your task is to generate the correct, fully runnable Python code for solving the problem "
                            "without additional explanations."
                 },  # System prompt to guide the LLM

                {"role": "user",
                 "content": original_prompt +
                            "If it is an unsteady problem, only save the solution at the final time step "
                            "If the problem is 1D, the saved array should be 1D. "
                            "If the problem is 2D, the saved array should be 2D."},  # Add the initial user prompt
            ]
        else:
            conversation_history = [
                {"role": "system",
                 "content": "You are a highly skilled assistant capable of generating Python code to solve CFD problems "
                            "using appropriate numerical methods."
                            "Given the problem description, you should reason through the problem and determine the best "
                            "approach for discretizing and solving it,"
                            "while respecting the specified boundary conditions, initial conditions, and domain.\n"
                            "For unsteady problems, save only the solution at the final time step. For 1D problems, "
                            "save a 1D array; for 2D problems, save a 2D array.\n"
                            "Ensure the code follows the user's specifications and saves the requested variables exactly "
                            "as named in `save_values`.\n"
                            "Your task is to generate the correct, fully runnable Python code for solving the problem "
                            "without additional explanations."
                 },  # System prompt to guide the LLM

                {"role": "user",
                 "content": original_prompt +
                            "If it is an unsteady problem, only save the solution at the final time step "
                            "If the problem is 1D, the saved array should be 1D. "
                            "If the problem is 2D, the saved array should be 2D."},  # Add the initial user prompt
            ]
        while retries < max_retries:
            print(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
            logging.info(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
            try:
                if llm_model == "o1-mini":
                    # Call OpenAI o1-mini API
                    response = client.chat.completions.create(
                        model=llm_model,  # Specify the model
                        messages=conversation_history
                    )
                else:
                    # Call OpenAI GPT-4o API
                    response = client.chat.completions.create(
                        model=llm_model,  # Specify the model
                        messages=conversation_history,
                        temperature=temperature
                    )
                # log the input message
                logging.info(
                    "---------------------------------INPUT TO LLM FIRST-----------------------------------------")
                logging.info(conversation_history)
                # log the LLM response
                logging.info(
                    "------------------------------------LLM RESPONSE--------------------------------------------")
                logging.info(response)

                # Extract model response
                model_response = response.choices[0].message.content.strip()

                # Add the response to the conversation as input
                conversation_history.append({"role": "assistant", "content": model_response})
                logging.info(
                    "---------------------------------INPUT TO LLM UPDATE----------------------------------------")
                logging.info(conversation_history)

                # Extract token usage from the response
                total_tokens = response.usage.total_tokens  # Get the total tokens used for this request
                input_tokens = len(
                    original_prompt.split())  # Get the number of tokens used by the prompt (input tokens)
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
                    updated_prompt = f"[Feedback]: The previous generated code had the following error:\n{execution_feedback}\nPlease correct it."

                    # Add the refine prompt feedback to the conversation as input
                    conversation_history.append({"role": "user", "content": updated_prompt})

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


# use gpt-4o general model
# use o1-mini reasoning model
# the following from easy to difficult
# the following from instruction follows to reasoning
# call_api("gpt-4o", "prompts_both_instructions.json")

# call_api("gpt-4o", "prompts_instruction_1.json")
#
# call_api("gpt-4o", "prompts_instruction_2.json")
#
# call_api("gpt-4o", "prompts_no_instruction.json")

# call_api("gpt-4o", "prompts.json")
#
# call_api("o1-mini", "prompts_both_instructions.json")
#
# call_api("o1-mini", "prompts_instruction_1.json")
#
# call_api("o1-mini", "prompts_instruction_2.json")
#
call_api("o1-mini", "prompts_no_instruction.json")

# call_api("gpt-4o", "prompts.json")