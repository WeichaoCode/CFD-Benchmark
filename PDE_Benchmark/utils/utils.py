import os
import json
import re
import subprocess
import logging
from datetime import datetime
import boto3
from openai import OpenAI


def generate_prompt(data):
    parts = [
        "You are given the following partial differential equation (PDE) problem:\n",
        "**Equation:**\n" + data.get("equation", "") + "\n",
        "**Boundary Conditions:**\n" + data.get("boundary conditions", "") + "\n",
        "**Initial Conditions:**\n" + data.get("initial conditions", "") + "\n",
        "**Domain:**\n" + data.get("domain", "") + "\n"
    ]

    # Check for 'save_values' and add to task description
    save_values = data.get("save_values", [])
    save_values_str = ", ".join(save_values) if save_values else "the relevant variables specified for the problem"
    # Always end with task specification for the code
    parts.append(
        "### Task:\n"
        "- Write Python code to numerically solve the given CFD problem. Choose an appropriate numerical method based "
        "on the problem characteristics.\n"
        "- If the problem is **unsteady**, only compute and save the **solution at the final time step**.\n"
        "- For each specified variable, save the final solution as a separate `.npy` file using NumPy:\n"
        "  - For **1D problems**, save each variable as a 1D NumPy array.\n"
        "  - For **2D problems**, save each variable as a 2D NumPy array.\n"
        "- The `.npy` files should contain only the final solution field (not intermediate steps) for each of the "
        "specified variables.\n"
        "- **Save the following variables** at the final time step:\n"
        + save_values_str + "\n"
                            "(Each variable should be saved separately in its own `.npy` file, using the same name as "
                            "provided in `save_values`).\n"
                            "- Ensure the generated code properly handles the solution for each specified variable "
                            "and saves it correctly in `.npy` format.\n"
                            "- **Return only the complete, runnable Python code** that implements the above tasks, "
                            "ensuring no extra explanations or information is included."
    )

    return "\n".join(parts)


class PromptGenerator:
    def __init__(self, root_dir):
        self.problems = None
        self.root_dir = root_dir
        self.input_json = os.path.join(root_dir, 'prompt/PDE_TASK_QUESTION_ONLY.json')
        self.output_json = os.path.join(root_dir, 'prompt/prompts.json')

    def load_problem_data(self):
        with open(self.input_json, "r") as f:
            self.problems = json.load(f)

    def create_prompts(self):
        output_data = {}
        for name, data in self.problems.items():
            output_data[name] = generate_prompt(data)
        return output_data

    def save_prompts(self, prompts):
        if os.path.exists(self.output_json):
            print(f"Skipped: {self.output_json} already exists.")
        else:
            with open(self.output_json, "w") as f:
                json.dump(prompts, f, indent=2)
            print(f"Created: {self.output_json}")


# === Function to Execute Python Script and Capture Errors and Warnings ===
def execute_python_script(filepath):
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


def build_conversation(original_prompt, llm_model):
    # Shared instruction
    instruction = ("You are a highly skilled assistant capable of generating Python code to solve CFD problems "
                   "using appropriate numerical methods."
                   "Given the problem description, you should reason through the problem and determine the best "
                   "approach for discretizing and solving it,"
                   "while respecting the specified boundary conditions, initial conditions, and domain.\n"
                   "For unsteady problems, save only the solution at the final time step. For 1D problems, "
                   "save a 1D array; for 2D problems, save a 2D array.\n"
                   "Ensure the code follows the user's specifications and saves the requested variables exactly "
                   "as named in `save_values`.\n"
                   "Your task is to generate the correct, fully runnable Python code for solving the problem "
                   "without additional explanations.")

    # Prompt augmentation
    user_prompt = (original_prompt +
                   "If it is an unsteady problem, only save the solution at the final time step "
                   "If the problem is 1D, the saved array should be 1D. "
                   "If the problem is 2D, the saved array should be 2D.")
    # Determine system or user role for the instruction
    if llm_model in ["o1-mini", "sonnet-35", "haiku"]:
        conversation_history = [
            {"role": "user", "content": instruction},
            {"role": "user", "content": user_prompt}
        ]
    elif llm_model == "gpt-4o":
        conversation_history = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_prompt}
        ]
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")

    return conversation_history


def extract_model_response(llm_model, response):
    # Extract model response
    if llm_model in ["gpt-4o", "o1-mini"]:
        model_response = response.choices[0].message.content.strip()
    elif llm_model in ["sonnet-35", "haiku"]:
        # Parse the response
        response_body = json.loads(response["body"].read().decode("utf-8"))

        # Extract model response
        if "content" in response_body:
            model_response = response_body["content"][0]["text"]
        else:
            model_response = "Error: No response received from model."
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")

    return model_response


def update_token_usage(llm_model, original_prompt, response):
    if llm_model in ["gpt-4o", "o1-mini"]:
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
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")


def save_model_outputs(task_name, output_folder, model_response):
    # Extract Python code using regex
    code_match = re.findall(r"```python(.*?)```", model_response, re.DOTALL)
    extracted_code = code_match[0].strip() if code_match else "# No valid Python code extracted"

    # Save the full model response
    response_file = os.path.join(output_folder, f"{task_name}.txt")
    with open(response_file, "w") as txt_file:
        txt_file.write(model_response)

    # Save the extracted Python code
    script_path = os.path.join(output_folder, f"{task_name}.py")
    with open(script_path, "w") as py_file:
        py_file.write(extracted_code)

    print(f"✅ Code saved: {script_path}")

    return script_path


def execute_check_errors(script_path, task_name, conversation_history):
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


def call_llm_api(llm_model, client, conversation_history, temperature, bedrock_runtime, inference_profile_arn):
    if llm_model == "o1-mini":
        # Call OpenAI o1-mini API
        response = client.chat.completions.create(
            model=llm_model,  # Specify the model
            messages=conversation_history
        )
    elif llm_model == "gpt-4o":
        # Call OpenAI GPT-4o API
        response = client.chat.completions.create(
            model=llm_model,  # Specify the model
            messages=conversation_history,
            temperature=temperature
        )
    elif llm_model in ["sonnet-35", "haiku"]:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8000,
            "temperature": 0.7,
            "messages": conversation_history
        }

        # Invoke AWS Bedrock API
        response = bedrock_runtime.invoke_model(
            modelId=inference_profile_arn,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")
    return response


def generate_code(llm_model, task_name, prompt, client, temperature, bedrock_runtime, inference_profile_arn,
                  output_folder, max_retries=5):
    retries = 0
    original_prompt = prompt  # Keep the original prompt unchanged
    # Initialize an empty list to store the conversation history
    conversation_history = build_conversation(original_prompt, llm_model)

    while retries < max_retries:
        print(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
        logging.info(f"🔹 Generating code for: {task_name} (Attempt {retries + 1}/{max_retries})")
        try:
            response = call_llm_api(llm_model, client, conversation_history, temperature, bedrock_runtime,
                                    inference_profile_arn)
            # log the input message
            logging.info(
                "---------------------------------INPUT TO LLM FIRST-----------------------------------------")
            logging.info(conversation_history)
            # log the LLM response
            logging.info(
                "------------------------------------LLM RESPONSE--------------------------------------------")
            logging.info(response)

            # Extract model response
            model_response = extract_model_response(llm_model, response)
            # Add the response to the conversation as input
            conversation_history.append({"role": "assistant", "content": model_response})
            logging.info(
                "---------------------------------INPUT TO LLM UPDATE----------------------------------------")
            logging.info(conversation_history)

            # tracking the token usage and cost
            update_token_usage(llm_model, original_prompt, response)

            # Extract Python code using regex, save the full model response and extracted python code
            script_path = save_model_outputs(task_name, output_folder, model_response)

            # Execute and check for errors
            execute_check_errors(script_path, task_name, conversation_history)

            retries += 1

        except Exception as e:
            print(f"❌ API Call Error for {task_name}: {str(e)}")
            logging.info(f"❌ API Call Error for {task_name}: {str(e)}")
            return  # Stop retrying if API call fails

    print(f"⚠️ Max retries reached for {task_name}. Check logs for remaining errors.")
    logging.info(f"⚠️ Max retries reached for {task_name}. Check logs for remaining errors.")


def api_key_configuration(llm_model):
    # === OpenAI API Configuration ===
    api_key = "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"  # Replace this with your OpenAI API key
    client = OpenAI(api_key=api_key)

    if llm_model == "sonnet-35":
        # Define Sonnet-3.5 profile Inference Profile ARN
        inference_profile_arn = "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/g47vfd2xvs5w"
    elif llm_model == "haiku":
        # Define Haiku profile Inference Profile ARN
        inference_profile_arn = "arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/g47vfd2xvs5w"
    else:
        raise ValueError(f"Unsupported model type: {llm_model}")

    return api_key, client, inference_profile_arn


class LLMCodeGenerator:
    def __init__(self, llm_model, prompt_json, temperature=0.0):
        self.llm_model = llm_model
        self.prompt_json = prompt_json
        self.temperature = temperature
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0  # Optional: for tracking cost
        # Initialize AWS Bedrock client
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-west-2")
        # Get current time with microseconds
        self.timestamp = datetime.now().strftime("%H-%M-%S-%f")  # %f gives microseconds
        # === Paths ===
        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
        self.PROMPTS_FILE = os.path.join(self.ROOT_DIR, "prompt", prompt_json)
        self.OUTPUT_FOLDER = os.path.join(self.ROOT_DIR, f"solver/{llm_model}/{prompt_json}")
        self.LOG_FILE = os.path.join(self.ROOT_DIR, f"report/{llm_model}_{prompt_json}_{self.timestamp}.log")
        # Ensure the output directory exists
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)

        logging.basicConfig(
            filename=self.LOG_FILE,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def call_api(self):
        logging.info(
            "####################################################################################################")
        logging.info(f"Using the {self.llm_model}, change temperature to {self.temperature}, "
                     f"use the prompt {self.prompt_json}")
        # === Get API credentials, client, and profile ===
        api_key, client, inference_profile_arn = api_key_configuration(self.llm_model)
        # Load prompts
        with open(self.PROMPTS_FILE, "r") as file:
            pde_prompts = json.load(file)

        # Loop through prompts and generate code
        for task_name, prompt in pde_prompts["prompts"].items():
            generate_code(
                self.llm_model,
                task_name,
                prompt,
                client,
                self.temperature,
                self.bedrock_runtime,
                inference_profile_arn,
                self.OUTPUT_FOLDER,
                max_retries=5
            )

        print("\n🎯 Execution completed. Check the solver directory for generated files.")
        logging.info("\n🎯 Execution completed. Check the solver directory for generated files.")
        logging.info(f"Total Input Tokens: {self.total_input_tokens}")
        logging.info(f"Total Output Tokens: {self.total_output_tokens}")
        logging.info(f"Total Estimated Cost: ${self.total_cost:.6f}")


def replacer_factory(base_name, save_dir):
    def replacer(match):
        var_name = match.group(3)  # u, v, p, etc.
        new_filename = f"{var_name}_{base_name}.npy"
        new_path = os.path.join(save_dir, new_filename).replace("\\", "/")  # Use Unix-style paths
        return f"np.save('{new_path}', {var_name})"

    return replacer


def call_post_process(llm_model, prompt_json):
    # === Paths ===
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
    folder_path = os.path.join(ROOT_DIR, f'solver/{llm_model}/{prompt_json}')
    save_dir = os.path.join(ROOT_DIR, f'results/prediction/{llm_model}/{prompt_json}')
    os.makedirs(save_dir, exist_ok=True)

    pattern = r"np\.save\((['\"])(.+?\.npy)\1\s*,\s*(\w+)\s*\)"

    for filename in os.listdir(folder_path):
        if filename.endswith(".py"):
            py_path = os.path.join(folder_path, filename)
            base_name = os.path.splitext(filename)[0]  # e.g., "burgers_solver"

            with open(py_path, "r") as f:
                code = f.read()

            # Create a replacer for this specific file
            replacer = replacer_factory(base_name, save_dir)

            # Apply the replacement
            new_code, count = re.subn(pattern, replacer, code)

            if count > 0:
                with open(py_path, "w") as f:
                    f.write(new_code)
                print(f"✅ Updated {filename}: replaced {count} save path(s)")
            else:
                print(f"ℹ️ No np.save calls updated in {filename}")


