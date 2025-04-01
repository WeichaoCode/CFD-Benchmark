import json
import os

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
# Load original problem data
JSON_FILE = os.path.join(ROOT_DIR, 'prompt/PDE_TASK_CONFIGURATION.json')
# Prompts save pah
PROMPTS_NO_INSTRUCTION = os.path.join(ROOT_DIR, 'prompt/prompts_no_instruction.json')
PROMPTS_INSTRUCTION_1 = os.path.join(ROOT_DIR, 'prompt/prompts_instruction_1.json')
PROMPTS_INSTRUCTION_2 = os.path.join(ROOT_DIR, 'prompt/prompts_instruction_2.json')
PROMPTS_BOTH_INSTRUCTION = os.path.join(ROOT_DIR, 'prompt/prompts_both_instructions.json')

with open(JSON_FILE, "r") as f:
    problems = json.load(f)

# Output file paths
output_files = {
    PROMPTS_NO_INSTRUCTION: {},
    PROMPTS_INSTRUCTION_1: {},
    PROMPTS_INSTRUCTION_2: {},
    PROMPTS_BOTH_INSTRUCTION: {}
}


# Prompt generation function
def generate_prompt(data, include_1=False, include_2=False):
    parts = [
        "You are given the following partial differential equation (PDE) problem:\n",
        "**Equation:**\n" + data.get("equation", "") + "\n",
        "**Boundary Conditions:**\n" + data.get("boundary conditions", "") + "\n",
        "**Initial Conditions:**\n" + data.get("initial conditions", "") + "\n",
        "**Domain:**\n" + data.get("domain", "") + "\n",
        "**Numerical Method:**\n" + data.get("numerical method", "") + "\n"
    ]
    # Expert instructions (optional)
    if include_1:
        parts.append("### Parameter Selection Guidance:\n" +
                     "Use the following expert guidance to choose appropriate numerical parameters such as "
                     "spatial/temporal resolution:\n" +
                     data.get("expert instruction group 1", "") + "\n")

    if include_2:
        parts.append("### Code Development Guidance:\n" +
                     "Follow this expert-provided step-by-step instruction to implement the solver in Python:\n" +
                     data.get("expert instruction group 2", "") + "\n")

    # Check for 'save_values' and add to task description
    save_values = data.get("save_values", [])
    save_values_str = ", ".join(save_values) if save_values else "the relevant variables specified for the problem"
    # Always end with task specification for the code
    parts.append(
        "### Task:\n"
        "- Write Python code to numerically solve the above CFD problem using the specified numerical method.\n"
        "- If the problem is **unsteady**, only compute and save the **solution at the final time step**.\n"
        "- Save the final solution for each specified variable as a separate `.npy` file using NumPy.\n"
        "- For **1D problems**, save a 1D NumPy array for each variable. For **2D problems**, save a 2D array.\n"
        "- The output `.npy` files **must contain only the final solution** (not intermediate steps).\n"
        "- ✅ **IMPORTANT**: You **must save each variable using exactly the same name as listed in `save_values`.**\n"
        "- ✅ Example: if `save_values = ['u', 'p']`, then save files as `u.npy` and `p.npy`, and use variable names "
        "`u` and `p` in your code.\n"
        "- Do not include extra print statements or explanations — return only the complete, runnable Python code."
    )

    return "\n".join(parts)


# Fill in the prompt data
for name, data in problems.items():
    output_files[PROMPTS_NO_INSTRUCTION][name] = generate_prompt(data)
    output_files[PROMPTS_INSTRUCTION_1][name] = generate_prompt(data, include_1=True)
    output_files[PROMPTS_INSTRUCTION_2][name] = generate_prompt(data, include_2=True)
    output_files[PROMPTS_BOTH_INSTRUCTION][name] = generate_prompt(data, include_1=True, include_2=True)

# Save each JSON file if it doesn't already exist
for filename, content in output_files.items():
    if os.path.exists(filename):
        print(f"Skipped: {filename} already exists.")
    else:
        with open(filename, "w") as f:
            json.dump(content, f, indent=2)
        print(f"Created: {filename}")
