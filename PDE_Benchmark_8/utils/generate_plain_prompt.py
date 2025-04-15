import json
import os

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
# Load original problem data
JSON_FILE = os.path.join(ROOT_DIR, 'prompt/PDE_TASK_QUESTION_ONLY.json')
# Prompts save pah
PROMPTS = os.path.join(ROOT_DIR, 'prompt/prompts.json')

with open(JSON_FILE, "r") as f:
    problems = json.load(f)

# Output file paths
output_files = {
    PROMPTS: {}
}


# Prompt generation function
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


# Fill in the prompt data
for name, data in problems.items():
    output_files[PROMPTS][name] = generate_prompt(data)

# Save each JSON file if it doesn't already exist
for filename, content in output_files.items():
    if os.path.exists(filename):
        print(f"Skipped: {filename} already exists.")
    else:
        with open(filename, "w") as f:
            json.dump(content, f, indent=2)
        print(f"Created: {filename}")
