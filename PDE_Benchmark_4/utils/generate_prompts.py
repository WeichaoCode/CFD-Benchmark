import json
import os

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
TASKS_FILE = os.path.join(ROOT_DIR, "data", "PDE_TASK_POOL.json")
PROMPTS_FILE = os.path.join(ROOT_DIR, "prompts", "PDE_TASK_PROMPT.json")

# Load PDE task pool JSON
with open(TASKS_FILE, "r") as file:
    pde_tasks = json.load(file)


def generate_prompt_function(pde_name, pde_info):
    prompt = f"""
    Write a Python function to solve the given PDE: **{pde_name.replace('_', ' ')}**.

    ### **Requirements**
    - Use the **Finite Difference Method (FDM)** for discretization.
    - Implement a **Manufactured Solution (MMS)** for validation.
    - Compute the **Mean Squared Error (MSE)** to compare the numerical solution with the exact MMS.
    - Plot both the **numerical solution** and the **exact MMS solution** in a 2D figure for comparison.
    - Ensure the code is **numerically stable**, and enforce appropriate **CFL conditions** if needed.
    - The function should be **modular**, allowing users to pass in problem-specific parameters.
    - The function must return the computed numerical solution.

    ### **Function Template**
    ```python
    def solve_{pde_name.lower()}(nx, ny, nt, dx, dy, dt, *params):
        \"\"\"
        Solve the {pde_name.replace('_', ' ')} using the Finite Difference Method (FDM).

        Parameters:
            nx, ny, nt: Grid points in x, y, and time dimensions.
            dx, dy, dt: Grid spacing and time step.
            *params: Additional problem-specific parameters.

        Returns:
            numpy.ndarray: Computed numerical solution.
        \"\"\"
        # Step 1: Define the grid and initialize solution variables
        # Step 2: Compute source term from MMS
        # Step 3: Apply initial and boundary conditions
        # Step 4: Solve the PDE iteratively using FDM
        # Step 5: Compute the exact MMS solution for comparison
        # Step 6: Calculate Mean Squared Error (MSE)
        # Step 7: Plot numerical vs exact solutions

        return solution
        """

    return prompt


# Generate prompts for all PDEs
pde_prompts = {"prompts": {}}
for task_name, task_info in pde_tasks["tasks"].items():
    pde_prompts["prompts"][task_name] = generate_prompt_function(task_name, task_info)

# Save the generated prompts to a new JSON file
with open(PROMPTS_FILE, "w") as file:
    json.dump(pde_prompts, file, indent=4)

print(f"✅ LLM prompts generated and saved to {PROMPTS_FILE}")
