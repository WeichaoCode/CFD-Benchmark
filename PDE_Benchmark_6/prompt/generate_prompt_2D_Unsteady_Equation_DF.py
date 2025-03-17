import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "1D_Linear_Convection_Dufort_Frankel": """
    You are given the **one-dimensional linear convection equation** with diffusion effects:

    \\[
    \\frac{\\partial u}{\\partial t} + c \\frac{\\partial u}{\\partial x} = \\epsilon \\frac{\\partial^2 u}{\\partial x^2}
    \\]

    ### **Objective**
    Solve the equation numerically using the **Dufort-Frankel Method**.

    ### **Numerical Method**
    - The **Dufort-Frankel method** is an explicit time integration scheme that improves stability by using values from two previous time steps.
    - It avoids the need for implicit solvers while allowing for larger time steps than explicit schemes.
    - The scheme is given by:

      \\[
      u^{n+1}_i = \\frac{ (1 - 2r) u^{n-1}_i + 2r (u^n_{i+1} + u^n_{i-1}) }{1 + 2r}
      \\]

      where \\( r = \\frac{ \\epsilon \\Delta t}{\\Delta x^2} \\).

    ### **Computational Domain and Parameters**
    - **Domain**: \\( x \\in (-5,5) \\).
    - **Initial Condition**: \\( u_0 = e^{-x^2} \\).
    - **Boundary Conditions**: Periodic boundaries.
    - **Convection Speed**: \\( c = 1 \\).
    - **Diffusion Cases**:
      - **Undamped:** \\( \\epsilon = 0 \\)
      - **Damped:** \\( \\epsilon = 5 \\times 10^{-4} \\).

    ### **Tasks**
    1. Implement the **Dufort-Frankel method** for solving the **1D linear convection equation**.
    2. Ensure numerical stability by setting an appropriate time step.
    3. Apply periodic boundary conditions.
    4. Visualize the solution over time.

    ### **Requirements**
    - Use **NumPy** for array computations.
    - Use **Matplotlib** for visualization.
    - Save the computed solution in `.npy` format.

    **Return only the Python code that implements this solution.**
    """
}


# Define the JSON file path
json_file = SAVE_FILE

# Load existing prompts if the file exists
if os.path.exists(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)
else:
    prompts = {}  # Create an empty dictionary if file does not exist

# Append the new prompt
prompts.update(prompt_text)

# Save the updated JSON file
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(prompts, f, indent=4, ensure_ascii=False)

print(f"1D Diffusion' equation prompt added to {json_file}")
