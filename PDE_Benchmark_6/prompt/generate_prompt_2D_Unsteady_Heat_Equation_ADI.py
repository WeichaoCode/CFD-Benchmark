import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "1D_Linear_Convection_ADI": """
    You are given the **one-dimensional linear convection equation** with diffusion effects:

    \\[
    \\frac{\\partial u}{\\partial t} + c \\frac{\\partial u}{\\partial x} = \\epsilon \\frac{\\partial^2 u}{\\partial x^2}
    \\]

    where:
    - \\( u(x,t) \\) represents the wave amplitude,
    - \\( c \\) is the convection speed,
    - \\( \\epsilon \\) is a damping coefficient.

    ### **Objective**
    Solve the equation numerically using the **Alternating Direction Implicit (ADI) Method**.

    ### **Numerical Method**
    - ADI splits the solution into two alternating time steps where half of the equation is solved implicitly.
    - First, solve for an intermediate state using an implicit scheme in the x-direction.
    - Then, solve for the next time step using an implicit scheme in the other direction.
    - This method allows for stable solutions with larger time steps.

    ### **Computational Domain and Parameters**
    - The equation is solved in a **periodic domain**: \\( x \\in (-5,5) \\).
    - The initial condition is given as:

      \\[
      u_0 = e^{-x^2}
      \\]

    - Two cases are considered:
      - **Undamped case:** \\( \\epsilon = 0 \\)
      - **Damped case:** \\( \\epsilon = 5 \\times 10^{-4} \\)
    - Convection speed: \\( c = 1 \\).

    ### **Tasks**
    1. Implement the **ADI Method** to solve the **1D linear convection equation**.
    2. Use a structured grid with uniform spacing.
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
