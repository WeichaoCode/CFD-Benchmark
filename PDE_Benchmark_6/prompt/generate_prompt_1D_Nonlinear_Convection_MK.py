import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "1D_Nonlinear_Convection_MK": """
    You are given the **one-dimensional nonlinear convection equation**, a fundamental PDE that models **advection**:

    \\[
    \\frac{\\partial u}{\\partial t} + u \\frac{\\partial u}{\\partial x} = 0
    \\]

    where:
    - \\( u(x,t) \\) represents the wave amplitude,
    - \\( x \\) is the spatial coordinate,
    - \\( t \\) is time.

    ### **Computational Domain**
    - Solve the equation in a periodic domain \\( x \\in [0, 2\\pi] \\)
    - The initial condition is given by:
      \\[
      u(x,0) = \\sin(x) + 0.5 \\sin(0.5x)
      \\]
    - The domain has **periodic boundary conditions**.

    ### Choose parameters
    - CFL number: `nu = 0.5`
    - Time step: `dt = 0.01`
    - Maximum number of time steps: `T = 500`
    - Space step: `dx = dt / nu`
    
    ### **Numerical Method**
    - Use the **MacCormack Method**, a predictor-corrector scheme:
      - **Predictor Step**:
        \\[
        u^*_j = u^n_j - \\frac{\\Delta t}{\\Delta x} (F^n_{j+1} - F^n_j)
        \\]
      - **Corrector Step**:
        \\[
        u^{n+1}_j = \\frac{1}{2} \\left[ u^n_j + u^*_j - \\frac{\\Delta t}{\\Delta x} (F^*_j - F^*_{j-1}) \\right]
        \\]
    - Choose an appropriate time step \\( \\Delta t = 0.1 \\) to ensure numerical stability.

    ### **Tasks**
    1. Implement the MacCormack method to update \\( u(x,t) \\).
    2. Apply periodic boundary conditions.
    3. Simulate wave propagation over time.
    4. Compare the solution with those from the Lax and Lax-Wendroff methods.

    ### **Requirements**
    - Use NumPy for numerical operations.
    - Use Matplotlib to visualize the results.
    - Save the computed solution in a `.npy` file.

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

print(f"1D_Nonlinear_Convection_MacCormack' equation prompt added to {json_file}")
