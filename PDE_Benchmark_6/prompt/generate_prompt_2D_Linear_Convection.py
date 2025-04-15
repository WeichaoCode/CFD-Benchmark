import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "2D_Linear_Convection": """
    You are given the **two-dimensional linear convection equation**, which governs the transport of a scalar field in two spatial dimensions:

    \\[
    \\frac{\\partial u}{\\partial t} + c \\frac{\\partial u}{\\partial x} + c \\frac{\\partial u}{\\partial y} = 0
    \\]

    where:
    - \( u(x, y, t) \) is the scalar quantity being transported.
    - \( c \) is the convection speed in both \( x \) and \( y \) directions.

    ### **Objective**
    Solve this equation numerically using the **Finite Difference Method (FDM)**.

    ### **Numerical Method**
    - Use **forward differencing** for the time derivative.
    - Use **backward differencing** for the spatial derivatives.
    - The numerical scheme is given by:

      \\[
      \\frac{u_{i,j}^{n+1} - u_{i,j}^{n}}{\\Delta t} + c \\frac{u_{i,j}^{n} - u_{i-1,j}^{n}}{\\Delta x} + c \\frac{u_{i,j}^{n} - u_{i,j-1}^{n}}{\\Delta y} = 0
      \\]

      Rearranging for \( u_{i,j}^{n+1} \):

      \\[
      u_{i,j}^{n+1} = u_{i,j}^{n} - \\frac{\\Delta t}{\\Delta x} c (u_{i,j}^{n} - u_{i-1,j}^{n}) - \\frac{\\Delta t}{\\Delta y} c (u_{i,j}^{n} - u_{i,j-1}^{n})
      \\]

    ### **Initial Condition**
    The initial condition is defined using a **hat function**, where:
    - \( u(x, y) = 2 \) for \( 0.5 \leq x \leq 1 \) and \( 0.5 \leq y \leq 1 \).
    - \( u(x, y) = 1 \) everywhere else.

    ### **Boundary Conditions**
    The value of \( u(x, y) \) is set to **1** on all boundaries:
    - \( u = 1 \) for \( x = 0, 2 \).
    - \( u = 1 \) for \( y = 0, 2 \).

    ### **Computational Domain and Parameters**
    - The equation is solved over a **square domain**:
      - \( x \in [0, 2] \), \( y \in [0, 2] \).
    - The **grid resolution** is:
      - Number of grid points in \( x \)-direction: \( nx = 81 \).
      - Number of grid points in \( y \)-direction: \( ny = 81 \).
      - Spatial step sizes:
        - \( dx = \\frac{L_x}{nx - 1} \).
        - \( dy = \\frac{L_y}{ny - 1} \).
    - **Time-stepping parameters**:
      - Number of time steps: \( nt = 100\).
      - Stability parameter: \( \sigma = 0.2 \).
      - Time step:  
        \[
        dt = \sigma \cdot \\frac{\\min(dx, dy)}{c}
        \]

    ### **Tasks**
    1. Implement the **Finite Difference Method** for solving the **2D Linear Convection Equation**.
    2. Use a structured grid with uniform spacing.
    3. Simulate the **propagation of the wave** over time.
    4. Visualize the computed **solution field** using contour plots.

    ### **Requirements**
    - Use **NumPy** for numerical computations.
    - Use **Matplotlib** for visualization.
    - Ensure **numerical stability** by choosing an appropriate time step based on the CFL condition.
    - Save the computed solution \( u(x, y) \) in `.npy` format.

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
