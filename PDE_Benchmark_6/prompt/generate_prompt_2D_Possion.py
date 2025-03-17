import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "2D_Poisson_Equation": """
    You are given the **two-dimensional Poisson equation**, which introduces a source term to the Laplace equation:

    \\[
    \\frac{\\partial^2 p}{\\partial x^2} + \\frac{\\partial^2 p}{\\partial y^2} = b
    \\]

    Unlike the Laplace equation, the Poisson equation includes a finite source term that affects the solution, acting to "relax" the initial sources in the field.

    ### **Objective**
    Solve the **2D Poisson equation** numerically using an iterative finite-difference approach.

    ### **Numerical Method**
    - Discretize the equation using **second-order central differencing** for both spatial derivatives.
    - The discretized form of the equation is:

      \\[
      \\frac{p_{i+1,j}^{n} - 2p_{i,j}^{n} + p_{i-1,j}^{n}}{\\Delta x^2} + \\frac{p_{i,j+1}^{n} - 2p_{i,j}^{n} + p_{i,j-1}^{n}}{\\Delta y^2} = b_{i,j}^{n}
      \\]

    - Rearranging for \( p_{i,j}^{n} \), the update formula is:

      \\[
      p_{i,j}^{n} = \\frac{(p_{i+1,j}^{n} + p_{i-1,j}^{n}) \\Delta y^2 + (p_{i,j+1}^{n} + p_{i,j-1}^{n}) \\Delta x^2 - b_{i,j}^{n} \\Delta x^2 \\Delta y^2}{2(\\Delta x^2 + \\Delta y^2)}
      \\]

    ### **Initial Condition:**
    - Assume an **initial state** of \( p = 0 \) everywhere.

    ### **Boundary Conditions:**
    - **Dirichlet boundary conditions**:
      - \( p = 0 \) at \( x = 0, 2 \) and \( y = 0, 1 \).

    ### **Source Term (RHS \( b_{i,j} \)):**
    - The source term consists of **two initial spikes** inside the domain:
      - \( b_{i,j} = 100 \) at \( i = \\frac{1}{4} nx, j = \\frac{1}{4} ny \).
      - \( b_{i,j} = -100 \) at \( i = \\frac{3}{4} nx, j = \\frac{3}{4} ny \).
      - \( b_{i,j} = 0 \) elsewhere.

    ### **Computational Domain and Parameters:**
    - The equation is solved over a **rectangular grid** with:
      - Number of grid points in \( x \)-direction: \( nx = 50\)
      - Number of grid points in \( y \)-direction: \( ny = 50\)
      - Spatial step sizes: 
        - \( dx = \\frac{L_x}{nx - 1} \)
        - \( dy = \\frac{L_y}{ny - 1} \)

    ### **Tasks**
    1. Implement the **2D Poisson solver** using an iterative approach.
    2. Discretize the domain using a structured grid with uniform spacing.
    3. Apply the specified boundary and source conditions.
    4. Iterate until convergence is reached.
    5. Visualize the computed **pressure field** using contour plots.

    ### **Requirements**
    - Use **NumPy** for array computations.
    - Use **Matplotlib** for visualization.
    - Ensure **numerical stability** with appropriate stopping criteria.
    - Save the final pressure field \( p \) in `.npy` format.

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
