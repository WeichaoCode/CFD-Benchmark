import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "2D_Laplace_Equation": """
    You are given the **two-dimensional Laplace equation**, which describes steady-state diffusion phenomena and is used to model equilibrium solutions for various physical systems:

    \\[
    \\frac{\\partial^2 p}{\\partial x^2} + \\frac{\\partial^2 p}{\\partial y^2} = 0
    \\]

    ### **Objective**
    Solve this equation numerically using the **Finite Difference Method (FDM)**.

    ### **Numerical Method**
    - Use **central differencing** for the second derivatives in both spatial directions.
    - The discretized form of the equation using a structured grid is:

      \\[
      \\frac{p_{i+1,j} - 2p_{i,j} + p_{i-1,j}}{\Delta x^2} + \\frac{p_{i,j+1} - 2p_{i,j} + p_{i,j-1}}{\Delta y^2} = 0
      \\]

      Rearranging for \( p_{i,j} \):

      \\[
      p_{i,j} = \\frac{\\Delta y^2(p_{i+1,j} + p_{i-1,j}) + \\Delta x^2(p_{i,j+1} + p_{i,j-1})}{2(\\Delta x^2 + \\Delta y^2)}
      \\]

      This discretization is known as the **five-point difference operator**, which is iteratively solved until the solution converges.

    ### **Initial and Boundary Conditions**
    - **Initial Condition:** Assume \( p = 0 \) everywhere in the computational domain.
    - **Boundary Conditions:**
      - **Left boundary (\( x = 0 \))**: \( p = 0 \)
      - **Right boundary (\( x = 2 \))**: \( p = y \)
      - **Top and Bottom boundaries (\( y = 0, 1 \))**: Neumann boundary condition: \( \\frac{\\partial p}{\\partial y} = 0 \)

    ### **Computational Domain and Parameters**
    - Solve over a **rectangular domain** where:
      - \( x \in [0, 2] \), \( y \in [0, 1] \)
    - Grid resolution:
      - Number of grid points in \( x \)-direction: \( nx = 31\)
      - Number of grid points in \( y \)-direction: \( ny = 31\)
      - Spatial step sizes:
        - \( dx = \\frac{L_x}{nx - 1} \)
        - \( dy = \\frac{L_y}{ny - 1} \)

    ### **Tasks**
    1. Implement an **iterative finite difference solver** to compute the equilibrium solution of the 2D Laplace equation.
    2. Use a **structured Cartesian grid** with uniform spacing.
    3. Iterate until the solution converges, ensuring that the maximum change between iterations is below a predefined tolerance.
    4. Visualize the computed solution using a **contour plot**.

    ### **Requirements**
    - Use **NumPy** for numerical computations.
    - Use **Matplotlib** to generate contour plots of the solution.
    - Implement iterative solvers such as **Jacobi iteration, Gauss-Seidel, or Successive Over-Relaxation (SOR)**.
    - Ensure convergence and stability of the numerical scheme.
    - Save the final p(x,y) in `.npy` format.
    - Compare the numerical solution with the analytical solution:

      \\[
      p(x, y) = \\frac{x}{4} - \\frac{4}{\pi} \\sum_{n=1, odd}^{\infty} \\frac{1}{(n \pi)^2} \sinh(2 n \pi) \cos(n \pi y)
      \\]

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
