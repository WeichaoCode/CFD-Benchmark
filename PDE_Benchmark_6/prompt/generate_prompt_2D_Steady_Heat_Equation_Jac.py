import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "2D_Steady_Heat_Equation_Jac": """
    You are given the **steady two-dimensional heat equation**, which models heat distribution over a rectangular domain:

    \\[
    \\frac{\\partial^2 T}{\\partial x^2} + \\frac{\\partial^2 T}{\\partial y^2} = 0
    \\]

    ### **Computational Domain**
    The equation is solved in a **rectangular domain** with spatial coordinates:
    - \( x \in [0, 5] \), \( y \in [0, 4] \)
    
    The domain is discretized using **finite difference methods** with a structured uniform grid:
    - Grid spacing in the \( x \)-direction: \( \Delta x = 0.05 \)
    - Grid spacing in the \( y \)-direction: \( \Delta y = 0.05 \)
    
    The number of grid points in each direction is computed as:
    - Number of grid points in the \( x \)-direction: \( n_x = \frac{5.0}{0.05} + 1 = 101 \)
    - Number of grid points in the \( y \)-direction: \( n_y = \frac{4.0}{0.05} + 1 = 81 \)
    
    Thus, the total grid consists of a uniform grid of **101 points** in the \( x \)-direction and **81 points** in the \( y \)-direction.

    ### **Boundary Conditions**
    The temperature values along the boundaries of the domain are fixed:
    - **Left boundary (AB)**: \( T = 10^\circ C \)
    - **Top boundary (CD)**: \( T = 0^\circ C \)
    - **Right boundary (EF)**: \( T = 40^\circ C \)
    - **Bottom boundary (G)**: \( T = 20^\circ C \)

    ### **Numerical Methods**
    Solve the steady-state heat equation usingthe following iterative methods:

     Jacobi Method**  
    - Update the temperature at each grid point using the average of its four neighboring points:
      \\[
      T_{i,j}^{k+1} = \\frac{T_{i+1,j}^{k} + T_{i-1,j}^{k} + \\beta^2 (T_{i,j+1}^{k} + T_{i,j-1}^{k})}{2(1 + \\beta^2)}
      \\]
      where \( \\beta \) is the grid aspect ratio \( \\beta = \\frac{\\Delta x}{\\Delta y} \).

    ### **Tasks**
    1. Implement the selected iterative method (Jacobi, Gauss-Seidel, or SOR) for solving the **2D steady heat equation**.
    2. Discretize the computational domain using a uniform structured grid.
    3. Apply the given **Dirichlet boundary conditions**.
    4. Iterate until convergence, using a stopping criterion based on **residual reduction**.
    5. Visualize the **steady-state temperature distribution** as a contour plot.

    ### **Requirements**
    - Use **NumPy** for matrix computations.
    - Use **Matplotlib** to generate contour plots of the temperature field.
    - Save the final computed temperature field \( T(x, y) \) as a `.npy` file.

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
