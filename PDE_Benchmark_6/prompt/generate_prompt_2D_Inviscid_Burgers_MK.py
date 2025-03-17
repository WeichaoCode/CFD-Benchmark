import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "2D_Inviscid_Burgers_MK": """
    You are given the **two-dimensional inviscid Burgers' equation**, which governs nonlinear convection of a velocity field in two spatial dimensions:
    
    \\[
    \\frac{\\partial \\mathbf{U}}{\\partial t} + \\mathbf{U} \\frac{\\partial \\mathbf{U}}{\\partial x} = 0
    \\]
    
    where:
    
    \\[
    \\mathbf{U} = \\begin{bmatrix} u \\\\ v \\end{bmatrix}, \\quad
    \\mathbf{x} = \\begin{bmatrix} x \\\\ y \\end{bmatrix}
    \\]
    
    This system consists of two coupled nonlinear PDEs:
    
    \\[
    \\frac{\\partial u}{\\partial t} + u \\frac{\\partial u}{\\partial x} + v \\frac{\\partial u}{\\partial y} = 0
    \\]
    \\[
    \\frac{\\partial v}{\\partial t} + u \\frac{\\partial v}{\\partial x} + v \\frac{\\partial v}{\\partial y} = 0
    \\]
    
    ### **Objective**
    Solve this equation numerically using the **First-Order Upwind Method**.
    
    ### **Numerical Method**
    - Use **forward differencing** for the time derivative.
    - Use **backward differencing** for the spatial derivatives.
    - The numerical scheme is given by:
    
      \\[
      \\frac{u_{i,j}^{n+1} - u_{i,j}^{n}}{\\Delta t} + u_{i,j}^{n} \\frac{u_{i,j}^{n} - u_{i-1,j}^{n}}{\\Delta x} + v_{i,j}^{n} \\frac{u_{i,j}^{n} - u_{i,j-1}^{n}}{\\Delta y} = 0
      \\]
    
      \\[
      \\frac{v_{i,j}^{n+1} - v_{i,j}^{n}}{\\Delta t} + u_{i,j}^{n} \\frac{v_{i,j}^{n} - v_{i-1,j}^{n}}{\\Delta x} + v_{i,j}^{n} \\frac{v_{i,j}^{n} - v_{i,j-1}^{n}}{\\Delta y} = 0
      \\]
    ### **Initial Condition:**
    The initial condition is defined using a **hat function**, where the velocity components \( u(x,y) \) and \( v(x,y) \) are initialized to **1** everywhere in the domain, 
    except in the region \( 0.5 \leq x \leq 1 \) and \( 0.5 \leq y \leq 1 \), where they are set to **2**.
    
    ### **Boundary Conditions:**
    The velocity components \( u(x,y) \) and \( v(x,y) \) are set to **1** on all boundaries of the domain. 
    This represents **Dirichlet boundary conditions**, ensuring that the velocity remains fixed at 1 along the edges:
    - **Top boundary** (\( y = \max \)): \( u = 1 \), \( v = 1 \)
    - **Bottom boundary** (\( y = \min \)): \( u = 1 \), \( v = 1 \)
    - **Left boundary** (\( x = \min \)): \( u = 1 \), \( v = 1 \)
    - **Right boundary** (\( x = \max \)): \( u = 1 \), \( v = 1 \)
    
    ### **Computational Domain and Parameters:**
    - The equation is solved over a **square domain** with spatial extent:
      - \( x \in [0, 2] \), \( y \in [0, 2] \)
    - The **grid resolution** is:
      - Number of grid points in \( x \)-direction: \( nx = 151 \)
      - Number of grid points in \( y \)-direction: \( ny = 151 \)
      - Spatial step sizes: 
        - \( dx = \frac{L_x}{nx - 1} \)
        - \( dy = \frac{L_y}{ny - 1} \)
    - **Time-stepping parameters:**
      - Number of time steps: \( nt = 300 \)
      - Stability parameter: \( \sigma = 0.2 \)
      - Time step:  
        \[
        dt = \sigma \cdot \frac{\min(dx, dy)}{2}
        \]
    
    
    ### **Tasks**
    1. Implement the F**MacCormack Method** for solving the **2D Inviscid Burgers' equation**.
    2. Use a structured grid with uniform spacing.
    4. Simulate the velocity field evolution over time.
    5. Visualize the computed **velocity field** using quiver plots.
    
    ### **Requirements**
    - Use **NumPy** for array computations.
    - Use **Matplotlib** for visualization.
    - Ensure **numerical stability** by choosing an appropriate time step based on the CFL condition.
    - Save the final velocity field (u, v) in `.npy` format.
    
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

print(f"2D_Inviscid_Burgers_MK' equation prompt added to {json_file}")
