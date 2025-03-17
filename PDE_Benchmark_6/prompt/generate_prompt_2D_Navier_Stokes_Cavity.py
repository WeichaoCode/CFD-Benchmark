import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "2D_Cavity_Flow_Navier_Stokes": """
    You are given the **two-dimensional incompressible Navier-Stokes equations**, which describe the motion of a viscous fluid in a square cavity:

    \\[
    \\frac{\\partial u}{\\partial t} + u \\frac{\\partial u}{\\partial x} + v \\frac{\\partial u}{\\partial y} = -\\frac{1}{\\rho} \\frac{\\partial p}{\\partial x} + \\nu \\left( \\frac{\\partial^2 u}{\\partial x^2} + \\frac{\\partial^2 u}{\\partial y^2} \\right)
    \\]

    \\[
    \\frac{\\partial v}{\\partial t} + u \\frac{\\partial v}{\\partial x} + v \\frac{\\partial v}{\\partial y} = -\\frac{1}{\\rho} \\frac{\\partial p}{\\partial y} + \\nu \\left( \\frac{\\partial^2 v}{\\partial x^2} + \\frac{\\partial^2 v}{\\partial y^2} \\right)
    \\]

    \\[
    \\frac{\\partial^2 p}{\\partial x^2} + \\frac{\\partial^2 p}{\\partial y^2} = -\\rho \\left( \\frac{\\partial u}{\\partial x} \\frac{\\partial u}{\\partial x} + 2 \\frac{\\partial u}{\\partial y} \\frac{\\partial v}{\\partial x} + \\frac{\\partial v}{\\partial y} \\frac{\\partial v}{\\partial y} \\right)
    \\]

    ### **Objective**
    Solve the **2D cavity flow problem** numerically using the **finite difference method (FDM)**.

    ### **Numerical Method**
    - Discretize the momentum and pressure equations using finite difference schemes.
    - Solve for velocity components \\( u \\) and \\( v \\) using an explicit time-stepping method.
    - Solve for pressure \\( p \\) using an iterative solver.

    ### **Initial Condition**
    The velocity and pressure fields are initialized to **zero** everywhere:

    \\[
    u, v, p = 0
    \\]

    ### **Boundary Conditions**
    - **Top boundary (Lid-driven motion)**:  
      - \\( u = 1 \\) at \\( y = 2 \\)  
      - \\( v = 0 \\) at \\( y = 2 \\)
    - **Other boundaries (no-slip walls)**:  
      - \\( u = 0 \\), \\( v = 0 \\)  
    - **Pressure boundary conditions**:
      - \\( \\frac{\\partial p}{\\partial y} = 0 \\) at \\( y = 0 \\)  
      - \\( p = 0 \\) at \\( y = 2 \\)  
      - \\( \\frac{\\partial p}{\\partial x} = 0 \\) at \\( x = 0, 2 \\)

    ### **Computational Domain and Parameters**
    - Solve in a **square cavity** with:  
      - \\( x \\in [0,2] \\), \\( y \\in [0,2] \\)
    - Grid resolution:
      - Number of grid points in \\( x \\)-direction: \\( nx = 41\\)  
      - Number of grid points in \\( y \\)-direction: \\( ny = 41\\)  
    - Time-stepping:
      - Number of time steps: \\( nt = 500\\)  
      - Time step size: \\( \\Delta t = 0.001\\)  
    ### **Fluid Properties**
    - The fluid density is set to **\\( \\rho = 1 \\)**.
    - The kinematic viscosity is set to **\\( \\nu = 0.1 \\)**.

    ### **Tasks**
    1. Implement the **finite difference method (FDM)** for solving the **2D incompressible Navier-Stokes equations**.
    2. Use a **structured grid** with uniform spacing.
    3. Iterate over time and solve for \\( u, v, p \\).
    4. Visualize the **velocity field (quiver plot)** and **pressure distribution (contour plot)**.

    ### **Requirements**
    - Use **NumPy** for array computations.
    - Use **Matplotlib** for visualization.
    - Ensure **numerical stability** by choosing an appropriate time step based on the CFL condition.
    - Save the computed velocity \\( u, v \\) and pressure \\( p \\) in `.npy` format.

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
