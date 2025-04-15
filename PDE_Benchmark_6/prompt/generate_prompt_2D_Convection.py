import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
prompt_text = {
    "2D_Convection": """
    You are tasked with solving the **two-dimensional nonlinear convection equations**, which model the transport of quantities in a fluid flow.

    ### **Governing Equations**
    The 2D nonlinear convection equations are given by:

    \[
    \begin{align*}
    \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} &= 0 \\
    \frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} &= 0
    \end{align*}
    \]

    where:
    - \( u(x,y,t) \) and \( v(x,y,t) \) are the velocity components in the \( x \) and \( y \) directions, respectively.

    ### **Computational Domain**
    - **Spatial domain:** \( x \) and \( y \) both range from 0 to 2.
    - **Grid parameters:**
      - Number of grid points in each direction: \( n_x = n_y = 101 \).
      - Grid spacing: \( \Delta x = \Delta y = \frac{2}{n_x - 1} \).
    - **Temporal domain:**
    - number of time steps: nt = 80
    - sigma = 0.2
    - dt = sigma * dx

    ### **Initial and Boundary Conditions**
    - **Initial condition:** Both \( u \) and \( v \) are initialized to 1 throughout the domain, except in the region \( 0.5 \leq x, y \leq 1 \), where they are set to 2.
    - **Boundary conditions:** All boundaries are subject to Dirichlet conditions with \( u = 1 \) and \( v = 1 \).

    ### **Numerical Method**
    - **Time integration:** Utilize the **Explicit Euler Method**.
    - **Spatial discretization:** Apply **first-order upwind differences** for the convection terms.

    ### **Implementation Steps**
    1. **Define Parameters:**
       - Set the domain size, grid resolution, time step, and total simulation time.
    2. **Initialize Variables:**
       - Create 2D arrays for \( u \) and \( v \) with the specified initial conditions.
    3. **Time Integration Loop:**
       - For each time step until \( t_{\text{final}} \):
         - Compute the temporary arrays \( u_{\text{temp}} \) and \( v_{\text{temp}} \) using the current values of \( u \) and \( v \).
         - Update \( u \) and \( v \) using the Explicit Euler formula:
           \[
           \begin{align*}
           u^{n+1}_{i,j} &= u^n_{i,j} - \Delta t \left( u_{i,j} \frac{u_{i,j} - u_{i-1,j}}{\Delta x} + v_{i,j} \frac{u_{i,j} - u_{i,j-1}}{\Delta y} \right) \\
           v^{n+1}_{i,j} &= v^n_{i,j} - \Delta t \left( u_{i,j} \frac{v_{i,j} - v_{i-1,j}}{\Delta x} + v_{i,j} \frac{v_{i,j} - v_{i,j-1}}{\Delta y} \right)
           \end{align*}
           \]
         - Apply boundary conditions to maintain \( u = 1 \) and \( v = 1 \) at the domain boundaries.
    4. **Visualization:**
       - Generate surface plots of \( u \) and \( v \) at selected time intervals to observe the evolution of the velocity fields.

    ### **Requirements**
    - Use **NumPy** for numerical computations.
    - Use **Matplotlib** for plotting and visualization.
    - Save the final velocity fields \( u \) and \( v \) in `.npy` files.

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
