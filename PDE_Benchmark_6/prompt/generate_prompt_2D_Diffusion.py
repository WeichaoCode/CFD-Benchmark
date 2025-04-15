import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
prompt_text = {
    "2D_Diffusion": """
    You are tasked with solving the **two-dimensional diffusion equation**, which models the spreading of a scalar field due to diffusion.

    ### **Governing Equation**
    The 2D diffusion equation is given by:

    \[
    \frac{\partial u}{\partial t} = \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
    \]

    where:
    - \( u(x,y,t) \) represents the diffused quantity (e.g., temperature or concentration),
    - \( nu \) is the diffusion coefficient,
    - \( x, y \) are the spatial coordinates,
    - \( t \) is time.

    ### **Computational Domain**
    - **Spatial domain:** \( x \) and \( y \) both range from 0 to 2.
    - **Grid parameters:**
      - Number of grid points: \( n_x = n_y = 31 \).
      - Grid spacing: \( \Delta x = \Delta y = \frac{2}{n_x - 1} \).
    - **Temporal domain:**
    - number of time steps: nt = 50
    - sigma = .25
    - dt = sigma * dx * dy / nu

    ### **Initial and Boundary Conditions**
    - **Initial condition:** \( u = 2 \) in the region \( 0.5 \leq x, y \leq 1 \), and \( u = 1 \) elsewhere.
    - **Boundary conditions:** Fixed Dirichlet boundary conditions with \( u = 1 \) at all domain boundaries.

    ### **Numerical Method**
    - **Time integration:** Utilize the **Explicit Euler Method**.
    - **Spatial discretization:** Use **second-order central differences** for the diffusion terms.

    ### **Implementation Steps**
    1. **Define Parameters:**
       - Set the domain size, grid resolution, diffusion coefficient, time step, and total simulation time.
    2. **Initialize Variables:**
       - Create a 2D array for \( u \) with the specified initial condition.
    3. **Time Integration Loop:**
       - For each time step until \( t_{\text{final}} \):
         - Compute the temporary array \( u_{\text{temp}} \) using the current values of \( u \).
         - Update \( u \) using the Explicit Euler formula:
           \[
           u^{n+1}_{i,j} = u^n_{i,j} + \nu \Delta t \left( \frac{u^n_{i+1,j} - 2u^n_{i,j} + u^n_{i-1,j}}{\Delta x^2} + \frac{u^n_{i,j+1} - 2u^n_{i,j} + u^n_{i,j-1}}{\Delta y^2} \right)
           \]
         - Apply boundary conditions to maintain \( u = 1 \) at the domain boundaries.
    4. **Visualization:**
       - Generate contour plots of \( u \) at selected time intervals to observe the evolution of the diffused field.

    ### **Requirements**
    - Use **NumPy** for numerical computations.
    - Use **Matplotlib** for plotting and visualization.
    - Save the final solution \( u \) in a `.npy` file.

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
