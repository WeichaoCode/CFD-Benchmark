import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "2D_Diffusion_FVM": """
        You are given the **two-dimensional diffusion equation**, which models the steady-state diffusion of velocity in a square duct using the **Finite Volume Method (FVM)**.

    ### **Governing Equation**
    The equation governing the flow field is derived from the incompressible **Navier-Stokes equations** for the velocity component **w** in the z-direction:

    \\[
    \\nabla \\cdot ( \\mu \\nabla w ) - \\frac{dP}{dz} = 0
    \\]

    where:
    - \\( w(x, y) \\) represents the velocity component in the z-direction.
    - \\( \\mu \\) is the fluid's dynamic viscosity.
    - \\( P \\) is the pressure.
    - \\( \\nabla \\) is the two-dimensional gradient operator.

    ### **Computational Domain**
    - The flow is considered **steady** (\\( \\partial w / \\partial t = 0 \\)).
    - The flow occurs in a **square duct** with sides of length \\( h = 0.1m \\).
    - The **no-slip** boundary condition applies at all walls.
    - The **fluid properties** are given as:
      - Dynamic viscosity: \\( \\mu = 1 \\times 10^{-3} \\) Pa·s.
      - Density: \\( \\rho = 1 \\) kg/m³.
    - The **pressure gradient** is \\( dP/dz = -3.2 \\) Pa/m.

    ### **Numerical Method**
    - Use the **Finite Volume Method (FVM)** for discretization.
    - Apply the **Gauss divergence theorem** to convert the governing PDE into a system of algebraic equations over **control volumes**.
    - Use **centered finite differences** to approximate derivatives.
    - Solve the resulting algebraic system iteratively using **Jacobi iteration**.

    ### **Discretization**
    - The domain is discretized into a **uniform Cartesian grid**.
    - The control volume is centered at **P** with neighboring points **E, W, N, S**.
    - The resulting algebraic equation at each grid point is:

      \\[
      a_P w_P = a_E w_E + a_W w_W + a_N w_N + a_S w_S - S_u
      \\]

    - Coefficients:

      \\[
      a_E = a_W = \\frac{\\mu \\Delta y}{\\Delta x}, \\quad a_N = a_S = \\frac{\\mu \\Delta x}{\\Delta y}
      \\]

      \\[
      a_P = a_E + a_W + a_N + a_S
      \\]

    - The pressure gradient source term is:

      \\[
      S_u = \\frac{dP}{dz} \\Delta x \\Delta y
      \\]

    ### **Implementation Steps**
    1. **Define the computational grid:**
       - Grid points: **nx × ny**
       - Grid spacing: **Δx = Δy = h / (nx - 1)**
    
    2. **Discretize the governing equation using the Finite Volume Method.**
    
    3. **Construct the coefficient matrix for the algebraic system.**
    
    4. **Solve for w using an iterative solver (Jacobi iteration).**
    
    5. **Visualize the solution**:
       - Plot the velocity field \\( w(x,y) \\) using contour plots.

    ### **Expected Output**
    - A **contour plot** showing the velocity distribution in the square duct.
    - Save the computed velocity field as a `.npy` file.

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
