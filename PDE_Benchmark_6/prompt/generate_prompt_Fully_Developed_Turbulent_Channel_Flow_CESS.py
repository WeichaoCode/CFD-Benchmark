import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "Fully_Developed_Turbulent_Channel_Flow_CESS": """
    You are tasked with solving a **fully-developed turbulent flow in a channel** using the **Reynolds-Averaged Navier-Stokes (RANS) equations** 
    and the **Cess algebraic turbulence model**. The goal is to numerically compute the velocity profile using the **finite difference method (FDM)** and 
    solve the resulting system of equations.
    ---
    
    ### **Governing Equation**
    The RANS equation for this problem simplifies to:
    
    \[
    \frac{d}{dy} \left( (\mu + \mu_t) \frac{d\bar{u}}{dy} \right) = -1.
    \]
    
    where:
    - \( \mu \) is the molecular viscosity.
    - \( \mu_t \) is the turbulent eddy viscosity (computed using the Cess turbulence model).
    - The effective viscosity is defined as \( \mu_{\text{eff}} = \mu + \mu_t \).
    
    By applying the product rule, we rewrite it as:
    
    \[
    \left[ \frac{d\mu_{\text{eff}}}{dy} \frac{d}{dy} + \mu_{\text{eff}} \frac{d^2}{dy^2} \right] u = -1.
    \]
    
    This can be expressed as a **linear system**:
    
    \[
    A u = b
    \]
    
    which will be solved numerically.
    
    ---
    
    ### **Tasks**
    #### **1️⃣ Generate a Non-Uniform Mesh**
    - Use a **MESH class** to compute **y-direction mesh points** and **finite difference matrices**:
      - \( n = 100 \): Number of mesh points.
      - \( H = 2 \): Channel height.
      - **Cluster mesh points near the walls** using an appropriate stretching function.
    - Implement the **MESH class**, including:
      - \( y \) coordinates.
      - First derivative matrix \( d/dy \).
      - Second derivative matrix \( d^2/dy^2 \).
    
    #### **2️⃣ Compute Effective Viscosity**
    - Implement the **Cess algebraic turbulence model**:
    
      \[
      \frac{\mu_{\text{eff}}}{\mu} = \frac{1}{2} \left(1 + \frac{1}{9} \kappa^2 Re_{\tau}^2 (2y - y^2)^2 (3 - 4y + 2y^2)^2 \left[ 1 - \exp\left(-\frac{y^+}{A}\right) \right] \right)^{1/2} - \frac{1}{2}
      \]
    
      where:
      - \( \kappa = 0.42 \) (von Kármán constant).
      - \( A = 25.4 \) (constant).
      - \( y^+ = y Re_{\tau} \).
      - \( Re_{\tau} = \frac{\rho u_{\tau} H}{\mu} \).
    
    #### **3️⃣ Discretize the Governing Equation Using Finite Difference Method**
    - Use **central difference discretization** for \( d/dy \) and \( d^2/dy^2 \):
    
      \[
      \frac{1}{\Delta y} \left( \mu_{\text{eff},i+\frac{1}{2}} \frac{u_{i+1} - u_i}{\Delta y} - \mu_{\text{eff},i-\frac{1}{2}} \frac{u_i - u_{i-1}}{\Delta y} \right) = -1.
      \]
    
    - Formulate the **linear system** \( A u = b \).
    
    #### **4️⃣ Solve the Linear System**
    - Solve the system using:
      - **Direct solvers** (e.g., LU decomposition).
      - **Under-relaxation iterative solvers** (if needed).
    
    #### **5️⃣ Plot the Velocity Profile**
    - Plot the velocity distribution \( u(y) \).
    - Compare the **turbulent velocity profile** to a **laminar parabolic profile**.
    
    ### **User-Defined Inputs**
    - Reynolds number based on friction velocity: \\( Re_\\tau = 395 \\)
    - Density: \\( \\rho = 1.0 \\)
    - Dynamic viscosity: \\( \\mu = \\frac{1}{Re_\\tau} \\)
    ---
    
    ### **Requirements**
    - Implement the solution in **Python**.
    - Use **NumPy** for numerical operations.
    - Use **Matplotlib** for visualization.
    - Save the computed velocity profile in `.npy` format.
    - Structure the code modularly, including:
      - A `Mesh` class for grid generation.
      - A function to compute turbulent viscosity.
      - A function to solve the linear system.
    ---
    
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
