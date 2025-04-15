import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "Fully_Developed_Turbulent_Channel_Flow_SA": """
    You are tasked with solving a **fully-developed turbulent flow in a channel** using the **Reynolds-Averaged Navier-Stokes (RANS) equations** 
    and the **Spalart-Allmaras (SA) turbulence model**. The goal is to numerically compute the velocity profile using the **finite difference method (FDM)** and 
    solve the resulting system of equations.
    ---
    
    ### **Governing Equation**
    The RANS equation for this problem simplifies to:
    
    \[
    \frac{d}{dy} \left( (\mu + \mu_t) \frac{d\bar{u}}{dy} \right) = -1.
    \]
    
    where:
    - \( \mu \) is the molecular viscosity.
    - \( \mu_t \) is the turbulent eddy viscosity (computed using the Spalart-Allmaras turbulence model).
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
    
    ### **Turbulence Model: Spalart-Allmaras**
    Implement the **Spalart-Allmaras turbulence model**, which consists of a transport equation for an eddy viscosity-like variable \( \tilde{\nu} \):

    \[
    0 = c_{b1} \hat{S} \tilde{\nu} - c_{w1} f_w \left( \frac{\tilde{\nu}}{y} \right)^2 + \frac{1}{c_{b3}} \frac{d}{dy} \left[ \left( \nu + \tilde{\nu} \right) \frac{d \tilde{\nu}}{dy} \right] + \frac{c_{b2}}{c_{b3}} \left( \frac{d \tilde{\nu}}{dy} \right)^2
    \]

    where the eddy viscosity is given by:

    \[
    \mu_t = \rho \tilde{\nu} f_{\nu 1}
    \]

    with:

    \[
    f_{\nu 1} = \frac{\chi^3}{\chi^3 + c_{\nu 1}^3}, \quad \chi = \frac{\tilde{\nu}}{\nu}
    \]

    and the constants:

    \[
    c_{v1} = 7.1, \quad c_{b1} = 0.1355, \quad c_{b2} = 0.622, \quad c_{b3} = \frac{2}{3}
    \]

    \[
    c_{w1} = \frac{c_{b1}}{\kappa^2} + \frac{1.0 + c_{b2}}{c_{b3}}, \quad c_{\nu 2} = 0.3, \quad c_{\nu 3} = 2.0, \quad \kappa = 0.41
    \]

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
    - Solve the **Spalart-Allmaras model** for \( \tilde{\nu} \) and compute \( \mu_t \).
    
    #### **3️⃣ Discretize the Governing Equation Using Finite Difference Method**
    - Use **central difference discretization** for \( d/dy \) and \( d^2/dy^2 \).
    
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
