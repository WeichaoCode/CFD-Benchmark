import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "Fully_Developed_Turbulent_Channel_Flow_SST": """
   You are tasked with solving a **fully-developed turbulent flow in a channel** using the **Reynolds-Averaged Navier-Stokes (RANS) equations** 
    and the **Menter Shear-Stress Transport (SST) turbulence model**. The goal is to numerically compute the velocity profile using the **finite difference method (FDM)** 
    and solve the resulting system of equations.
    
    ---
    
    ### **Governing Equations**
    The RANS equation for this problem simplifies to:
    
    #### **Turbulent Kinetic Energy** \( k \):
    
    \[
    0 = P_k - \beta^* \rho k \omega + \frac{d}{dy} \left[ \left( \mu + \frac{\mu_t}{\sigma_k} \right) \frac{d k}{dy} \right]
    \]
    
    #### **Specific Turbulent Dissipation** \( \omega \):
    
    \[
    0 = \frac{\rho P_k}{\mu_t} - \beta \omega^2 + \frac{d}{dy} \left[ \left( \mu + \mu_t \omega \right) \frac{d \omega}{dy} \right] + (1 - F_1) C_D k \omega
    \]
    
    where:
    - \( P_k \) is the turbulent production term.
    - \( \mu_t \) is the turbulent eddy viscosity.
    - \( F_1 \) is a blending function.
    - \( C_D \) is a constant.
    
    #### **Eddy Viscosity** \( \mu_t \):
    
    \[
    \mu_t = \rho k \min \left( \frac{1}{\omega}, \frac{a_1}{\|S\| F_2} \right)
    \]
    
    where:
    - \( S \) is the strain rate tensor.
    - \( F_2 \) is another blending function.
    - \( a_1 \) is a constant.
    
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
    
    #### **2️⃣ Compute Turbulent Kinetic Energy and Dissipation**
    - Implement the **Menter SST model** to solve for \( k \) and \( \omega \).
    
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
