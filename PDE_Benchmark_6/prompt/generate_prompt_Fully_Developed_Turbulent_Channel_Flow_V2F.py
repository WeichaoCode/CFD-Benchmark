import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "Fully_Developed_Turbulent_Channel_Flow_V2F": """
    You are tasked with solving a **fully-developed turbulent flow in a channel** using the **Reynolds-Averaged Navier-Stokes (RANS) equations** 
    and the **V2F turbulence model**. The goal is to numerically compute the velocity profile using the **finite difference method (FDM)** 
    and solve the resulting system of equations.
    
    ---
    
    ### **Governing Equations**
    The RANS equation for this problem simplifies to:
    
    #### **Turbulent Kinetic Energy** \( k \):
    
    \[
    0 = P_k - \rho \epsilon + \frac{d}{dy} \left[ \left( \mu + \frac{\mu_t}{\sigma_k} \right) \frac{d k}{dy} \right]
    \]
    
    where \( P_k \) is the turbulent production term.
    
    #### **Turbulent Dissipation** \( \epsilon \):
    
    \[
    0 = \frac{1}{T} \left( C_{e1} P_k - C_{e2} \rho \epsilon \right) + \frac{d}{dy} \left[ \left( \mu + \frac{\mu_t}{\sigma_\epsilon} \right) \frac{d \epsilon}{dy} \right]
    \]
    
    #### **Wall-Normal Fluctuation Component** \( v^2 \):
    
    \[
    0 = \rho k f - 6 \rho v^2 \frac{\epsilon}{k} + \frac{d}{dy} \left[ \left( \mu + \frac{\mu_t}{\sigma_k} \right) \frac{d v^2}{dy} \right]
    \]
    
    #### **Elliptic Relaxation Equation** \( f \):
    
    \[
    L^2 \frac{d^2 f}{dy^2} - f = \frac{1}{T} \left[ C_1 \left( 6 - v^2 \right) - \frac{2}{3} \left( C_1 - 1 \right) \right] - C_2 P_k
    \]
    
    where \( L \) is a characteristic length scale and \( P_k \) is the turbulent kinetic energy production term.
    
    #### **Eddy Viscosity** \( \mu_t \):
    
    \[
    \mu_t = C_\mu \rho \left( \frac{\epsilon}{k} \right)^{1/2} T_t
    \]
    
    where:
    - \( C_\mu \) is a constant.
    - \( T_t \) is the turbulent temperature.
    
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
    - Implement the **V2F model** for computing \( k \), \( \epsilon \), and \( v^2 \).
    
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
