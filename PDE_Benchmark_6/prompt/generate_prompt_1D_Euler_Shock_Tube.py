import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "1D_Euler_Shock_Tube": """
   You are given the **one-dimensional Euler equations**, which govern compressible flow dynamics in a **shock tube**. 
   
   Your task is to numerically solve these equations using the **MacCormack method**.

   ### **Governing Equations**
   The Euler equations in conservative form are:

   \\[
   \\frac{\\partial \\mathbf{U}}{\\partial t} + \\frac{\\partial \\mathbf{F}}{\\partial x} = 0
   \\]

   where:

   \\[
   \\mathbf{U} = 
   \\begin{bmatrix} 
   \\rho \\\\ 
   \\rho u \\\\ 
   \\rho E 
   \\end{bmatrix}, 
   \\quad 
   \\mathbf{F} = 
   \\begin{bmatrix} 
   \\rho u \\\\ 
   \\rho u^2 + p \\\\ 
   u(\\rho E + p) 
   \\end{bmatrix}
   \\]

   with:

   - \\( \\rho \\): Density  
   - \\( u \\): Velocity  
   - \\( p \\): Pressure  
   - \\( E \\): Total energy per unit mass,  
     \\( E = \\frac{p}{(\\gamma - 1)\\rho} + \\frac{u^2}{2} \\)  
   - \\( \\gamma \\): Ratio of specific heats (typically 1.4 for air)

   ### **Computational Domain**
   - Spatial domain: \\( x \\in [-1, 1] \\)
   - Temporal domain: \\( t \\in [0, 0.25] \\)

   ### **Initial Conditions**
   The shock tube is initially divided into two regions:

   - **Left region** (\\( 0 \\leq x < x_0 \\)):
     - \\( \\rho_L = 1.0 \\)
     - \\( u_L = 0.0 \\)
     - \\( p_L = 1.0 \\)

   - **Right region** (\\( x_0 \\leq x \\leq L \\)):
     - \\( \\rho_R = 0.125 \\)
     - \\( u_R = 0.0 \\)
     - \\( p_R = 0.1 \\)

   ### **Boundary Conditions**
   - **Reflective (no-flux) boundary conditions** at both ends of the tube.

   ### **Numerical Method**
   - Implement the **MacCormack Method**, a two-step predictor-corrector approach for hyperbolic systems.

   ### **Implementation Steps**
   1. **Define Parameters:**
      - Tube length \\( L \\) 2
      - Number of spatial points \\( N_x \\) N_x = 81
      - Time step \\( \\Delta t \\) determined using CFL, CFL = 1
      - Total simulation time \\( T \\) 0.25
      - Ratio of specific heats \\( \\gamma \\) gamma = 1.4

   2. **Discretize the Domain:**
      - Spatial grid: \\( x = \\{x_0, x_1, ..., x_{N_x-1}\\} \\)
      - Time steps: \\( t = \\{t_0, t_1, ..., t_{N_t-1}\\} \\)

   3. **Initialize Variables:**
      - Set initial \\( \\rho \\), \\( u \\), and \\( p \\) based on the initial conditions.
      - Compute initial conservative variables \\( \\mathbf{U} \\).

   4. **Time Integration:**
      - For each time step:
        - Compute flux \\( \\mathbf{F} \\) from \\( \\mathbf{U} \\).
        - Apply the predictor step to estimate \\( \\mathbf{U}^{*} \\).
        - Compute flux \\( \\mathbf{F}^{*} \\) from \\( \\mathbf{U}^{*} \\).
        - Apply the corrector step to update \\( \\mathbf{U}^{n+1} \\).
        - Update primitive variables \\( \\rho \\), \\( u \\), and \\( p \\) from \\( \\mathbf{U}^{n+1} \\).

   5. **Visualization:**
      - Plot the **density** (\\( \\rho \\)), **velocity** (\\( u \\)), and **pressure** (\\( p \\)) profiles at various time steps.
      - Observe the formation of shock waves, contact discontinuities, and expansion fans.

   ### **Expected Output**
   The simulation should capture and visualize the following flow features over time:
   - **Shock wave propagation**
   - **Contact discontinuity movement**
   - **Expansion fan formation**

   ### **Requirements**
   - Use an appropriate **finite difference scheme**.
   - Ensure **stability and accuracy** using a suitable time step.
   - Save the computed density, velocity, and pressure profiles in `.npy` format.
   - Save mathbf{U}, mathbf{F} in `.npy` format.

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
