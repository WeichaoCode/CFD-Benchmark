import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "2D_Unsteady_Heat_Equation_ADI": """
    You are given the **two-dimensional unsteady heat equation** with a source term:
    
    \\[
    \\frac{\\partial T}{\\partial t} - \\alpha \\left( \\frac{\\partial^2 T}{\\partial x^2} + \\frac{\\partial^2 T}{\\partial y^2} \\right) = q(x, y, t).
    \\]
    
    ### **Objective**
    Solve the equation numerically using the **Alternating Direction Implicit (ADI) Method**.
    
    ### **Numerical Method**
    - The ADI scheme solves the equation in **two steps**:
      1. Solve implicitly in the **x-direction**, explicitly in the **y-direction**.
      2. Solve implicitly in the **y-direction**, explicitly in the **x-direction**.
    
    - The intermediate step is given by:
    
      \\[
      T_{i,j}^{n+1/2} = 0.5r (T_{i+1,j}^{n+1/2} - 2T_{i,j}^{n+1/2} + T_{i-1,j}^{n+1/2}) + 0.5\\beta^2 r (T_{i,j+1}^{n} - 2T_{i,j}^{n} + T_{i,j-1}^{n}) + T_{i,j}^{n} + 0.5 \\Delta t q
      \\]
    
    - The second step solves for the next full time step:
    
      \\[
      T_{i,j}^{n+1} = 0.5r (T_{i+1,j}^{n+1} - 2T_{i,j}^{n+1} + T_{i-1,j}^{n+1}) + 0.5\\beta^2 r (T_{i,j+1}^{n+1/2} - 2T_{i,j}^{n+1/2} + T_{i,j-1}^{n+1/2}) + T_{i,j}^{n+1/2} + 0.5 \\Delta t q
      \\]
    
    - The ADI scheme is **unconditionally stable** and allows larger time steps.
    
    ### **Computational Domain and Parameters**
    - The equation is solved on a rectangular domain: \\( x, y \\in [-1,1] \\).
    - Boundary conditions: **Fixed at 0°C on all sides**.
    - Source term:
    
      \\[
      q(x,y) = Q_0 \\exp\\left(-\\frac{x^2 + y^2}{2\\sigma^2} \\right)
      \\]
    
    - \\( \\sigma = 0.1 \\), \\( Q_0 = 200°C/s \\).
    
    ### **Computational Domain and Parameters**
    - Grid resolution: \\( nx = 41, ny = 41 \\) (41 points in the x and y directions).
    - Maximum simulation time: \\( t_{max} = 3 \\) seconds.
    - Thermal diffusivity coefficient: \\( \\alpha = 1 \\).
    - Grid spacing relationships:
      - \\( \\beta = \\frac{dx}{dy} \\) (ratio of grid spacing).
      - \\( r = \\frac{r}{1 + \\beta^2} \\) (adjusted stability parameter).
    - Time step size:
      - \\( dt = r \\cdot \\frac{dx^2}{\\alpha} \\).
      
    ### **Tasks**
    1. Implement the **ADI Method**.
    2. Use a structured grid with uniform spacing.
    3. Apply **Dirichlet boundary conditions**.
    4. Visualize the temperature evolution.
    
    ### **Requirements**
    - Use **NumPy** for array computations.
    - Use **Matplotlib** for visualization.
    - Save the computed solution in `.npy` format.
    
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
