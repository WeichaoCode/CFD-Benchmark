import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "1D_Linear_Convection_corr": """
    You are given the **one-dimensional linear convection equation**, which models wave propagation with convection and damping:

    \\[
    \\frac{\\partial u}{\\partial t} + c \\frac{\\partial u}{\\partial x} = \\epsilon \\frac{\\partial^2 u}{\\partial x^2}
    \\]

    where:
    - \\( u(x,t) \\) represents the wave amplitude,
    - \\( c \\) is the convection speed,
    - \\( \\epsilon \\) is a damping factor.

    ### **Computational Domain**
    - Solve the equation in a **periodic domain**:  
      \\[
      x \\in (-5, 5)
      \\]
    - The **initial condition** is given by:  
      \\[
      u_0 = e^{-x^2}
      \\]
    - Consider two cases:  
      - **Undamped case**: \\( \\epsilon = 0 \\)  
      - **Damped case**: \\( \\epsilon = 5 \\times 10^{-4} \\)  

    ### **Numerical Method**
    - Use the **Predictor-Corrector Method** for **time discretization**.  
    - Apply **2nd-order central differences** for **spatial discretization** of the derivatives.  
    - Ensure numerical stability by choosing an appropriate time step.  

    ### **Implementation Steps**
    1. **Define Parameters:**
       - Spatial domain: \\( x \\in (-5,5) \\)  
       - Number of grid points: \\( N_x = 101 \\)  
       - Time step: \\( \\Delta t \\) determined using CFL condition  
       - Convection speed: \\( c = 1 \\)  
       - Damping factor: \\( \\epsilon = 0 \\) or \\( \\epsilon = 5 \\times 10^{-4} \\)  

    2. **Discretize the Domain:**
       - Spatial grid: \\( x = \\{x_0, x_1, ..., x_{N_x-1}\\} \\)  
       - Time steps: \\( t = \\{t_0, t_1, ..., t_{N_t-1}\\} \\)  

    3. **Initialize Variables:**
       - Set initial wave profile \\( u(x,0) = e^{-x^2} \\).  

    4. **Time Integration using Predictor-Corrector Method:**
       - **Predictor Step** (Explicit Euler Method):
         \\[
         u^{*} = u^n + \\Delta t f(t_n, u^n)
         \\]
       - **Corrector Step** (Trapezoidal Rule):
         \\[
         u^{n+1} = u^n + \\frac{\\Delta t}{2} \\left[ f(t_n, u^n) + f(t_{n+1}, u^{*}) \\right]
         \\]
       - Apply **periodic boundary conditions**.  

    5. **Visualization:**
       - Plot the **wave profile** at different time steps.  
       - Compare the **damped** and **undamped** cases.  

    ### **Expected Output**
    The simulation should capture and visualize the following:  
    - **Wave propagation over time**.  
    - **Comparison between damped and undamped cases**.  
    - **Final wave profile at \\( t = T \\)**.  

    ### **Requirements**
    - Use **NumPy** for numerical operations.  
    - Use **Matplotlib** for visualization.  
    - Save the computed solution in a `.npy` file.  

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

print(f"1D_Linear_Convection_Predictor_Corrector' equation prompt added to {json_file}")
