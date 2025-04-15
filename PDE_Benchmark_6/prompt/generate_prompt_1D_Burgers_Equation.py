import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {"1D_Burgers_Equation": """You are given the **one-dimensional Burgers' equation**, a fundamental PDE that models **nonlinear convection** and **diffusion**:

\\[
\\frac{\\partial u}{\\partial t} + u \\frac{\\partial u}{\\partial x} = \\nu \\frac{\\partial^2 u}{\\partial x^2}
\\]

where:
- \\( u(x,t) \\) is the velocity field,
- \\( \\nu \\) is the viscosity coefficient,
- \\( x \\) is the spatial coordinate,
- \\( t \\) is time.
variable declarations: 
- number of grid points in x: nx = 101
- number of time steps: nt = 100
- dx = 2 * numpy.pi / (nx - 1)
- viscosity coefficient: nu = .07
- dt = dx * nu

### **Task:**
1. **Numerically solve Burgers' equation** over a given spatial and temporal domain.
2. Apply **periodic boundary conditions**:
   \\[
   u(0) = u(2\\pi)
   \\]
3. Use the following **initial condition**:
   \\[
   u = -\\frac{2\\nu}{\\phi} \\frac{\\partial \\phi}{\\partial x} + 4
   \\]
   where:
   \\[
   \\phi = \\exp{\\left(\\frac{-x^2}{4\\nu}\\right)} + \\exp{\\left(\\frac{-(x - 2\\pi)^2}{4\\nu}\\right)}
   \\]
4. Compare the numerical solution to the **analytical solution** given by:
   \\[
   u = -\\frac{2\\nu}{\\phi} \\frac{\\partial \\phi}{\\partial x} + 4
   \\]
   where:
   \\[
   \\phi = \\exp{\\left(\\frac{-(x - 4t)^2}{4\\nu(t + 1)}\\right)} + \\exp{\\left(\\frac{-(x - 4t - 2\\pi)^2}{4\\nu(t + 1)}\\right)}
   \\]

### **Requirements:**
- Use an appropriate **finite difference scheme** for the numerical solution.
- Ensure **stability and accuracy** of the scheme.
- Output the numerical solution in a format that allows easy comparison with the analytical solution.
- Save the velocity field at last time step in a .npy file.

**Return only the Python code that implements this solution.**
"""}

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

print(f"Burgers' equation prompt added to {json_file}")
