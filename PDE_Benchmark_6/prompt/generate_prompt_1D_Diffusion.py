import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "1D_Diffusion": """You are given the **one-dimensional diffusion equation**, a fundamental PDE that models **diffusive transport**:

\\[
\\frac{\\partial u}{\\partial t} = \\nu \\frac{\\partial^2 u}{\\partial x^2}
\\]

where:
- \\( u(x,t) \\) represents the quantity being diffused (e.g., temperature, concentration),
- \\( \\nu \\) is the diffusion coefficient,
- \\( x \\) is the spatial coordinate,
- \\( t \\) is time.

### **Variable Declarations:**
- Number of grid points in x: `nx = 41`
- Number of time steps: `nt = 20`
- Grid spacing: `dx = 2 / (nx - 1)`
- Diffusion coefficient: `nu = 0.3`
- sigma = 0.2
- Time step: `dt = sigma * dx**2 / nu`

### **Task:**
1. **Numerically solve the diffusion equation** over a given spatial and temporal domain.
2. Apply **Dirichlet boundary conditions**:
   \\[
   u(0) = 1, \quad u(1) = 0
   \\]
3. Use the following **initial condition**:
   - At \\( t = 0 \\), let \\( u(x, 0) = 2 \\) for \\( 0.5 \leq x \leq 1 \\), and \\( u = 1 \\) elsewhere.
4. **Visualize the solution**:
   - Generate plots showing how the field \\( u(x, t) \\) evolves over time.
   - Show the final solution at \\( t = T \\).

### **Requirements:**
- Use an appropriate **finite difference scheme** for the numerical solution.
- Ensure **stability and accuracy** by choosing an appropriate time step.
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

print(f"1D Diffusion' equation prompt added to {json_file}")
