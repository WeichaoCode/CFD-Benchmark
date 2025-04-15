import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "2D_Navier_Stokes_Channel": """
    You are given the **two-dimensional channel flow problem** governed by the **Navier-Stokes equations**. The flow is driven by a constant source term \( F \) in the \( u \)-momentum equation to mimic pressure-driven channel flow.

    ### **Governing Equations**
    The modified incompressible Navier-Stokes equations are:

    \[
    \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + F
    \]

    \[
    \frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
    \]

    \[
    \frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} = -\rho \left( \frac{\partial u}{\partial x} \frac{\partial u}{\partial x} + 2 \frac{\partial u}{\partial y} \frac{\partial v}{\partial x} + \frac{\partial v}{\partial y} \frac{\partial v}{\partial y} \right)
    \]

    where:
    - \( u(x,y,t) \) and \( v(x,y,t) \) are the velocity components in the \( x \) and \( y \) directions.
    - \( p(x,y,t) \) is the pressure field.
    - \( \rho \) is the fluid density.
    - \( \nu \) is the kinematic viscosity.
    - \( F \) is the constant external force applied in the \( x \)-direction.

    ### **Initial Condition:**
    - The velocity components \( u, v \) and pressure \( p \) are initialized to **zero everywhere** in the domain.

    ### **Boundary Conditions:**
    - **Periodic boundary conditions** for \( u, v, p \) in the \( x \)-direction at \( x = 0, 2 \).
    - **No-slip boundary condition** for \( u, v \) at \( y = 0, 2 \) (i.e., \( u = 0, v = 0 \)).
    - **Pressure gradient condition:** \( \frac{\partial p}{\partial y} = 0 \) at \( y = 0, 2 \).
    - **Source term:** \( F = 1 \) applied uniformly throughout the domain.

    ### **Computational Domain and Parameters:**
    - The equation is solved over a **rectangular domain** with spatial extent:
      - \( x \in [0, 2] \), \( y \in [0, 2] \)
    - The **fluid properties** are:
      - Density: \( \rho = 1 \)
      - Kinematic viscosity: \( \nu = 0.1 \)
    - The equation is solved over a **rectangular grid** with:
      - Number of grid points in \( x \)-direction: \( nx = 41 \)
      - Number of grid points in \( y \)-direction: \( ny = 41 \)
      - Number of time steps: \( nt = 10 \)

    ### **Tasks**
    1. Implement the **Navier-Stokes solver** for **channel flow**.
    2. Use a structured grid with uniform spacing.
    3. Apply periodic and no-slip boundary conditions as specified.
    4. Simulate the velocity and pressure field evolution over time.
    5. Visualize the computed **velocity field** using quiver plots.

    ### **Requirements**
    - Use **NumPy** for array computations.
    - Use **Matplotlib** for visualization.
    - Ensure **numerical stability** by choosing an appropriate time step.
    - Save the final velocity field (u, v) and pressure field (p) in `.npy` format.

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
