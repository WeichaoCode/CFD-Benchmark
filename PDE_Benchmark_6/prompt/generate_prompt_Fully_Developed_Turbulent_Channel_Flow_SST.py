import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "Fully_Developed_Turbulent_Channel_Flow_SST": """
    You are given the **fully-developed turbulent flow in a channel**, governed by the **Reynolds-averaged Navier-Stokes (RANS) equations**. The goal is to solve for the mean velocity profile in the channel by modeling the Reynolds shear stress using an **eddy viscosity model**.

    ### **Governing Equations**
    The RANS equation for fully-developed turbulent flow in a channel is:

    \\[
    \\frac{d}{dy} \\left( (\\mu + \\mu_t) \\frac{du}{dy} \\right) = -1
    \\]

    where:
    - \\( u(y) \\) is the mean velocity,
    - \\( \\mu \\) is the molecular viscosity,
    - \\( \\mu_t \\) is the eddy viscosity, representing the effects of turbulence.

    Using the **Boussinesq approximation**, the Reynolds stress is modeled as:

    \\[
    \\rho \\overline{u_i' u_j'} \\approx \\mu_t \\left( \\frac{\\partial u_i}{\\partial x_j} + \\frac{\\partial u_j}{\\partial x_i} \\right)
    \\]

    Defining an **effective viscosity** as \\( \\mu_{eff} = \\mu + \\mu_t \\), we can rewrite the equation as:

    \\[
    \\left[ \\frac{d \\mu_{eff}}{dy} \\frac{d}{dy} + \\mu_{eff} \\frac{d^2}{dy^2} \\right] u = -1
    \\]

    ### **Objective**
    Solve for the mean velocity profile \\( u(y) \\) in a **fully-developed turbulent channel flow**.

    ### **Numerical Method**
    - Discretize the equation using **finite difference methods**.
    - Solve the resulting **linear system** using an appropriate solver.
    - Implement an **eddy viscosity model** to account for turbulence effects.

    ### **Computational Domain and Parameters**
    - The channel is **steady** and **fully-developed**.
    - The mean velocity depends only on \\( y \\).
    - Use an appropriate **grid resolution** in the \\( y \\)-direction.
    
    ### **Turbulence Model: Menter SST Model**
    The turbulent kinetic energy equation:

    \[
    0 = P_k - \beta^* \rho k \omega + \frac{d}{dy} \left[ (\mu + \mu_t / \sigma_k) \frac{dk}{dy} \right]
    \]

    The specific turbulent dissipation equation:

    \[
    0 = \rho \alpha \frac{P_k}{\mu_t} - \beta \rho \omega^2 + \frac{d}{dy} \left[ (\mu + \mu_t \sigma_{\omega}) \frac{d\omega}{dy} \right] + (1 - F_1) C_{D \omega}
    \]

    The eddy viscosity is given by:

    \[
    \mu_t = \rho k \min \left( \frac{1}{\omega}, \frac{a_1}{|S|F_2} \right)
    \]

    
    ### **Tasks**
    1. Implement the finite difference discretization for the governing equation.
    2. Use an eddy viscosity model to define \\( \\mu_t \\).
    3. Solve for the velocity profile \\( u(y) \\).
    4. Visualize the velocity profile.

    ### **Requirements**
    - Use **NumPy** for numerical computations.
    - Use **Matplotlib** for visualization.
    - Save the computed velocity profile in `.npy` format.

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
