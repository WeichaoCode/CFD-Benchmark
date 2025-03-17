import json
import os

# Define the directory where generated solver scripts are stored
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
GENERATED_SOLVERS_DIR = os.path.join(ROOT_DIR, "prompt")
SAVE_FILE = os.path.join(GENERATED_SOLVERS_DIR, "PDE_TASK_PROMPT.json")
# Define the prompt as a string
prompt_text = {
    "Fully_Developed_Turbulent_Channel_Flow_SA": """
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
    
    ### **Turbulence Model: Spalart-Allmaras Model**
    The turbulence is governed by the transport equation:

    \[
    0 = c_{b1} \hat{S} \tilde{\nu} - c_{w1} f_w \left( \frac{\tilde{\nu}}{y} \right)^2 + \frac{1}{c_{b3}} \frac{d}{dy} \left[ (\nu + \tilde{\nu}) \frac{d \tilde{\nu}}{dy} \right] + \frac{c_{b2}}{c_{b3}} \left( \frac{d \tilde{\nu}}{dy} \right)^2
    \]

    where the eddy viscosity is:

    \[
    \mu_t = \rho \tilde{\nu} f_{v1}
    \]

    with the following constants:
    - \( c_{v1} = 7.1 \)
    - \( c_{b1} = 0.1355 \)
    - \( c_{b2} = 0.622 \)
    - \( c_{b3} = \frac{2}{3} \)
    - \( c_{w1} = \frac{c_{b1}}{\kappa^2} + \frac{1.0 + c_{b2}}{c_{b3}} \)
    - \( c_{w2} = 0.3 \)
    - \( c_{w3} = 2.0 \)
    - \( \kappa = 0.41 \)

    Additional functions:
    - \( f_{v1} = \frac{\chi^3}{\chi^3 + c_{v1}^3} \), where \( \chi = \frac{\tilde{\nu}}{\nu} \).

    
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
