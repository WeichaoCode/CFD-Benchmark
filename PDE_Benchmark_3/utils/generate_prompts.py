import json
import os

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
TASKS_FILE = os.path.join(ROOT_DIR, "data", "PDE_TASK_POOL.json")
PROMPTS_FILE = os.path.join(ROOT_DIR, "prompts", "PDE_TASK_PROMPT.json")

# Load PDE task pool JSON
with open(TASKS_FILE, "r") as file:
    pde_tasks = json.load(file)


def generate_prompt_function(pde_name, pde_info):
    prompt = f"""
    Solve the **{pde_name.replace('_', ' ')}** using the given **Manufactured Solution (MMS)**:
    
        MMS: {pde_info["mms"]}
    
    Write a **Python solver** `solve_{pde_name.lower()}` using an appropriate **Finite Difference Method (FDM)** to solve the PDE:
    
        {pde_info["pde"]}
    
    Ensure numerical **stability** by choosing a suitable **time step** and **grid resolution**.
    Hint 1:
    - Use central differences for second derivatives  
    - Use upwind schemes for advection to prevent numerical oscillations.
    - Use an implicit scheme for stability when solving diffusion-dominated problems.
    - If uncertain, search online for examples of Python solvers for similar PDEs and adapt the best numerical scheme.
    
    Hint 2: your can write the code follow the following steps
    - step 1: define PARAMETERS, like grid points, time steps, final time, domain size and other parameters
    - step 2: check CFL CONDITION, you can also use other method to ensure stable
    - step 3: compute source term from MMS solution
    - step 4: compute the initial and boundary conditions from MMS
    - step 5: solve the PDE using FINITE DIFFERENCE
    - step 6: compute exact solution for comparison
    - step 7: error ana,lysis and plot numerical, exact solution and error.
    """
    return prompt


# Generate prompts for all PDEs
pde_prompts = {"prompts": {}}
for task_name, task_info in pde_tasks["tasks"].items():
    pde_prompts["prompts"][task_name] = generate_prompt_function(task_name, task_info)

# Save the generated prompts to a new JSON file
with open(PROMPTS_FILE, "w") as file:
    json.dump(pde_prompts, file, indent=4)

print(f"✅ LLM prompts generated and saved to {PROMPTS_FILE}")
