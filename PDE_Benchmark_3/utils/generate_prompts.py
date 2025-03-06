import json
import os

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
TASKS_FILE = os.path.join(ROOT_DIR, "data", "PDE_TASK_POOL.json")
PROMPTS_FILE = os.path.join(ROOT_DIR, "prompts", "PDE_TASK_PROMPT.json")

# Load PDE task pool JSON
with open(TASKS_FILE, "r") as file:
    pde_tasks = json.load(file)


# Define a template for structured LLM prompts
def generate_prompt_solver(pde_name, pde_info):
    prompt = f"""
You are an expert in **Computational Fluid Dynamics (CFD)** and **Numerical Methods**. Your task is to generate a **Python solver** for the given PDE using a suitable numerical scheme.

---

### **1️⃣ PDE Definition**
Solve the following **{pde_name.replace('_', ' ')}**:

    {pde_info["pde"]}

where:
- The primary variable(s) are {', '.join(pde_info["variables"])}.
- The source term(s) {', '.join([var for var in pde_info["variables"] if 'f' in var])} are derived from the **Manufactured Solution (MMS)**.

---

### **2️⃣ Manufactured Solution (MMS)**
To validate the solver, use the following **MMS solution**:

    {pde_info["mms"]}

From this, compute the **source terms** by substituting the MMS into the PDE.

---

### **3️⃣ Discretization Strategy**
- Select the most appropriate **Finite Difference Method (FDM)**:
  - **Explicit Methods** (Forward Euler, MacCormack, RK4) for convection-dominated problems.
  - **Implicit Methods** (Backward Euler, Crank-Nicolson) for diffusion-dominated problems.
  - **Hybrid Methods** (ADI, Predictor-Corrector) for coupled PDEs.
- If convection terms exist (e.g., Burgers, Navier-Stokes), use **Upwind Differencing** for numerical stability.
- If diffusion terms exist, **Implicit Methods** (Backward Euler or Crank-Nicolson) should be preferred to remove stability constraints.

### **4️⃣ Stability and Time-Stepping**
- Compute **CFL condition** dynamically:
- If using **explicit schemes**, enforce a **Courant number ≤ 0.5** for stability.
- For implicit solvers, choose **dt adaptively** based on error control.

### **5️⃣ Grid Resolution & Adaptive Refinement**
- Carefully grid resolution (nx, ny, nt).
- Compute **grid spacing dynamically**:  
- If using **explicit methods**, refine grid to maintain stability.
- If using **adaptive refinement**, adjust grid resolution based on **solution smoothness**.
---

### **4️⃣ Implementation Guidelines**
✅ Write a **fully functional Python script** using **NumPy & SciPy**.  
✅ Solve the PDE using an **appropriate numerical method**.  
✅ Enforce **boundary and initial conditions** from MMS.  
✅ Compute and visualize:
    1. The **numerical solution**.
    2. The **MMS (exact) solution**.
    3. The **absolute error**.

---

### **5️⃣ Output Requirements**
- Return **only the Python code** (do not explain it).
- Ensure the code is **well-commented and numerically stable**.

### **Hints for Code Generation**
- If you are unsure how to implement a numerical scheme, **search for existing Python implementations** of similar PDE solvers.
- Study how **grid resolution (`nx, ny, nt`), time step (`dt`), and CFL conditions** are handled in existing solvers.
- Learn from well-documented sources such as:
  - **Barba Group's CFD-Python**: https://github.com/barbagroup/CFDPython
  - **ENGR 491 Computational Fluid Dynamics**: https://github.com/okcfdlab/engr491.git
  - **Numerical PDEs with Python**: https://fenicsproject.org/
  - **Relevant Stack Overflow discussions**.
- Use **best practices from existing implementations** but modify them to incorporate the given **Manufactured Solution (MMS)**.
- Ensure that your function adheres to the **Finite Difference Method (FDM)** with the **appropriate numerical stability constraints**.

    """
    return prompt


def generate_prompt_function(pde_name, pde_info):
    prompt = f"""
    You are an expert in **Computational Fluid Dynamics (CFD)** and **Numerical Methods**. Your task is to write a **Python function** to solve the given PDE using a suitable numerical scheme.
    
    📌 **DO NOT include any input/output handling, function calls, or explanations.**  
    📌 **Only write a single function definition, which users will call externally.**  
    📌 **The function should take numerical parameters as arguments and return the computed solution.**  
    
    ---
    
    ### **1️⃣ PDE Definition**
    Write a function to solve the **{pde_name.replace('_', ' ')}**:
    
        {pde_info["pde"]}
    
    where:
    - The primary variable(s) are **{', '.join(pde_info["variables"])}**, which are passed as function arguments.
    - The source term(s) **{', '.join([var for var in pde_info["variables"] if 'f' in var])}** are derived from the **Manufactured Solution (MMS)**.
    
    ---
    
    ### **2️⃣ Manufactured Solution (MMS)**
    To validate the solver, the function should be tested using the following **MMS solution**:
    
        {pde_info["mms"]}
    
    Users will substitute this into the PDE externally to compute the **source terms**.
    
    ---
    
    ### **3️⃣ Function Requirements**
    - Define a **function** named `solve_{pde_name.lower()}`.
    - The function **must not** contain hardcoded values.
    - All **parameters (e.g., u, ν, x, y, t, dt, dx, dy)** must be **function arguments**.
    - Use a **Finite Difference Method (FDM)** or another stable numerical scheme.
    - Ensure **stability** by selecting an appropriate **time step size (CFL condition if needed)**.
    
    ---
    
    ### **4️⃣ Implementation Guidelines**
    ✅ Define the function as:
    ```python
    def solve_{pde_name.lower()}({', '.join(pde_info["variables"])}):
        \"\"\"
        Solves the {pde_name.replace('_', ' ')} using a numerical method.
    
        Parameters:
            {', '.join(pde_info["variables"])}: User-provided numerical values.
    
        Returns:
            numpy.ndarray: Computed numerical solution.
        \"\"\"
    5️⃣ Output Requirements
    Return only the function definition (no explanations, no function calls).
    The function should be fully self-contained and accept all numerical parameters as arguments.
    No test cases, input handling, or plotting—users will handle testing externally.
    """
    return prompt


# Generate prompts for all PDEs
pde_prompts = {"prompts": {}}
for task_name, task_info in pde_tasks["tasks"].items():
    pde_prompts["prompts"][task_name] = generate_prompt_solver(task_name, task_info)

# Save the generated prompts to a new JSON file
with open(PROMPTS_FILE, "w") as file:
    json.dump(pde_prompts, file, indent=4)

print(f"✅ LLM prompts generated and saved to {PROMPTS_FILE}")
