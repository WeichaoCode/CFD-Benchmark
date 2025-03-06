import json
import os

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
TASKS_FILE = os.path.join(ROOT_DIR, "data", "PDE_TASK_POOL.json")
# PROMPTS_FILE = os.path.join(ROOT_DIR, "prompts", "PDE_TASK_PROMPT.json")
PROMPTS_FILE = os.path.join(ROOT_DIR, "prompts", "PDE_TASK_FUNCTION_PROMPT.json")

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
- Use a **Finite Difference Method (FDM)** or an alternative stable scheme.
- Ensure numerical **stability** using **appropriate CFL conditions**.
- Choose **grid resolution** and **time step size**.

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
    # pde_prompts["prompts"][task_name] = generate_prompt_solver(task_name, task_info)
    pde_prompts["prompts"][task_name] = generate_prompt_function(task_name, task_info)

# Save the generated prompts to a new JSON file
with open(PROMPTS_FILE, "w") as file:
    json.dump(pde_prompts, file, indent=4)

print(f"✅ LLM prompts generated and saved to {PROMPTS_FILE}")
