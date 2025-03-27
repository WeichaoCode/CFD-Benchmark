import numpy as np

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.001  # Reduced time step for stability
dx = dt / nu
T = 500  # number of time steps

# Discretize the spatial domain
x = np.linspace(0, L, int(np.ceil(L / dx)), endpoint=False)
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Lax-Wendroff method
for n in range(T):
    u_next = np.zeros_like(u)
    for j in range(nx):
        # Periodic boundary conditions
        jp1 = (j + 1) % nx
        jm1 = (j - 1) % nx
        
        # Flux
        F_jp1 = 0.5 * u[jp1]**2
        F_jm1 = 0.5 * u[jm1]**2
        F_j = 0.5 * u[j]**2
        
        # Jacobian
        A_jp1 = u[jp1]
        A_j = u[j]
        A_jm1 = u[jm1]
        
        # Lax-Wendroff update
        u_next[j] = (u[j] 
                     - dt / (2 * dx) * (F_jp1 - F_jm1) 
                     + dt**2 / (2 * dx**2) * (A_jp1 * (F_jp1 - F_j) - A_jm1 * (F_j - F_jm1)))
    
    u = u_next

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_1/u_1D_Nonlinear_Convection_LW.npy', u)