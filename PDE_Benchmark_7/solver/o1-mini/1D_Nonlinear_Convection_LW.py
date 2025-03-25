import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.5           # CFL number
dt = 0.01          # Time step
T = 500            # Number of time steps
dx = dt / nu       # Space step

# Spatial grid
x = np.arange(0, 2 * np.pi, dx)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Time-stepping using Lax-Wendroff method
for _ in range(T):
    F = 0.5 * u**2
    u_p = np.roll(u, -1)   # u_{j+1}
    u_m = np.roll(u, 1)    # u_{j-1}
    F_p = np.roll(F, -1)   # F_{j+1}
    F_m = np.roll(F, 1)    # F_{j-1}
    
    A_p_half = (u + u_p) / 2
    A_m_half = (u_m + u) / 2
    
    u_new = (u 
             - (dt / (2 * dx)) * (F_p - F_m) 
             + (dt**2 / (2 * dx**2)) * (A_p_half * (F_p - F) - A_m_half * (F - F_m)))
    
    u = u_new

# Plot the final solution
plt.figure(figsize=(8, 4))
plt.plot(x, u, label='Lax-Wendroff')
plt.xlabel('x')
plt.ylabel('u(x, T)')
plt.title('Solution of Nonlinear Convection Equation at Final Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_1D_Nonlinear_Convection_LW.npy', u)