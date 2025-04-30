import numpy as np
from scipy.integrate import solve_ivp

# Spatial domain
x = np.linspace(0, 2*np.pi, 100)

# Time domain
t_span = (0, 5)

# Initial condition
u0 = np.sin(x) + 0.5 * np.sin(0.5 * x)

def rhs(t, u):
  dudt = -u * np.gradient(u, x)
  return dudt

# Solve the PDE
sol = solve_ivp(rhs, t_span, u0, t_eval=np.linspace(0, 5, 100), vectorized=True)

# Save the solution at the final time step
u_final = sol.y[:, -1]
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/u_final_1D_Nonlinear_Convection.npy', u_final)