import numpy as np

# Problem parameters
u = 0.2  # m/s
rho = 1.0  # kg/m^3
Gamma = 0.001  # kg/(m.s)

# Domain parameters
x_start = 0.0
x_end = 2.0
t_start = 0.0
t_end = 2.5

# Numerical parameters
nx = 100
nt = 500
dx = (x_end - x_start) / nx
dt = (t_end - t_start) / nt

# Initialize solution
x = np.linspace(x_start, x_end, nx, endpoint=False)
phi = np.exp(-((x - 0.5) / 0.1)**2)

# Time loop
for n in range(nt):
    phi_old = phi.copy()

    # Finite Volume discretization
    F_w = u * phi_old
    F_e = u * phi_old
    D_w = Gamma / rho
    D_e = Gamma / rho

    a_W = D_w + max(F_w, 0)
    a_E = D_e + max(-F_e, 0)
    a_P = D_w + D_e + max(F_w, 0) + max(-F_e, 0) + rho * dx / dt
    S_u = phi_old * rho * dx / dt

    # Solve for phi
    phi[1:-1] = (a_W[1:-1] * phi_old[0:-2] + a_E[1:-1] * phi_old[2:] + S_u[1:-1]) / a_P[1:-1]

    # Periodic boundary conditions
    phi[0] = phi[-2]
    phi[-1] = phi[1]

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/phi_1D_Unsteady_Convection_Diffusion_Periodic.npy', phi)