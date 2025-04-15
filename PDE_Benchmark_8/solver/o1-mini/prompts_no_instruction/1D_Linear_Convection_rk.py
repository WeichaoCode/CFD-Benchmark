import numpy as np

# Parameters
c = 1.0
epsilon = 5e-4
x_start = -5.0
x_end = 5.0
N_x = 101
t_final = 1.0
CFL_advection = 0.4
CFL_diffusion = 0.4

# Spatial discretization
x = np.linspace(x_start, x_end, N_x)
dx = x[1] - x[0]

# Initial condition
u = np.exp(-x**2)

# Time step based on CFL condition
dt_advection = CFL_advection * dx / abs(c)
dt_diffusion = CFL_diffusion * dx**2 / (2 * epsilon)
dt = min(dt_advection, dt_diffusion)

# Number of time steps
N_t = int(np.ceil(t_final / dt))
dt = t_final / N_t

def rhs(u):
    # Periodic boundary conditions
    u_padded = np.concatenate([u[-1:], u, u[:1]])
    # First derivative (central difference)
    du_dx = (u_padded[2:] - u_padded[0:-2]) / (2 * dx)
    # Second derivative (central difference)
    d2u_dx2 = (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[0:-2]) / dx**2
    return -c * du_dx + epsilon * d2u_dx2

# Time integration using RK4
for _ in range(N_t):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    u += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/u_1D_Linear_Convection_rk.npy', u)