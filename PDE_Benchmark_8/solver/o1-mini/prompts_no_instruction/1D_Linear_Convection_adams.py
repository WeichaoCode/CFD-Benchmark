import numpy as np

# Parameters
c = 1.0
epsilon = 5e-4
x_start = -5.0
x_end = 5.0
N_x = 101
t_final = 1.0

# Spatial discretization
x = np.linspace(x_start, x_end, N_x)
dx = x[1] - x[0]

# Time step based on CFL condition
dt_conv = dx / c
dt_diff = dx**2 / (2 * epsilon)
dt = 0.5 * min(dt_conv, dt_diff)

# Initial condition
u = np.exp(-x**2)

# Function to compute RHS
def compute_rhs(u):
    u_forward = np.roll(u, -1)
    u_backward = np.roll(u, 1)
    du_dx = (u_forward - u_backward) / (2 * dx)
    d2u_dx2 = (u_forward - 2 * u + u_backward) / dx**2
    return -c * du_dx + epsilon * d2u_dx2

# Initial RHS
f = compute_rhs(u)

# Time integration
t = 0.0
u_new = u.copy()

# First step using Explicit Euler
u_new = u + dt * f
t += dt
f_prev = f.copy()
f = compute_rhs(u_new)
u = u_new.copy()

# Subsequent steps using Adams-Bashforth
while t < t_final:
    if t + dt > t_final:
        dt = t_final - t
    u_new = u + dt * (3 * f - f_prev) / 2
    t += dt
    f_new = compute_rhs(u_new)
    u = u_new.copy()
    f_prev = f.copy()
    f = f_new.copy()

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/u_1D_Linear_Convection_adams.npy', u)