import numpy as np

# Parameters
Lx, Lz = 1.0, 2.0
Nx, Nz = 128, 256
dx, dz = Lx / Nx, Lz / Nz
dt = 0.0001  # Adjusted time step for stability
T = 20.0
nu = 1 / 5e4
D = nu
save_values = ['u', 'w', 's']

# Grid
x = np.linspace(0, Lx, Nx, endpoint=False)
z = np.linspace(-Lz/2, Lz/2, Nz, endpoint=False)
X, Z = np.meshgrid(x, z, indexing='ij')

# Initial conditions
u = 0.5 * (1 + np.tanh((Z - 0.5) / 0.1) - np.tanh((Z + 0.5) / 0.1))
w = 0.01 * np.sin(2 * np.pi * X) * (np.exp(-((Z - 0.5) ** 2) / 0.01) + np.exp(-((Z + 0.5) ** 2) / 0.01))
s = u.copy()

# Helper functions for periodic boundary conditions
def periodic_bc(arr):
    arr[0, :] = arr[-2, :]
    arr[-1, :] = arr[1, :]
    arr[:, 0] = arr[:, -2]
    arr[:, -1] = arr[:, 1]

# Time-stepping loop
t = 0.0
max_steps = int(T / dt)
for step in range(max_steps):
    # Compute derivatives
    u_x = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    u_z = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dz)
    w_x = (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) / (2 * dx)
    w_z = (np.roll(w, -1, axis=1) - np.roll(w, 1, axis=1)) / (2 * dz)
    
    u_xx = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / (dx ** 2)
    u_zz = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / (dz ** 2)
    w_xx = (np.roll(w, -1, axis=0) - 2 * w + np.roll(w, 1, axis=0)) / (dx ** 2)
    w_zz = (np.roll(w, -1, axis=1) - 2 * w + np.roll(w, 1, axis=1)) / (dz ** 2)
    
    s_x = (np.roll(s, -1, axis=0) - np.roll(s, 1, axis=0)) / (2 * dx)
    s_z = (np.roll(s, -1, axis=1) - np.roll(s, 1, axis=1)) / (2 * dz)
    s_xx = (np.roll(s, -1, axis=0) - 2 * s + np.roll(s, 1, axis=0)) / (dx ** 2)
    s_zz = (np.roll(s, -1, axis=1) - 2 * s + np.roll(s, 1, axis=1)) / (dz ** 2)
    
    # Update equations
    u_new = u + dt * (-u * u_x - w * u_z + nu * (u_xx + u_zz))
    w_new = w + dt * (-u * w_x - w * w_z + nu * (w_xx + w_zz))
    s_new = s + dt * (-u * s_x - w * s_z + D * (s_xx + s_zz))
    
    # Apply periodic boundary conditions
    periodic_bc(u_new)
    periodic_bc(w_new)
    periodic_bc(s_new)
    
    # Update variables
    u, w, s = u_new, w_new, s_new
    t += dt

    # Break early if the solution stabilizes
    if step % 1000 == 0:
        if np.allclose(u, u_new, atol=1e-6) and np.allclose(w, w_new, atol=1e-6) and np.allclose(s, s_new, atol=1e-6):
            break

# Save final results
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)