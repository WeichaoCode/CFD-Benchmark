import numpy as np

# Parameters
r_min, r_max = 0.5, 10.0
theta_min, theta_max = 0, 2 * np.pi
nr, ntheta = 100, 100
dr = (r_max - r_min) / (nr - 1)
dtheta = (theta_max - theta_min) / ntheta
dt = 0.001
nu = 0.005
v_infinity = 1.0
psi_0 = 20
psi_1 = 0

# Create grid
r = np.linspace(r_min, r_max, nr)
theta = np.linspace(theta_min, theta_max, ntheta)
R, Theta = np.meshgrid(r, theta, indexing='ij')

# Initialize fields
psi = np.zeros((nr, ntheta))
omega = np.zeros((nr, ntheta))

# Time-stepping parameters
t_final = 1.0
n_steps = int(t_final / dt)

# Helper functions
def laplacian(f, dr, dtheta, r):
    d2f_dr2 = (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dr**2
    d2f_dtheta2 = (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dtheta**2
    df_dr = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dr)
    return d2f_dr2 + (1 / r[:, np.newaxis]) * df_dr + (1 / r[:, np.newaxis]**2) * d2f_dtheta2

# Time-stepping loop
for step in range(n_steps):
    # Update boundary conditions
    psi[0, :] = psi_0
    psi[-1, :] = v_infinity * r[-1] * np.sin(Theta[-1, :]) + psi_0
    omega[0, :] = 2 * (psi_0 - psi_1) / dr**2
    omega[-1, :] = 0

    # Solve Poisson equation for psi
    for _ in range(100):  # Simple iterative solver
        psi[1:-1, :] = 0.25 * (np.roll(psi, -1, axis=0)[1:-1, :] + np.roll(psi, 1, axis=0)[1:-1, :] +
                               np.roll(psi, -1, axis=1)[1:-1, :] + np.roll(psi, 1, axis=1)[1:-1, :] -
                               dr**2 * omega[1:-1, :])

    # Compute velocity field
    u_r = (1 / R) * (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * dtheta)
    u_theta = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * dr)

    # Update vorticity using the vorticity transport equation
    omega_new = omega.copy()
    omega_new[1:-1, :] = omega[1:-1, :] + dt * (
        -u_r[1:-1, :] * (np.roll(omega, -1, axis=0)[1:-1, :] - np.roll(omega, 1, axis=0)[1:-1, :]) / (2 * dr)
        - (u_theta[1:-1, :] / R[1:-1, :]) * (np.roll(omega, -1, axis=1)[1:-1, :] - np.roll(omega, 1, axis=1)[1:-1, :]) / (2 * dtheta)
        + nu * laplacian(omega, dr, dtheta, r)[1:-1, :]
    )
    omega = np.nan_to_num(omega_new)  # Replace NaNs and Infs with zero

# Save final results
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/psi_Flow_Past_Circular_Cylinder.npy', psi)
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/omega_Flow_Past_Circular_Cylinder.npy', omega)