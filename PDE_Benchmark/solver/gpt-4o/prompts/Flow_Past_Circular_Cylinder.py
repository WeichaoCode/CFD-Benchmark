import numpy as np

# Parameters
r_min, r_max = 0.5, 10.0
theta_min, theta_max = 0, 2 * np.pi
t_min, t_max = 0, 10
nu = 0.005
v_infinity = 1.0

# Discretization
Nr = 100
Ntheta = 100
Nt = 500
dr = (r_max - r_min) / (Nr - 1)
dtheta = (theta_max - theta_min) / Ntheta
dt = (t_max - t_min) / Nt

# Grid
r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)
R, Theta = np.meshgrid(r, theta, indexing='ij')

# Initial conditions
psi = np.zeros((Nr, Ntheta))
omega = np.zeros((Nr, Ntheta))

# Time-stepping loop
for n in range(Nt):
    # Compute velocity components
    ur = (1 / R) * np.gradient(psi, axis=1) / dtheta
    utheta = -np.gradient(psi, axis=0) / dr

    # Update vorticity using finite difference method
    omega_new = np.copy(omega)
    for i in range(1, Nr-1):
        for j in range(Ntheta):
            jp = (j + 1) % Ntheta
            jm = (j - 1) % Ntheta
            omega_new[i, j] = (omega[i, j] +
                               dt * (-ur[i, j] * (omega[i+1, j] - omega[i-1, j]) / (2 * dr) -
                                     utheta[i, j] * (omega[i, jp] - omega[i, jm]) / (2 * dtheta * R[i, j]) +
                                     nu * ((omega[i+1, j] - 2 * omega[i, j] + omega[i-1, j]) / dr**2 +
                                           (omega[i, jp] - 2 * omega[i, j] + omega[i, jm]) / (dtheta**2 * R[i, j]**2))))

    # Apply boundary conditions
    omega_new[0, :] = 2 * (20 - psi[1, :]) / dr**2  # Inner boundary
    omega_new[-1, :] = 0  # Outer boundary

    # Update vorticity
    omega = omega_new

    # Solve Poisson equation for streamfunction
    for _ in range(100):  # Iterative solver
        psi_new = np.copy(psi)
        for i in range(1, Nr-1):
            for j in range(Ntheta):
                jp = (j + 1) % Ntheta
                jm = (j - 1) % Ntheta
                psi_new[i, j] = 0.25 * (psi[i+1, j] + psi[i-1, j] +
                                        (psi[i, jp] + psi[i, jm]) / (R[i, j]**2) +
                                        dr**2 * omega[i, j])
        psi = psi_new

    # Apply boundary conditions for psi
    psi[0, :] = 20  # Inner boundary
    psi[-1, :] = v_infinity * R[-1, :] * np.sin(Theta[-1, :]) + 20  # Outer boundary

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/psi_Flow_Past_Circular_Cylinder.npy', psi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/omega_Flow_Past_Circular_Cylinder.npy', omega)