#!/usr/bin/env python3
import numpy as np

# Parameters
v_infty = 1.0           # free-stream velocity for outer BC
nu = 0.005              # kinematic viscosity
T_final = 0.005         # final time (shortened for stability)
dt = 5e-5               # time step (small for stability)

# Grid parameters
r_min = 0.5
r_max = 10.0
Nr = 100                # number of radial grid points
Ntheta = 128            # number of angular grid points

r = np.linspace(r_min, r_max, Nr, dtype=np.float64)
theta = np.linspace(0, 2*np.pi, Ntheta, endpoint=False, dtype=np.float64)
dr = r[1] - r[0]
dtheta = theta[1] - theta[0]

# Create 2D meshgrid for polar coordinates (r varying along axis=0)
R, Theta = np.meshgrid(r, theta, indexing='ij')

# Initialize streamfunction (psi) and vorticity (omega)
psi = np.zeros((Nr, Ntheta), dtype=np.float64)
omega = np.zeros((Nr, Ntheta), dtype=np.float64)

# Set Dirichlet boundary conditions for psi
psi[0, :] = 20.0
psi[-1, :] = v_infty * (r_max * np.sin(theta)) + 20.0

def solve_poisson(psi, omega, tol=1e-4, max_iter=50):
    # Solve: (psi[i+1,j] - 2*psi[i,j] + psi[i-1,j])/dr**2 +
    #        (psi[i+1,j] - psi[i-1,j])/(2*dr*r[i]) +
    #        (psi[i,j+1] - 2*psi[i,j] + psi[i,j-1])/(r[i]**2*dtheta**2) = -omega[i,j]
    # Rearranged to update psi[i,j]:
    # psi[i,j] = { (psi[i+1,j]+psi[i-1,j])/dr**2 +
    #              (psi[i,j+1]+psi[i,j-1])/(r[i]**2*dtheta**2) +
    #              (psi[i+1,j]-psi[i-1,j])/(2*dr*r[i]) + omega[i,j] }
    #           / { 2/dr**2 + 2/(r[i]**2*dtheta**2) }
    psi_new = psi.copy()
    # Vectorized r for interior nodes (i=1:Nr-1, exclude boundaries)
    r_int = r[1:-1].reshape(-1, 1)
    coeff_denom = 2.0/dr**2 + 2.0/(r_int**2 * dtheta**2)
    for it in range(max_iter):
        psi_old = psi_new[1:-1, :].copy()
        psi_ip1 = psi_new[2:, :]
        psi_im1 = psi_new[:-2, :]
        psi_jp = np.roll(psi_new[1:-1, :], -1, axis=1)
        psi_jm = np.roll(psi_new[1:-1, :], 1, axis=1)
        numerator = (psi_ip1 + psi_im1)/dr**2 \
                    + (psi_jp + psi_jm)/(r_int**2 * dtheta**2) \
                    + (psi_ip1 - psi_im1)/(2*dr*r_int) \
                    + omega[1:-1, :]
        psi_update = numerator / coeff_denom
        psi_new[1:-1, :] = psi_update
        psi_new[0, :] = 20.0
        psi_new[-1, :] = v_infty * (r_max * np.sin(theta)) + 20.0
        if np.max(np.abs(psi_new[1:-1, :] - psi_old)) < tol:
            break
    return psi_new

nsteps = int(T_final/dt)
for step in range(nsteps):
    # Solve Poisson for streamfunction psi
    psi = solve_poisson(psi, omega)
    
    # Compute velocity components from psi
    # u_r = (1/r) * (dpsi/dtheta) using central differences with periodic boundaries
    u_r = (1.0/R) * (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2*dtheta)
    # u_theta = - dpsi/dr using central differences; use one-sided at boundaries
    u_theta = np.zeros_like(psi)
    u_theta[1:-1, :] = -(psi[2:, :] - psi[:-2, :])/(2*dr)
    u_theta[0, :] = -(psi[1, :] - psi[0, :])/dr
    u_theta[-1, :] = -(psi[-1, :] - psi[-2, :])/dr

    # Compute spatial derivatives for omega
    domega_dr = np.zeros_like(omega)
    domega_dr[1:-1, :] = (omega[2:, :] - omega[:-2, :])/(2*dr)
    domega_dtheta = (np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1))/(2*dtheta)
    
    # Laplacian of omega in polar coordinates (for interior nodes)
    lap_r = (omega[2:, :] - 2*omega[1:-1, :] + omega[:-2, :])/(dr**2)
    lap_r += (omega[2:, :] - omega[:-2, :])/(2*dr*r[1:-1].reshape(-1, 1))
    lap_theta = (np.roll(omega, -1, axis=1)[1:-1, :] - 2*omega[1:-1, :] + np.roll(omega, 1, axis=1)[1:-1, :]) / ((r[1:-1].reshape(-1, 1))**2 * dtheta**2)
    laplacian_omega = np.zeros_like(omega)
    laplacian_omega[1:-1, :] = lap_r + lap_theta

    # Compute advective term (only for interior nodes)
    advective = u_r * domega_dr + (u_theta / R) * domega_dtheta

    # Update vorticity omega using explicit Euler for interior nodes
    omega_update = omega[1:-1, :] + dt * (-advective[1:-1, :] + nu * laplacian_omega[1:-1, :])
    omega[1:-1, :] = omega_update

    # Enforce vorticity boundary conditions
    omega[0, :] = 2*(psi[0, :] - psi[1, :])/(dr**2)
    omega[-1, :] = 0.0

np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/psi_Flow_Past_Circular_Cylinder.npy', psi)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/omega_Flow_Past_Circular_Cylinder.npy', omega)