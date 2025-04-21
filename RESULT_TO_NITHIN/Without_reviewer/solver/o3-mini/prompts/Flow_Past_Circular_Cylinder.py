#!/usr/bin/env python3
import numpy as np

# Parameters
r_min = 0.5
r_max = 10.0
N_r = 100         # radial grid points
N_theta = 128     # angular grid points
nu = 0.005        # kinematic viscosity
v_infty = 1.0     # free stream velocity
t_final = 1.0
dt = 0.001
num_steps = int(t_final/dt)

# Create grid
r = np.linspace(r_min, r_max, N_r)
d_r = r[1] - r[0]
theta = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
d_theta = theta[1] - theta[0]

# Initialize fields: psi and omega (2D arrays: shape (N_r, N_theta))
psi = np.zeros((N_r, N_theta))
omega = np.zeros((N_r, N_theta))

# Set boundary conditions for psi (streamfunction)
# Inner boundary (cylinder surface): psi = 20
psi[0, :] = 20.0
# Outer boundary: psi = v_infty * y + 20, with y = r*sin(theta) where r=r_max
for j in range(N_theta):
    y_outer = r_max * np.sin(theta[j])
    psi[-1, j] = v_infty * y_outer + 20.0

# Set initial vorticity BC (will be updated in time stepping)
# Inner boundary: omega = 2*(psi[0]-psi[1])/(d_r^2)
omega[0, :] = 2*(psi[0, :] - psi[1, :])/(d_r**2)
# Outer boundary: omega = 0
omega[-1, :] = 0.0

# SOR parameters for solving Poisson equation for psi
sor_omega = 1.5
sor_tol = 1e-6
sor_max_iter = 5000

def solve_psi(psi, omega):
    """
    Solve the Poisson equation in polar coordinates:
        (psi_rr + 1/r * psi_r + 1/r^2 * psi_tt) = -omega
    using SOR iterative method on the interior points.
    The boundary values of psi are prescribed.
    """
    psi_new = psi.copy()
    for it in range(sor_max_iter):
        max_diff = 0.0
        # Loop over interior radial points i=1...N_r-2; theta is periodic
        for i in range(1, N_r-1):
            r_i = r[i]
            for j in range(N_theta):
                jp = (j+1) % N_theta
                jm = (j-1) % N_theta
                # Coefficients from finite difference approximations
                coef_r_minus = 1.0/(d_r**2) - 1.0/(2.0*r_i*d_r)
                coef_r_plus  = 1.0/(d_r**2) + 1.0/(2.0*r_i*d_r)
                coef_theta   = 1.0/( (r_i**2) * (d_theta**2) )
                coef_center  = -2.0/(d_r**2) - 2.0/( (r_i**2) * (d_theta**2) )
                residual = (coef_r_minus * psi_new[i-1, j] +
                            coef_r_plus  * psi_new[i+1, j] +
                            coef_theta*(psi_new[i, jp] + psi_new[i, jm]) +
                            omega[i, j] - coef_center*psi_new[i, j])
                psi_old = psi_new[i, j]
                psi_new[i, j] = psi_old + sor_omega * residual / (-coef_center)
                diff = abs(psi_new[i, j] - psi_old)
                if diff > max_diff:
                    max_diff = diff
        if max_diff < sor_tol:
            break
    return psi_new

# Main time-stepping loop for the vorticity transport (unsteady problem)
for n in range(num_steps):
    # Step 1: Solve for psi from Poisson eq: ∇²ψ = -ω, with current omega and BCs
    psi = solve_psi(psi, omega)
    # Enforce boundary conditions for psi (they remain fixed)
    psi[0, :] = 20.0
    for j in range(N_theta):
        y_outer = r_max * np.sin(theta[j])
        psi[-1, j] = v_infty * y_outer + 20.0
    
    # Step 2: Compute velocity field from psi
    u_r = np.zeros_like(psi)
    u_theta = np.zeros_like(psi)
    # Use central differences for interior points only (skip boundaries)
    for i in range(1, N_r-1):
        r_i = r[i]
        for j in range(N_theta):
            jp = (j+1) % N_theta
            jm = (j-1) % N_theta
            u_r[i,j] = (psi[i, jp] - psi[i, jm])/(2.0*d_theta*r_i)
            u_theta[i,j] = - (psi[i+1, j] - psi[i-1, j])/(2.0*d_r)
    # For boundaries, we can use one-sided approximations if needed; here we keep zero interior update.
    
    # Step 3: Update vorticity omega using explicit Euler time stepping
    omega_new = omega.copy()
    for i in range(1, N_r-1):
        r_i = r[i]
        for j in range(N_theta):
            jp = (j+1) % N_theta
            jm = (j-1) % N_theta
            # Radial derivative d(omega)/dr (central)
            d_omega_dr = (omega[i+1, j] - omega[i-1, j])/(2.0*d_r)
            # Angular derivative d(omega)/dtheta (central, periodic)
            d_omega_dtheta = (omega[i, jp] - omega[i, jm])/(2.0*d_theta)
            # Laplacian of omega in polar coordinates
            omega_rr = (omega[i+1, j] - 2.0*omega[i, j] + omega[i-1, j])/(d_r**2)
            omega_tt = (omega[i, jp] - 2.0*omega[i, j] + omega[i, jm])/( (r_i**2)*(d_theta**2) )
            laplacian_omega = omega_rr + omega_tt
            # Advective term
            advective = u_r[i,j]*d_omega_dr + (u_theta[i,j]/r_i)*d_omega_dtheta
            # Time update (explicit Euler)
            omega_new[i, j] = omega[i, j] - dt * advective + dt * nu * laplacian_omega

    # Update omega interior
    omega = omega_new.copy()
    
    # Step 4: Enforce boundary conditions on omega
    # Inner boundary (i=0): omega = 2*(psi[0]-psi[1])/(d_r^2)
    omega[0, :] = 2*(psi[0, :] - psi[1, :])/(d_r**2)
    # Outer boundary (i=N_r-1): omega = 0
    omega[-1, :] = 0.0

# Save final solutions (only the final time step, as 2D arrays)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/psi_Flow_Past_Circular_Cylinder.npy', psi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/omega_Flow_Past_Circular_Cylinder.npy', omega)