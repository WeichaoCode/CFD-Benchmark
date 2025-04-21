#!/usr/bin/env python3
import numpy as np

# Parameters
Lx = 4.0
Lz = 1.0
Ra = 2e6
Pr = 1.0
nu = (Ra/Pr)**(-0.5)      # kinematic viscosity
kappa = (Ra*Pr)**(-0.5)   # thermal diffusivity

# Grid parameters (coarse grid for speed)
Nx = 21           # number of grid points in x (periodic)
Nz = 11           # number of grid points in z (nonperiodic)
dx = Lx / Nx
dz = Lz / (Nz - 1)
x = np.linspace(0, Lx, Nx, endpoint=False)
z = np.linspace(0, Lz, Nz)

# Time parameters
t_end = 50.0
dt = 0.005      # reduced time step for stability
nsteps = int(t_end / dt)

# Initialize fields (shape: (Nz, Nx), first index = z, second = x)
u = np.zeros((Nz, Nx))      # horizontal velocity
w = np.zeros((Nz, Nx))      # vertical velocity
p = np.zeros((Nz, Nx))      # pressure
b = np.empty((Nz, Nx))      # buoyancy (temperature deviation)
for j in range(Nz):
    b[j, :] = Lz - z[j]
b += 1e-3 * (np.random.rand(Nz, Nx) - 0.5)

# Upwind derivative functions for convection terms
def upwind_deriv_x(f, velocity):
    deriv = np.empty_like(f)
    pos = velocity >= 0
    neg = ~pos
    deriv[pos] = (f - np.roll(f, 1, axis=1))[pos] / dx
    deriv[neg] = (np.roll(f, -1, axis=1) - f)[neg] / dx
    return deriv

def upwind_deriv_z(f, velocity):
    deriv = np.empty_like(f)
    # Interior points
    deriv[1:-1, :] = np.where(velocity[1:-1, :] >= 0,
                               (f[1:-1, :] - f[0:-2, :]) / dz,
                               (f[2:, :] - f[1:-1, :]) / dz)
    # Boundaries: use forward difference at bottom and backward at top
    deriv[0, :] = (f[1, :] - f[0, :]) / dz
    deriv[-1, :] = (f[-1, :] - f[-2, :]) / dz
    return deriv

# Central difference derivatives for pressure projection and laplacian
def ddx(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dx)

def ddz(f):
    f_z = np.empty_like(f)
    f_z[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*dz)
    f_z[0, :] = (f[1, :] - f[0, :]) / dz
    f_z[-1, :] = (f[-1, :] - f[-2, :]) / dz
    return f_z

def laplacian(f):
    lap_x = (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / (dx**2)
    lap_z = np.empty_like(f)
    lap_z[1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / (dz**2)
    lap_z[0, :] = (2*f[1, :] - 2*f[0, :]) / (dz**2)
    lap_z[-1, :] = (2*f[-2, :] - 2*f[-1, :]) / (dz**2)
    return lap_x + lap_z

def divergence(u_field, w_field):
    return ddx(u_field) + ddz(w_field)

def poisson_solver(rhs, max_iter=50, tol=1e-6):
    # Solve Laplacian(phi) = rhs using Jacobi iteration
    phi = np.zeros_like(rhs)
    D = 2*(dx**2 + dz**2)
    for _ in range(max_iter):
        phi_new = np.copy(phi)
        phi_new[1:-1, :] = (
            (np.roll(phi, -1, axis=1)[1:-1, :] + np.roll(phi, 1, axis=1)[1:-1, :]) * (dz**2) +
            (phi[2:, :] + phi[:-2, :]) * (dx**2) -
            rhs[1:-1, :] * (dx**2) * (dz**2)
        ) / D
        # Neumann boundary conditions in z (dphi/dz = 0)
        phi_new[0, :] = phi_new[1, :]
        phi_new[-1, :] = phi_new[-2, :]
        if np.linalg.norm(phi_new - phi) < tol:
            phi = phi_new
            break
        phi = phi_new
    return phi

# Main time-stepping loop
for step in range(nsteps):
    # Advective derivatives computed with upwind scheme
    du_dx = upwind_deriv_x(u, u)
    du_dz = upwind_deriv_z(u, w)
    dw_dx = upwind_deriv_x(w, u)
    dw_dz = upwind_deriv_z(w, w)
    
    # Compute intermediate velocities (u_star, w_star) without pressure gradient
    u_star = u - dt * (u * du_dx + w * du_dz) + dt * nu * laplacian(u)
    w_star = w - dt * (u * dw_dx + w * dw_dz) + dt * nu * laplacian(w) + dt * b

    # Enforce Dirichlet BCs for velocity at top (z = Lz) and bottom (z = 0)
    u_star[0, :] = 0.0
    u_star[-1, :] = 0.0
    w_star[0, :] = 0.0
    w_star[-1, :] = 0.0

    # Pressure projection step to enforce incompressibility
    div_u_star = divergence(u_star, w_star)
    rhs = div_u_star / dt
    phi = poisson_solver(rhs, max_iter=50, tol=1e-6)
    
    # Compute gradients of phi
    phi_x = ddx(phi)
    phi_z = ddz(phi)
    
    # Correct velocities
    u_new = u_star - dt * phi_x
    w_new = w_star - dt * phi_z
    u_new[0, :] = 0.0
    u_new[-1, :] = 0.0
    w_new[0, :] = 0.0
    w_new[-1, :] = 0.0

    # Update pressure field by accumulating pressure correction
    p = p + phi

    # Buoyancy update with upwind advection and diffusion
    db_dx = upwind_deriv_x(b, u)
    db_dz = upwind_deriv_z(b, w)
    b_new = b - dt * (u * db_dx + w * db_dz) + dt * kappa * laplacian(b)
    # Enforce buoyancy BCs: b = Lz at bottom (z=0) and b = 0 at top (z=Lz)
    b_new[0, :] = Lz
    b_new[-1, :] = 0.0

    # Update fields for next time step
    u = u_new.copy()
    w = w_new.copy()
    b = b_new.copy()

# Save final solution arrays (2D arrays)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Rayleigh_Benard_Convection.npy', u)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/w_2D_Rayleigh_Benard_Convection.npy', w)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Rayleigh_Benard_Convection.npy', p)
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/b_2D_Rayleigh_Benard_Convection.npy', b)