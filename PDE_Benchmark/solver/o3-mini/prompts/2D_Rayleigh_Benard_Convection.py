#!/usr/bin/env python3
import numpy as np

# Set NumPy error handling to ignore overflow warnings in critical sections.
np.seterr(over='ignore', divide='ignore', invalid='ignore')

# Parameters
Ra = 2e6
Pr = 1
nu = (Ra / Pr) ** (-0.5)
kappa = (Ra * Pr) ** (-0.5)

Lx = 4.0
Lz = 1.0
t_final = 50.0

# Discretization parameters (coarse grid for speed and stability)
Nx = 65           # number of grid points in x
Nz = 17           # number of grid points in z
dx = Lx / (Nx - 1)
dz = Lz / (Nz - 1)

# Time stepping parameters (dt chosen for stability)
dt = 0.005
nt = int(t_final / dt)

# Pressure Poisson solver parameters
p_tol = 1e-6
p_max_iter = 300
p_omega = 1.7  # SOR relaxation factor

# Create grids
x = np.linspace(0, Lx, Nx)
z = np.linspace(0, Lz, Nz)
X, Z = np.meshgrid(x, z, indexing='ij')

# Initialize fields: u: horizontal, w: vertical, p, b (buoyancy)
u = np.zeros((Nx, Nz))
w = np.zeros((Nx, Nz))
p = np.zeros((Nx, Nz))
np.random.seed(42)
b = (Lz - Z) + 1e-3 * (np.random.rand(Nx, Nz) - 0.5)

# Apply buoyancy boundary conditions: bottom: b = Lz, top: b = 0
b[:, 0] = Lz
b[:, -1] = 0.0

def laplacian(f):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        lap = ((np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dx**2 +
               (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dz**2)
    return lap

def divergence(u_field, w_field):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        du_dx = (np.roll(u_field, -1, axis=0) - np.roll(u_field, 1, axis=0)) / (2 * dx)
        dw_dz = (np.roll(w_field, -1, axis=1) - np.roll(w_field, 1, axis=1)) / (2 * dz)
        div = du_dx + dw_dz
    return div

def pressure_poisson(p, u_star, w_star, dt):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        rhs = divergence(u_star, w_star) / dt
    denom = 2 / dx**2 + 2 / dz**2
    for _ in range(p_max_iter):
        p_old = p.copy()
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            p_new = (1 - p_omega) * p + (p_omega / denom) * (
                (np.roll(p, -1, axis=0) + np.roll(p, 1, axis=0)) / dx**2 +
                (np.roll(p, -1, axis=1) + np.roll(p, 1, axis=1)) / dz**2 - rhs)
        p = np.nan_to_num(p_new, nan=0.0, posinf=1e6, neginf=-1e6)
        # Enforce Neumann boundary condition in z for p (dp/dz = 0)
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        diff = p - p_old
        if not np.all(np.isfinite(diff)):
            break
        res = np.linalg.norm(diff.ravel(), 2)
        if res < p_tol:
            break
    return p

def apply_velocity_bc(u_field, w_field):
    # No-slip at top and bottom: u = 0, w = 0
    u_field[:, 0] = 0.0
    u_field[:, -1] = 0.0
    w_field[:, 0] = 0.0
    w_field[:, -1] = 0.0
    return u_field, w_field

def apply_buoyancy_bc(b_field):
    # Enforce buoyancy boundary conditions: bottom: b = Lz, top: b = 0
    b_field[:, 0] = Lz
    b_field[:, -1] = 0.0
    return b_field

def upwind_x(f, velocity):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        diff_backward = (f - np.roll(f, 1, axis=0)) / dx
        diff_forward  = (np.roll(f, -1, axis=0) - f) / dx
    return np.where(velocity >= 0, diff_backward, diff_forward)

def upwind_z(f, velocity):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        diff_backward = (f - np.roll(f, 1, axis=1)) / dz
        diff_forward  = (np.roll(f, -1, axis=1) - f) / dz
    return np.where(velocity >= 0, diff_backward, diff_forward)

# Time stepping loop
for n in range(nt):
    # Upwind derivatives for velocity components
    du_dx = upwind_x(u, u)
    du_dz = upwind_z(u, w)
    dw_dx = upwind_x(w, u)
    dw_dz = upwind_z(w, w)
    
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        conv_u = u * du_dx + w * du_dz
        conv_w = u * dw_dx + w * dw_dz
    conv_u = np.nan_to_num(conv_u, nan=0.0, posinf=1e6, neginf=-1e6)
    conv_w = np.nan_to_num(conv_w, nan=0.0, posinf=1e6, neginf=-1e6)
    
    lap_u = laplacian(u)
    lap_w = laplacian(w)
    
    # Compute intermediate velocities (u_star, w_star) including buoyancy in vertical momentum
    u_star = u + dt * (-conv_u + nu * lap_u)
    w_star = w + dt * (-conv_w + nu * lap_w + b)
    
    u_star, w_star = apply_velocity_bc(u_star, w_star)
    
    # Solve for pressure correction
    p = pressure_poisson(p, u_star, w_star, dt)
    
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        dp_dx = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dx)
        dp_dz = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dz)
    dp_dx = np.nan_to_num(dp_dx, nan=0.0, posinf=1e6, neginf=-1e6)
    dp_dz = np.nan_to_num(dp_dz, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Correct velocities for incompressibility
    u = u_star - dt * dp_dx
    w = w_star - dt * dp_dz
    u, w = apply_velocity_bc(u, w)
    
    # Buoyancy advection-diffusion; upwind for advection terms
    db_dx = upwind_x(b, u)
    db_dz = upwind_z(b, w)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        conv_b = u * db_dx + w * db_dz
    conv_b = np.nan_to_num(conv_b, nan=0.0, posinf=1e6, neginf=-1e6)
    lap_b = laplacian(b)
    b = b + dt * (-conv_b + kappa * lap_b)
    b = apply_buoyancy_bc(b)
    
    # Exit early if any non-finite values are encountered
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(w)) and 
            np.all(np.isfinite(p)) and np.all(np.isfinite(b))):
        print("Non-finite values encountered at time step", n)
        break

# Save final solutions as .npy files (each variable saved as a 2D NumPy array)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Rayleigh_Benard_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/w_2D_Rayleigh_Benard_Convection.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Rayleigh_Benard_Convection.npy', p)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/b_2D_Rayleigh_Benard_Convection.npy', b)