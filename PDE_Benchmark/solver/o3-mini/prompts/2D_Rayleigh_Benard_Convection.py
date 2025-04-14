#!/usr/bin/env python3
import numpy as np

# Domain and simulation parameters
Lx = 4.0
Lz = 1.0
Nx = 128           # number of grid points in x (periodic)
Nz = 33            # number of grid points in z (including boundaries)
dx = Lx / Nx
dz = Lz / (Nz - 1)

t_final = 50.0
dt = 1e-4         # reduced time step for stability
nt = int(t_final / dt)

# Physical parameters
Ra = 2e6
Pr = 1.0
nu = (Ra/Pr)**(-0.5)        # kinematic viscosity
kappa = (Ra*Pr)**(-0.5)     # thermal diffusivity

# Create the grid
x = np.linspace(0, Lx, Nx, endpoint=False)  # periodic in x
z = np.linspace(0, Lz, Nz)
X, Z = np.meshgrid(x, z)

# Initialize fields
u = np.zeros((Nz, Nx))   # horizontal velocity
w = np.zeros((Nz, Nx))   # vertical velocity
p = np.zeros((Nz, Nx))   # pressure
# Buoyancy initial condition: b = Lz - z + small random perturbation
np.random.seed(42)
perturb = 1e-4 * (np.random.rand(Nz, Nx) - 0.5)
b = (Lz - Z) + perturb

# Finite difference helper functions

def ddx_central(f):
    # central difference, periodic in x
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dx)

def ddz_central(f):
    dfdz = np.zeros_like(f)
    dfdz[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*dz)
    dfdz[0, :] = (f[1, :] - f[0, :]) / dz
    dfdz[-1, :] = (f[-1, :] - f[-2, :]) / dz
    return dfdz

def laplacian(f):
    lap = (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dx**2
    lap[1:-1, :] += (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / dz**2
    lap[0, :] += (f[1, :] - 2*f[0, :] + f[1, :]) / dz**2
    lap[-1, :] += (f[-2, :] - 2*f[-1, :] + f[-2, :]) / dz**2
    return lap

def ddx_upwind(f, vel):
    """First order upwind derivative in x (periodic)."""
    res = np.empty_like(f)
    pos = vel >= 0
    neg = ~pos
    res[pos] = (f[pos] - np.roll(f, 1, axis=1)[pos]) / dx
    res[neg] = (np.roll(f, -1, axis=1)[neg] - f[neg]) / dx
    return res

def ddz_upwind(f, vel):
    """First order upwind derivative in z (nonperiodic)."""
    res = np.empty_like(f)
    # Bottom boundary (i=0): use forward difference
    res[0, :] = (f[1, :] - f[0, :]) / dz
    # Top boundary (i=Nz-1): use backward difference
    res[-1, :] = (f[-1, :] - f[-2, :]) / dz
    # Interior points
    for i in range(1, Nz-1):
        pos = vel[i, :] >= 0
        neg = ~pos
        res[i, pos] = (f[i, pos] - f[i-1, pos]) / dz
        res[i, neg] = (f[i+1, neg] - f[i, neg]) / dz
    return res

def pressure_poisson(p, div, dx, dz):
    pn = np.empty_like(p)
    for _ in range(100):  # fixed number of Jacobi iterations for demonstration
        pn[:,:] = p[:,:]
        # Compute updates for interior rows 1 to Nz-2, periodic in x
        left = np.roll(pn, 1, axis=1)[1:-1, :]
        right = np.roll(pn, -1, axis=1)[1:-1, :]
        up = pn[2:, :]
        down = pn[:-2, :]
        p[1:-1, :] = ((left + right) / dx**2 + (up + down) / dz**2 - div[1:-1, :]) / (2/dx**2 + 2/dz**2)
        # Enforce Neumann BC in z for pressure (dp/dz = 0)
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
    return p

# Time-stepping loop
for n in range(nt):
    # Compute upwind derivatives for advection terms
    du_dx = ddx_upwind(u, u)
    du_dz = ddz_upwind(u, w)
    dw_dx = ddx_upwind(w, u)
    dw_dz = ddz_upwind(w, w)
    
    adv_u = u * du_dx + w * du_dz
    adv_w = u * dw_dx + w * dw_dz

    diff_u = nu * laplacian(u)
    diff_w = nu * laplacian(w)
    
    u_tilde = u + dt * (-adv_u + diff_u)
    w_tilde = w + dt * (-adv_w + diff_w + b)  # buoyancy force added in vertical momentum

    # Enforce no-slip BC for intermediate velocities (top and bottom)
    u_tilde[0, :] = 0.0
    u_tilde[-1, :] = 0.0
    w_tilde[0, :] = 0.0
    w_tilde[-1, :] = 0.0

    # Pressure correction step
    div = ddx_central(u_tilde) + ddz_central(w_tilde)
    rhs = div / dt
    p = pressure_poisson(p, rhs, dx, dz)
    
    dp_dx = ddx_central(p)
    dp_dz = ddz_central(p)
    
    u = u_tilde - dt * dp_dx
    w = w_tilde - dt * dp_dz

    # Enforce no-slip BC
    u[0, :] = 0.0
    u[-1, :] = 0.0
    w[0, :] = 0.0
    w[-1, :] = 0.0
    
    # Update buoyancy field b using upwind method for advection
    db_dx = ddx_upwind(b, u)
    db_dz = ddz_upwind(b, w)
    adv_b = u * db_dx + w * db_dz
    diff_b = kappa * laplacian(b)
    b = b + dt * (-adv_b + diff_b)
    
    # Enforce buoyancy boundary conditions:
    # Bottom (z=0): b = Lz and top (z=Lz): b = 0.
    b[0, :] = Lz
    b[-1, :] = 0.0

np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Rayleigh_Benard_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/w_2D_Rayleigh_Benard_Convection.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Rayleigh_Benard_Convection.npy', p)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/b_2D_Rayleigh_Benard_Convection.npy', b)