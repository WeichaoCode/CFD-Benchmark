#!/usr/bin/env python3
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import os

# Parameters
nx = 128
nz = 128
Lx = 1.0
Lz = 2.0
dx = Lx / nx
dz = Lz / nz

dt = 1e-3     # reduced time step for stability
t_final = 20.0
nt = int(t_final/dt)

nu = 1.0/50000.0    # kinematic viscosity
D  = nu             # tracer diffusivity

# Create grid
x = np.linspace(0, Lx, nx, endpoint=False)
z = np.linspace(-Lz/2, Lz/2, nz, endpoint=False)
X, Z = np.meshgrid(x, z, indexing='ij')

# Initial conditions
u = 0.5 * (1.0 + np.tanh((Z - 0.5)/0.1) - np.tanh((Z + 0.5)/0.1))
w = 1e-3 * np.sin(2*np.pi*X) * (np.exp(-((Z - 0.5)/0.1)**2) + np.exp(-((Z + 0.5)/0.1)**2))
s = np.copy(u)

# Setup wave numbers for spectral differentiation
kx = 2 * np.pi * fftfreq(nx, d=dx)  # shape (nx,)
kz = 2 * np.pi * fftfreq(nz, d=dz)  # shape (nz,)
KX, KZ = np.meshgrid(kx, kz, indexing='ij')
K2 = KX**2 + KZ**2
K2[0,0] = 1.0  # prevent division by zero in pressure solve

# Create dealiasing mask (2/3 rule)
kx_max = np.max(np.abs(kx))
kz_max = np.max(np.abs(kz))
dealias_mask = ((np.abs(KX) < (2.0/3.0)*kx_max) & (np.abs(KZ) < (2.0/3.0)*kz_max))

def dealias(f_hat):
    return f_hat * dealias_mask

# Functions for spectral derivatives and Laplacian using dealiasing
def deriv(f, K):
    f_hat = fftn(f)
    f_hat = dealias(f_hat)
    df_hat = 1j * K * f_hat
    return np.real(ifftn(df_hat))

def laplacian(f):
    f_hat = fftn(f)
    f_hat = dealias(f_hat)
    lap_hat = -K2 * f_hat
    return np.real(ifftn(lap_hat))

# Time-stepping loop using Forward Euler with projection method
for step in range(nt):
    # Compute spatial derivatives for velocity components
    u_x = deriv(u, KX)
    u_z = deriv(u, KZ)
    w_x = deriv(w, KX)
    w_z = deriv(w, KZ)
    
    # Nonlinear advection terms for velocity
    nonlin_u = u * u_x + w * u_z
    nonlin_w = u * w_x + w * w_z

    # Diffusion terms (Laplacian)
    lap_u = laplacian(u)
    lap_w = laplacian(w)
    
    # Compute intermediate velocities (without pressure)
    u_star = u + dt * (-nonlin_u + nu * lap_u)
    w_star = w + dt * (-nonlin_w + nu * lap_w)
    
    # Enforce incompressibility: Solve Poisson eqn for pressure
    div_star = deriv(u_star, KX) + deriv(w_star, KZ)
    div_star_hat = fftn(div_star)
    # Pressure solve in spectral space
    p_hat = div_star_hat / (-K2 * (1.0/dt))
    p_hat[0, 0] = 0.0  # set arbitrary constant for pressure
    p = np.real(ifftn(p_hat))
    
    # Compute pressure gradients
    p_x = deriv(p, KX)
    p_z = deriv(p, KZ)
    
    # Correct velocities to enforce incompressibility
    u = u_star - dt * p_x
    w = w_star - dt * p_z
    
    # Tracer equation
    s_x = deriv(s, KX)
    s_z = deriv(s, KZ)
    nonlin_s = u * s_x + w * s_z
    lap_s = laplacian(s)
    s = s + dt * (-nonlin_s + D * lap_s)

# Save final solutions as .npy files (2D arrays)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)