#!/usr/bin/env python3
import numpy as np

# Parameters and domain
nx = 64
nz = 64
Lx = 1.0
Lz = 2.0  # z in [-1, 1]
dx = Lx / nx
dz = Lz / nz
x = np.linspace(0, Lx, nx, endpoint=False)
z = np.linspace(-1, 1, nz, endpoint=False)
X, Z = np.meshgrid(x, z)

t_end = 20.0
dt = 0.001
nt = int(t_end/dt)

# Physical parameters
nu = 1.0/5e4  # kinematic viscosity
D = nu        # tracer diffusivity

# Initial conditions for u, w, and tracer s
u = 0.5 * (1.0 + np.tanh((Z - 0.5)/0.1) - np.tanh((Z + 0.5)/0.1))
w = 0.01 * np.sin(2*np.pi*X) * (np.exp(-((Z - 0.5)/0.1)**2) + np.exp(-((Z + 0.5)/0.1)**2))
s = u.copy()  # tracer initial condition

# Pressure field initialization
p = np.zeros_like(u)

# Helper functions for periodic finite differences
def upwind_dx(f, vel):
    # Upwind difference in x-direction
    f_roll_forward = np.roll(f, -1, axis=1)
    f_roll_backward = np.roll(f, 1, axis=1)
    dfdx = np.where(vel >= 0, (f - f_roll_backward) / dx, (f_roll_forward - f) / dx)
    return dfdx

def upwind_dz(f, vel):
    # Upwind difference in z-direction
    f_roll_forward = np.roll(f, -1, axis=0)
    f_roll_backward = np.roll(f, 1, axis=0)
    dfdz = np.where(vel >= 0, (f - f_roll_backward) / dz, (f_roll_forward - f) / dz)
    return dfdz

def central_laplace(f):
    return (np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) - 2*f) / (dx**2) + \
           (np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) - 2*f) / (dz**2)

def central_deriv_x(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dx)

def central_deriv_z(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dz)

# Precompute wave numbers for FFT-based Poisson solver for pressure
kx = 2*np.pi * np.fft.fftfreq(nx, d=dx)
kz = 2*np.pi * np.fft.fftfreq(nz, d=dz)
KX, KZ = np.meshgrid(kx, kz)
K2 = KX**2 + KZ**2
K2[0, 0] = 1.0  # avoid division by zero

# Time integration loop (explicit Euler with projection method)
for tstep in range(nt):
    # Compute advective derivatives using upwind differences
    du_dx = upwind_dx(u, u)
    du_dz = upwind_dz(u, w)
    dw_dx = upwind_dx(w, u)
    dw_dz = upwind_dz(w, w)
    
    adv_u = u * du_dx + w * du_dz
    adv_w = u * dw_dx + w * dw_dz

    # Viscous diffusion terms (central differencing)
    diff_u = nu * central_laplace(u)
    diff_w = nu * central_laplace(w)
    
    # Predictor step: intermediate velocity without pressure gradient
    u_star = u + dt * (-adv_u + diff_u)
    w_star = w + dt * (-adv_w + diff_w)
    
    # Compute divergence of the intermediate velocity field (central differences)
    div_star = central_deriv_x(u_star) + central_deriv_z(w_star)
    
    # Pressure Poisson equation: Laplace(p) = (1/dt)*div(u_star)
    div_star_hat = np.fft.fftn(div_star)
    p_hat = div_star_hat / (-K2 * dt)
    p_hat[0, 0] = 0.0  # set the zero mode to zero (gauge condition)
    p = np.real(np.fft.ifftn(p_hat))
    
    # Pressure gradient (computed spectrally)
    dp_dx = np.real(np.fft.ifftn(1j*KX*p_hat))
    dp_dz = np.real(np.fft.ifftn(1j*KZ*p_hat))
    
    # Correct velocities (projection step)
    u = u_star - dt * dp_dx
    w = w_star - dt * dp_dz
    
    # Update tracer s using upwind scheme for advection and central for diffusion
    ds_dx = upwind_dx(s, u)
    ds_dz = upwind_dz(s, w)
    adv_s = u * ds_dx + w * ds_dz
    diff_s = D * central_laplace(s)
    s = s + dt * (-adv_s + diff_s)

# Save final solutions as .npy files (2D arrays)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Shear_Flow_With_Tracer.npy', p)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)