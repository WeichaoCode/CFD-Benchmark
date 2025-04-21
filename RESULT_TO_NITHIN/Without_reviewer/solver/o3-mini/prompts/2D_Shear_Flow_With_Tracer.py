#!/usr/bin/env python3
import numpy as np

# Domain parameters
Lx = 1.0
Lz = 2.0         # from z=-1 to 1, so total length = 2
nx = 64
nz = 128
dx = Lx / nx
dz = Lz / nz

# Time parameters
T_final = 20.0
dt = 0.01
nt = int(T_final/dt)

# Physical parameters
nu = 1.0/(5e4)  # kinematic viscosity
D  = nu        # tracer diffusivity

# Create grid (x from 0 to 1, z from -1 to 1)
x = np.linspace(0, Lx, nx, endpoint=False)
z = np.linspace(-1, 1, nz, endpoint=False)
X, Z = np.meshgrid(x, z)

# Initial conditions
# u initial: function of z only.
u = 0.5 * (1.0 + np.tanh((Z - 0.5) / 0.1) - np.tanh((Z + 0.5) / 0.1))
# w initial: small sinusoidal perturbations localized around z=+/-0.5
amp = 1e-3
w = amp * np.sin(2*np.pi * X) * (np.exp(-((Z-0.5)**2)/(0.1**2)) + np.exp(-((Z+0.5)**2)/(0.1**2)))
# tracer field initial condition: same as u
s = u.copy()

# Pressure field: initialize to zero
p = np.zeros_like(u)

# Helper functions for periodic finite differences
def ddx(f):
    # central difference in x (axis=1)
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dx)

def ddz(f):
    # central difference in z (axis=0)
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dz)

def laplacian(f):
    return (np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) - 2*f)/(dx**2) + (np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) - 2*f)/(dz**2)

# Precompute Fourier wavenumbers for pressure Poisson solver (periodic domain)
kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
kz = 2 * np.pi * np.fft.fftfreq(nz, d=dz)
kx, kz = np.meshgrid(kx, kz)
Ksq = kx**2 + kz**2
# To avoid division by zero at the zero mode, set it to 1 (will reset later)
Ksq[0,0] = 1.0

# Time-stepping loop
for step in range(nt):
    # ------ Compute Intermediate Velocity (u*, w*) ------
    # Compute non-linear (advection) terms for u and w
    u_adv = u * ddx(u) + w * ddz(u)
    w_adv = u * ddx(w) + w * ddz(w)
    
    # Diffusion terms
    u_diff = nu * laplacian(u)
    w_diff = nu * laplacian(w)
    
    # Compute intermediate velocities (without pressure)
    u_star = u + dt * (- u_adv + u_diff)
    w_star = w + dt * (- w_adv + w_diff)
    
    # ------ Pressure Correction Step (Projection) ------
    # Compute divergence of intermediate velocity
    div_u_star = ddx(u_star) + ddz(w_star)
    
    # Solve Poisson equation: Laplacian(p) = (div_u_star)/dt
    rhs = div_u_star / dt
    rhs_hat = np.fft.fft2(rhs)
    p_hat = rhs_hat / (-Ksq)
    p_hat[0,0] = 0.0  # enforce zero mean pressure
    p = np.real(np.fft.ifft2(p_hat))
    
    # Compute pressure gradients
    dpdx = ddx(p)
    dpdz = ddz(p)
    
    # Correct velocities to enforce incompressibility
    u = u_star - dt * dpdx
    w = w_star - dt * dpdz

    # ------ Tracer Update ------
    s_adv = u * ddx(s) + w * ddz(s)
    s_diff = D * laplacian(s)
    s = s + dt * (- s_adv + s_diff)
    
    # (Optional) You can print progress every 500 steps
    if (step+1) % 500 == 0:
        print(f"Time step {step+1}/{nt}")

# Save final solutions into .npy files
# Save u (horizontal velocity), w (vertical velocity), p (pressure), s (tracer)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Shear_Flow_With_Tracer.npy', p)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)