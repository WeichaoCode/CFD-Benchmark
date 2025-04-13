#!/usr/bin/env python3
import numpy as np

# Parameters
L = 10.0            # spatial domain length
T = 10.0            # final time
Nx = 256            # number of spatial grid points
dx = L / Nx
a = 1e-4            # diffusion coefficient
b = 2e-4            # dispersion coefficient
n_val = 20          # parameter for initial condition

# Time-stepping parameters
dt = 1e-4           # time step size
num_steps = int(T/dt)

# Spatial grid (periodic, endpoint excluded)
x = np.linspace(0, L, Nx, endpoint=False)

# Initial condition
u = (1/(2*n_val)) * np.log(1 + (np.cosh(n_val)**2)/np.cosh(n_val*(x - 0.2*L))**2)

# Precompute Fourier wave numbers for spectral derivatives
k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)

def compute_derivatives(u):
    u_hat = np.fft.fft(u)
    u_x   = np.fft.ifft(1j * k * u_hat).real
    u_xx  = np.fft.ifft(- (k**2) * u_hat).real
    u_xxx = np.fft.ifft(-1j * (k**3) * u_hat).real
    return u_x, u_xx, u_xxx

def rhs(u):
    u_x, u_xx, u_xxx = compute_derivatives(u)
    return - u * u_x + a * u_xx + b * u_xxx

# Time integration using RK4
for _ in range(num_steps):
    k1 = rhs(u)
    k2 = rhs(u + dt/2 * k1)
    k3 = rhs(u + dt/2 * k2)
    k4 = rhs(u + dt * k3)
    u += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

# Save the final solution as a 1D numpy array in u.npy
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_1D_KdV_Burgers_Equation.npy', u)