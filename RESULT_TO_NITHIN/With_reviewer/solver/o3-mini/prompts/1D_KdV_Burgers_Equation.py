#!/usr/bin/env python3
import numpy as np

# Domain and simulation parameters
L = 10.0              # Spatial domain length
T = 10.0              # Final time
nx = 256              # Number of spatial points
dx = L / nx           # Spatial resolution

# Use a very small time step for stability
dt = 1e-4             
nt = int(T / dt)      # Number of time steps

# PDE parameters
a = 1e-4              # Diffusion coefficient (Burgers term)
b = 2e-4              # Dispersion coefficient (KdV term)

# Spatial grid
x = np.linspace(0, L, nx, endpoint=False)

# Initial condition parameters
n_param = 20
u0 = (1/(2*n_param)) * np.log1p((np.cosh(n_param)**2) / (np.cosh(n_param*(x - 0.2*L))**2))
u = u0.copy()

# Set up Fourier wave numbers for periodic domain
k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)

# Define the linear operator in Fourier space: Lhat = a*u_xx + b*u_xxx
Lhat = a*(1j*k)**2 + b*(1j*k)**3

# ETDRK4 precomputation
E   = np.exp(dt * Lhat)
E2  = np.exp(dt * Lhat/2.0)

M = 16  # Number of points for contour integrals
mu = 1j * np.pi * (np.arange(1, M+1) - 0.5) / M  # Contour points, shape (M,)
r = dt * Lhat.reshape(nx, 1)  # shape (nx, 1)
mu = mu.reshape(1, M)         # shape (1, M)
z = r + mu                    # shape (nx, M)

Q  = dt * np.mean((np.exp(z/2.0) - 1) / z, axis=1)
f1 = dt * np.mean((-4 - z + np.exp(z)*(4 - 3*z + z**2)) / z**3, axis=1)
f2 = dt * np.mean((2 + z + np.exp(z)*(-2 + z)) / z**3, axis=1)
f3 = dt * np.mean((-4 - 3*z - z**2 + np.exp(z)*(4 - z)) / z**3, axis=1)

# Dealiasing function using the 2/3 rule
def dealias(uhat):
    mask = np.abs(k) < (2 * np.pi / (3 * dx))
    return uhat * mask

# Nonlinear term: compute -u*u_x in Fourier space using pseudospectral method
def nonlinear(uhat_in):
    u_phys = np.fft.ifft(uhat_in)
    u_x_phys = np.fft.ifft(1j * k * uhat_in)
    # Clip values to prevent overflow during multiplication
    u_phys = np.clip(u_phys, -1e6, 1e6)
    u_x_phys = np.clip(u_x_phys, -1e6, 1e6)
    nonlinear_phys = - u_phys * u_x_phys
    nonlinear_hat = np.fft.fft(nonlinear_phys)
    return dealias(nonlinear_hat)

# Initialize Fourier transform of u
uhat = np.fft.fft(u)

# Time-stepping using ETDRK4 method
for it in range(nt):
    N1 = nonlinear(uhat)
    a_hat = E2 * uhat + Q * N1
    N2 = nonlinear(a_hat)
    b_hat = E2 * uhat + Q * N2
    N3 = nonlinear(b_hat)
    c_hat = E * uhat + f1 * N1 + 2 * f2 * (N2 + N3)
    N4 = nonlinear(c_hat)
    uhat = E * uhat + f1 * N1 + 2 * f2 * (N2 + N3) + f3 * N4

# Compute the final solution in physical space (real part)
u_final = np.real(np.fft.ifft(uhat))

# Save the final solution as a 1D NumPy array in 'u.npy'
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/u_final_1D_KdV_Burgers_Equation.npy', u_final)