import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

# Parameters
N = 64  # Number of grid points in each direction
L = 1.0  # Length of the domain
nu = 0.001  # Kinematic viscosity
dt = 0.001  # Time step
T = 1.0  # Final time

# Grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Initial conditions
psi = np.zeros((N, N))
omega = np.zeros((N, N))

# Vortex initialization
sigma = 0.05
omega = np.exp(-((X - 0.5)**2 + (Y - 0.25)**2) / (2 * sigma**2)) - np.exp(-((X - 0.5)**2 + (Y - 0.75)**2) / (2 * sigma**2))

# Functions for derivatives
def laplacian(phi):
    phi_xx = (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)) / dx**2
    phi_yy = (np.roll(phi, -1, axis=0) - 2 * phi + np.roll(phi, 1, axis=0)) / dy**2
    return phi_xx + phi_yy

def solve_poisson(omega):
    # Solve Poisson equation using FFT
    omega_hat = fft.fft2(omega)
    kx = 2 * np.pi * fft.fftfreq(N, d=dx)
    ky = 2 * np.pi * fft.fftfreq(N, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k_squared = KX**2 + KY**2
    k_squared[0, 0] = 1.0  # Avoid division by zero
    psi_hat = -omega_hat / k_squared
    psi_hat[0, 0] = 0  # Set DC component to zero (Dirichlet BC)
    psi = np.real(fft.ifft2(psi_hat))
    return psi

# Time loop
t = 0
while t < T:
    # Calculate velocities
    u = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * dy)
    v = -(np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * dx)

    # Advection term
    advection = u * (np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1)) / (2 * dx) + \
                v * (np.roll(omega, -1, axis=0) - np.roll(omega, 1, axis=0)) / (2 * dy)

    # Diffusion term
    diffusion = nu * laplacian(omega)

    # Update vorticity
    omega = omega + dt * (-advection + diffusion)

    # Boundary conditions for omega (approximation from interior)
    omega[0, :] = omega[1, :]
    omega[-1, :] = omega[-2, :]

    # Solve for streamfunction
    psi = solve_poisson(omega)

    t += dt

# Save the final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/psi_Vortex_Roll_Up.npy', psi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/omega_Vortex_Roll_Up.npy', omega)