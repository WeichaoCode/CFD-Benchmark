import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# Define the MMS solution
def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

# Define the source term
def f(x, t):
    return np.exp(-t) * ((np.pi**2 - 1) * np.sin(np.pi * x) - np.pi**2 * np.sin(2 * np.pi * x))

# Define the initial condition
def u_initial(x):
    return u_exact(x, 0)

# Define the spatial grid
L = 1.0  # length of the domain
N = 100  # number of grid points
dx = L / N  # grid resolution
x = np.linspace(0, L, N)

# Define the time grid
T = 1.0  # final time
dt = 0.001  # time step size
Nt = int(T / dt)  # number of time steps

# Initialize the solution array
u = np.empty((Nt, N))
u[0, :] = u_initial(x)

# Time-stepping loop
for n in range(Nt - 1):
    # Compute the source term
    F = f(x, n * dt)

    # Compute the spatial derivatives using FFT
    u_hat = fft(u[n, :])
    k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    u_xx = ifft(-k**2 * u_hat).real
    u_xxxx = ifft(k**4 * u_hat).real
    u_x = ifft(1j * k * u_hat).real
    u_xx = ifft(-k**2 * u_hat).real

    # Update the solution using an explicit scheme
    u[n + 1, :] = u[n, :] + dt * (u_xx + u_xxxx + 0.5 * u_x**2 + F)

# Compute the exact solution
u_exact_sol = u_exact(x[:, np.newaxis], dt * np.arange(Nt))

# Compute the absolute error
error = np.abs(u - u_exact_sol)

# Plot the numerical solution, exact solution, and error
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(u, extent=[0, T, 0, L], origin='lower', aspect='auto')
plt.title('Numerical solution')
plt.colorbar()
plt.subplot(2, 2, 2)
plt.imshow(u_exact_sol, extent=[0, T, 0, L], origin='lower', aspect='auto')
plt.title('Exact solution')
plt.colorbar()
plt.subplot(2, 2, 3)
plt.imshow(error, extent=[0, T, 0, L], origin='lower', aspect='auto')
plt.title('Absolute error')
plt.colorbar()
plt.tight_layout()
plt.show()