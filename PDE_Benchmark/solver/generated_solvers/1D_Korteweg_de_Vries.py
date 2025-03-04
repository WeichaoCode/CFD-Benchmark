import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# Define the MMS solution
def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

# Define the source term
def f(x, t):
    return np.exp(-t) * (np.pi**3 * np.sin(np.pi * x) - 3 * np.pi**2 * np.sin(np.pi * x) + 6 * np.pi * np.cos(np.pi * x))

# Define the grid resolution and time step size
N = 100
T = 1.0
dt = 0.01
x = np.linspace(0, 1, N)
t = np.arange(0, T, dt)

# Initialize the numerical solution
u = np.zeros((len(t), len(x)))

# Set the initial condition
u[0, :] = u_exact(x, 0)

# Define the wave numbers for the spectral method
k = fftpack.fftfreq(N, d=x[1]-x[0])
k = 2 * np.pi * k

# Time-stepping loop
for n in range(0, len(t)-1):
    # Compute the right-hand side of the PDE
    rhs = -6 * u[n, :] * np.real(fftpack.ifft(1j * k * fftpack.fft(u[n, :]))) + f(x, t[n])
    
    # Solve the PDE using the spectral method
    u_hat = fftpack.fft(u[n, :])
    u_hat = (u_hat + dt * fftpack.fft(rhs)) / (1 + dt * (1j * k)**3)
    u[n+1, :] = np.real(fftpack.ifft(u_hat))

# Compute the MMS solution
u_mms = u_exact(x[:, np.newaxis], t[np.newaxis, :])

# Compute the absolute error
error = np.abs(u - u_mms)

# Plot the numerical solution, the MMS solution, and the absolute error
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(u, extent=[0, T, 0, 1], origin='lower', aspect='auto')
plt.title('Numerical solution')
plt.colorbar()
plt.subplot(2, 2, 2)
plt.imshow(u_mms, extent=[0, T, 0, 1], origin='lower', aspect='auto')
plt.title('MMS solution')
plt.colorbar()
plt.subplot(2, 2, 3)
plt.imshow(error, extent=[0, T, 0, 1], origin='lower', aspect='auto')
plt.title('Absolute error')
plt.colorbar()
plt.tight_layout()
plt.show()