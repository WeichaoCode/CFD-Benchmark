import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant
from scipy.fftpack import fft, ifft

# Define the Korteweg–de Vries equation
def KdV(u, t, f):
    ue = np.r_[0.5*(3*u[-1]+4*u[0]-u[1]), u, 0.5*(3*u[-1]+4*u[0]-u[1])]
    ux = (ue[2:]-ue[:-2]) / (2*dx)
    uxx = (ue[2:] - 2*ue[1:-1] + ue[:-2]) / dx**2
    uxxx = (ue[2:] - 3*ue[1:-1] + 3*ue[:-2] - ue[:-3]) / dx**3
    return -6*u*ux - uxxx + f(x,t)

# MMS
def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi*x)

# Source term from MMS
def f(x, t):
    return np.exp(-t)*np.sin(np.pi*x) * (1+0.5*np.pi**2) - np.pi*np.cos(np.pi*x)*np.exp(-t)

# Parameters
L, T = 1.0, 1.0  # domain length and final time
Nx, Nt = 100, 100  # spatial and time discretization

# Space and time discretization
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]
t = np.linspace(0, T, Nt)
dt = t[1] - t[0]

# Initial condition
u0 = u_exact(x, 0)

# Prepare matrix for Crank-Nicolson
K = circulant([-2, 1]+[0]*(Nx-3)+[1]).T / dx**2
I = np.eye(Nx)
A = I - dt/2 * K

# Initialize arrays
u_old = u0.copy()
u_new = np.zeros_like(u_old)

# Time-stepping loop
for n in range(1, Nt):
    B = I + dt/2 * K
    rhs = B.dot(u_old) + dt*KdV(u_old, t[n-1], f)
    u_new = np.linalg.solve(A, rhs)
    u_old = u_new.copy()

# MMS and absolute error
u_mms = u_exact(x, T)
error = np.abs(u_new - u_mms)

# Plotting numerical solution, MMS and error
plt.subplot(1, 3, 1)
plt.plot(x, u_new, label='Numerical')
plt.plot(x, u_mms, label='MMS')
plt.legend()
plt.title('Solution at T')

plt.subplot(1, 3, 2)
plt.plot(x, u_mms)
plt.title('MMS at T')

plt.subplot(1, 3, 3)
plt.plot(x, error)
plt.title('Error at T')

plt.tight_layout()
plt.show()