import numpy as np
from matplotlib import pyplot as plt

# Domain
Nx = 100
Ny = 100
X, Y = np.linspace(0, 1, Nx), np.linspace(0, 1, Ny)
dx, dy = X[1]-X[0], Y[1]-Y[0]

# Time 
T = 0.5
dt = 0.01
Nt = int(T/dt)

# Diffusivity (assumed constant)
alpha = 0.05

# Source function
def f(x, y, t):
    return np.exp(-t) * (2 * np.pi**2 - alpha) * np.sin(np.pi * x) * np.sin(np.pi * y)
    
# Function to calculate the new time step 
def timestep(u, alpha, dt, dx, dy, f):
    term = alpha*(np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0))/dx**2 + alpha*(np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1))/dy**2 + f
    return u + dt * term

# Initialize the solution
u = np.sin(np.pi * X)[:, None] * np.sin(np.pi * Y)[None, :]  # Broadcasting

# Time loop 
for n in range(Nt):
    u = timestep(u, alpha, dt, dx, dy, f(X[:, None], Y[None, :], n * dt))

# Exact solution
ue = np.exp(-T) * np.sin(np.pi * X)[:, None] * np.sin(np.pi * Y)[None, :]

# Plotting the numerical solution
plt.figure()
plt.contourf(X, Y, u.T)
plt.title("Numerical solution")
plt.colorbar()

# Plotting the exact solution
plt.figure()
plt.contourf(X, Y, ue.T)
plt.title("Exact solution")
plt.colorbar()

# Calculate and plot the error
error = np.abs(u - ue)
plt.figure()
plt.contourf(X, Y, error.T)
plt.title("Absolute Error")
plt.colorbar()

plt.show()