import numpy as np
import matplotlib.pyplot as plt

# Define Manufactured Solution and Source Terms
def u_exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

def v_exact(x, y, t):
    return np.exp(-t) * np.cos(np.pi * x) * np.cos(np.pi * y)

def f_u(x, y, t, nu):
    left = np.exp(-t) * (np.sin(np.pi * x) * (-np.pi * np.sin(np.pi * y) * np.sin(np.pi * x) + np.cos(np.pi * y) * np.cos(np.pi * x)))
    right = 2 * nu * np.pi**2 * np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)
    return left - right

def f_v(x, y, t, nu):
    left = np.exp(-t) * (np.cos(np.pi * x) * (np.pi * np.sin(np.pi * y) * np.sin(np.pi * y) - np.cos(np.pi * y) * np.cos(np.pi * y)))
    right = 2 * nu * np.pi**2 * np.exp(-t) * np.cos(np.pi * x) * np.cos(np.pi * y)
    return left - right

# Set initial and boundary conditions, parameters
nu = 0.07
N = 101  # Number of grid points
x = y = np.linspace(0, 1, N)  # x, y coordinates
t_end = 0.5  # Time to integrate to
dt = 0.001  # Time step size

# Initialise solutions
u = np.zeros((N, N))
v = np.zeros((N, N))
u_new = np.zeros((N, N))
v_new = np.zeros((N, N))

# Set up X, Y grids for convenience, and initialise initial conditions
X, Y = np.meshgrid(x, y)
t = 0.0
u[:, :] = u_exact(X, Y, t)
v[:, :] = v_exact(X, Y, t)

# Explicit time stepping loop
while t < t_end:
    # Calculate source terms
    fu = f_u(X, Y, t, nu)
    fv = f_v(X, Y, t, nu)
    
    # Evaluate the derivatives at interior points
    for i in range(1, N-1):
        for j in range(1, N-1):
            u_x = (u[i+1, j] - u[i-1, j]) / (2 * dt)
            u_y = (u[i, j+1] - u[i, j-1]) / (2 * dt)
            v_x = (v[i+1, j] - v[i-1, j]) / (2 * dt)
            v_y = (v[i, j+1] - v[i, j-1]) / (2 * dt)
            lapl_u = (u[i+1, j] - 2 * u[i, j] + u[i-1, j]) / dt**2 + (u[i, j+1] - 2 * u[i, j] + u[i, j-1]) / dt**2
            lapl_v = (v[i+1, j] - 2 * v[i, j] + v[i-1, j]) / dt**2 + (v[i, j+1] - 2 * v[i, j] + v[i, j-1]) / dt**2
            
            u_new[i, j] = u[i, j] + dt * (-u[i, j] * u_x - v[i, j] * u_y + nu * lapl_u + fu[i, j])
            v_new[i, j] = v[i, j] + dt * (-u[i, j] * v_x - v[i, j] * v_y + nu * lapl_v + fv[i, j])
    
    # Update solution and time for next step
    u, u_new = u_new, u
    v, v_new = v_new, v
    t += dt

# Plot solution and exact solution for comparison
fig, axes = plt.subplots(ncols=2)
c = axes[0].contourf(X, Y, u)
fig.colorbar(c, ax=axes[0])
axes[0].set_title('Numerical solution')
c = axes[1].contourf(X, Y, u_exact(X, Y, t_end))
fig.colorbar(c, ax=axes[1])
axes[1].set_title('Exact solution')
plt.show()