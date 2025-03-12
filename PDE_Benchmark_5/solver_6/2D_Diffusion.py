import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
Lx, Ly = 2.0, 2.0
nx, ny = 41, 41
nt = 500
nu = 0.01
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
sigma = 0.25
dt = sigma * dx * dy / nu

# Discretize the domain
x = np.linspace(0, Lx, num=nx)
y = np.linspace(0, Ly, num=ny)
u0 = np.ones((ny, nx))
u0[int(0.5 / dy):int(1 / dy + 1),int(0.5 / dx):int(1 / dx + 1)] = 2

# Function that implements the finite difference scheme
def diffusion(u0, sigma, dt, dx, dy, nt, nu):
    u_hist = [u0]
    u = u0.copy()
    for n in range(nt):
        un = u.copy()
        # Update the solution at interior points
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] + nu * dt / dx**2 *
                         (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                         nu * dt / dy**2 * 
                         (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))
        # Apply boundary conditions
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
        u_hist.append(u.copy())
    return u_hist

# Solve the equation and animate
u_hist = diffusion(u0, sigma, dt, dx, dy, nt, nu)

# Function to plot result
def plot(u_hist, x, y, nt):
    fig = plt.figure(figsize=(5.0, 5.0))
    plt.xlabel('y')
    plt.ylabel('x')
    colorinterpolation = 20
    colourMap = plt.cm.jet 
    plt.contourf(y, x, u_hist[nt], colorinterpolation, cmap=colourMap)
    plt.colorbar()
    plt.show()
    
plot(u_hist, x, y, nt = 100) # change nt to visualize at different time steps