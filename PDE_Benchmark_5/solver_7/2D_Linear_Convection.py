import numpy as np
import matplotlib.pyplot as plt

def linconv(nx, ny):
    Lx, Ly = 2, 2
    dx, dy = Lx/(nx-1), Ly/(ny-1)
    c = 1
    sigma = .2
    dt = sigma*dx
    nt = 20  # the number of timesteps to calculate

    # Initialize data structures
    u = np.ones((ny, nx))
    un = np.ones((ny, nx))

    # Initial condition
    u[int(.5/dy):int(1/dy + 1),int(.5/dx):int(1/dx + 1)] = 2 

    for n in range(nt + 1): 
        un = u.copy()
        u[1:, 1:] = un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) - (c * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    return u

nx_values = [11, 21, 41, 81]
ny_values = [11, 21, 41, 81]

for nx,ny in zip(nx_values, ny_values):
    u = linconv(nx, ny)
    plt.figure(figsize=(7, 7))
    plt.contourf(u)
    plt.colorbar()
    plt.title(f"Solution for {nx} x {ny} grid")
    plt.show()