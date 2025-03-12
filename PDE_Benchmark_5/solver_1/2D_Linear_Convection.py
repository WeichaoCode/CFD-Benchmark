import numpy as np
import matplotlib.pyplot as plt

def linearconv(nx):
    dx = 2 / (nx - 1)
    nt = 20    
    dt = 0.025  
    c = 1

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, nx)
    u = np.ones((nx, nx)) 
    un = np.ones((nx, nx))
    
    # Set initial condition
    u[int(.5 / dx):int(1 / dx + 1), int(.5 / dx):int(1 / dx + 1)] = 2  

    for n in range(nt + 1): 
        un = u.copy()
        row, col = u.shape
        for j in range(1, row):
            for i in range(1, col):
                u[j, i] = (un[j, i] - (c * dt / dx * (un[j, i] - un[j, i - 1])) -
                                      (c * dt / dx * (un[j, i] - un[j - 1, i])))
                u[0, :] = 1
                u[-1, :] = 1
                u[:, 0] = 1
                u[:, -1] = 1

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, nx)

    plt.contourf(x, y, u)
    plt.colorbar()
    plt.show()

linearconv(81)