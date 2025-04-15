import numpy as np
import matplotlib.pyplot as plt

def linear_convection(nx, L=2.0, c=1.0, dt=0.025, nt=20):
    dx = L / (nx - 1)
    
    # set up grid
    x = np.linspace(0, L, nx)
    u = np.ones(nx)
    
    # set initial condition
    mask = np.where(np.logical_and(x >= 0.5, x <= 1))
    u[mask] = 2

    # initialize a temporary array
    un = np.ones(nx)
    
    for n in range(nt):
        un = u.copy()
        for i in range(1, nx - 1):
            # apply FDM on the convection equation
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
            
    return u, x

# define grid size
nx = 81

for nt in [20, 50, 100]:
    u, x = linear_convection(nx, nt=nt)
    plt.plot(x, u, label="nt = {}".format(nt))

plt.xlabel('Grid Spacing [x]')
plt.ylabel('Wave Amplitude [u]')
plt.title('1D Linear Convection')
plt.legend()
plt.show()