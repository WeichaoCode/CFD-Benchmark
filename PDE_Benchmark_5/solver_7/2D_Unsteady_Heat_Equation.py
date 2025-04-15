import numpy as np
import matplotlib.pyplot as plt

# grid size
Nx = 101
Ny = 101
Nt = 500

# spatial step
dx = 0.02
dy = 0.02

# time step
dt = 0.01

# thermal diffusivity
alpha = 0.0001

# source term parameters
Q0 = 200
sigma = 0.1

def source_term(x, y):
    return Q0 * np.exp(- (x**2 + y**2) / (2*sigma**2))

# temperature field
T = np.zeros((Nx, Ny))

# loop over grid
for i in range(Nx):
    for j in range(Ny):
        T[i,j] = source_term(i*dx, j*dy)

# heatmap of initial condition
plt.imshow(T, cmap='hot', interpolation='nearest')
plt.show()

for t in range(Nt):
    # temporary storage array
    T_new = np.zeros((Nx, Ny))

    # calculate new temperatures
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # explicit
            T_new[i,j] = T[i,j] + alpha*dt*((T[i+1,j]-2*T[i,j]+T[i-1,j])/dx**2 
                           + (T[i,j+1]-2*T[i,j]+T[i,j-1])/dy**2)

    # update temperature field
    T = T_new

    # plot every 50 timesteps
    if t % 50 == 0:
        plt.imshow(T, cmap='hot', interpolation='nearest')
        plt.show()