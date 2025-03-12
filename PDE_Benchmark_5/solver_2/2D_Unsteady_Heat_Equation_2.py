import numpy as np
import matplotlib.pyplot as plt

# Define grid parameters
Nx, Ny, Nt = 50, 50, 500
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
t = np.linspace(0, 0.1, Nt)

dx, dy, dt = x[1]-x[0], y[1]-y[0], t[1]-t[0]

alpha = 0.1 #thermal diffusivity
r = alpha*dt/dx/dy # Fourier number

# Initialize temperature field
T = np.zeros([Nx, Ny, Nt])

#Boundary conditions
T[:, 0, :] = 0   # y = -1
T[:, -1, :] = 0  # y =  1
T[0, :, :] = 0   # x = -1
T[-1, :, :] = 0  # x =  1

# Dufort-Frankel method
for k in range(0,Nt-1): 
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            TX = T[i-1,j,k] + T[i+1,j,k]
            TY = T[i,j-1,k] + T[i,j+1,k]
            T0 = T[i,j,k-1] 
            T[i,j,k] = ((1-2*r)*T0 + r*TX + r*TY )/(1+2*r)

# Visualize the result
plt.figure()
plt.imshow(T[:,:,Nt-1], cmap='hot', interpolation='nearest')
plt.show()