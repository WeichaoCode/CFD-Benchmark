import numpy as np
import matplotlib.pyplot as plt

# Computational Domain
Lx, Ly = 2.0, 2.0 
nx, ny = 101, 101 
x = np.linspace(0,Lx,nx)
y = np.linspace(0,Ly,ny)

# Parameters
T=1.0
dt = 0.01 
nt = int(T/dt)

# Initialization
u = np.ones((nx, ny))
v = np.ones((nx, ny))

dx = x[1]-x[0] 
dy = y[1]-y[0] 

# Set initial condition
u[int(.5/dx):int(1/dx),int(.5/dy):int(1/dy)] = 2
v[int(.5/dx):int(1/dx),int(.5/dy):int(1/dy)] = 2

# Boundary Conditions
u[0,:] = 1
u[-1,:] = 1
u[:,0] = 1
u[:,-1] = 1

v[0,:] = 1
v[-1,:] = 1
v[:,0] = 1
v[:,-1] = 1

for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Finite Difference Scheme
    u[1:,1:] = un[1:,1:] - dt/dx*un[1:,1:]*(un[1:,1:]-un[:-1,1:]) - dt/dy*vn[1:,1:]*(un[1:,1:]-un[1:,:-1])
    v[1:,1:] = vn[1:,1:] - dt/dx*un[1:,1:]*(vn[1:,1:]-vn[:-1,1:]) - dt/dy*vn[1:,1:]*(vn[1:,1:]-vn[1:,:-1])
    
    # Boundary Conditions applied after internal updates to ensure wave does not move out of the computation domain
    u[0,:] = 1
    u[-1,:] = 1
    u[:,0] = 1
    u[:,-1] = 1

    v[0,:] = 1
    v[-1,:] = 1
    v[:,0] = 1
    v[:,-1] = 1

# Quiver plot
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots()

q = ax.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])
plt.show()