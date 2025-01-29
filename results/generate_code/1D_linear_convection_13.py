import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 101  # Number of grid points
nt = 500  # Number of time steps
c = 1.0   # Wave speed
L = 2.0   # Domain length
T = 2.0   # Final time

# Grid
dx = L/(nx-1)
x = np.linspace(0, L, nx)
dt = T/nt
t = np.linspace(0, T, nt+1)

# Initial condition
u = np.ones(nx)
u[(x>=0.5) & (x<=1.0)] = 2.0

# Solution array
sol = np.zeros((nt+1, nx))
sol[0,:] = u

# Time integration using RK4
def rhs(u):
    # First order upwind for spatial derivative
    dudx = np.zeros_like(u)
    dudx[1:] = (u[1:] - u[:-1])/dx
    dudx[0] = (u[0] - u[-1])/dx
    return -c*dudx

for n in range(nt):
    # RK4 steps
    k1 = rhs(u)
    k2 = rhs(u + 0.5*dt*k1)
    k3 = rhs(u + 0.5*dt*k2)
    k4 = rhs(u + dt*k3)
    
    # Update solution
    u = u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    sol[n+1,:] = u

# Plot results
plt.figure(figsize=(10,6))
plt.plot(x, sol[0,:], 'b-', label='t=0')
plt.plot(x, sol[int(nt/4),:], 'g-', label=f't={T/4}')
plt.plot(x, sol[int(nt/2),:], 'r-', label=f't={T/2}')
plt.plot(x, sol[-1,:], 'k-', label=f't={T}')
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Linear Convection')
plt.legend()
plt.grid(True)
plt.show()