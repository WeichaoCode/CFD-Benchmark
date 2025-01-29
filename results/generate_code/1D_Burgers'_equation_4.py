import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.07
L = 2.0
Nx = 101
dx = L/(Nx-1)
x = np.linspace(0, L, Nx)
T = 2.0
nt = 500
dt = T/nt

# Initial condition
def phi(x, nu):
    return np.exp(-x**2/(4*nu)) + np.exp(-(x-2*np.pi)**2/(4*nu))

def phi_x(x, nu):
    return -x/(2*nu)*np.exp(-x**2/(4*nu)) + \
           -(x-2*np.pi)/(2*nu)*np.exp(-(x-2*np.pi)**2/(4*nu))

u = np.zeros(Nx)
for i in range(Nx):
    u[i] = -2*nu/phi(x[i],nu)*phi_x(x[i],nu) + 4

# Time stepping
u_n = np.copy(u)
u_np1 = np.copy(u)

for n in range(nt):
    u_n = np.copy(u)
    
    # Periodic boundary conditions
    for i in range(1, Nx-1):
        # Space derivatives
        u_x = (u_n[i+1] - u_n[i-1])/(2*dx)
        u_xx = (u_n[i+1] - 2*u_n[i] + u_n[i-1])/dx**2
        
        # Time integration
        u_np1[i] = u_n[i] - dt*u_n[i]*u_x + nu*dt*u_xx
    
    # Update periodic boundaries
    u_np1[0] = u_np1[-2]
    u_np1[-1] = u_np1[1]
    
    u = np.copy(u_np1)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(x, u, 'b-', label='Final solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Burgers Equation')
plt.legend()
plt.grid(True)
plt.show()