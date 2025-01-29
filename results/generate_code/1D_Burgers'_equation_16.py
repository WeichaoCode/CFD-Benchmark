import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.07
L = 2.0
Nx = 101
dx = L/(Nx-1)
x = np.linspace(0, L, Nx)
dt = 2/500
nt = 500

# Initial condition function
def phi(x, nu):
    return np.exp(-x**2/(4*nu)) + np.exp(-(x-2*np.pi)**2/(4*nu))

def phi_x(x, nu):
    return (-x/(2*nu))*np.exp(-x**2/(4*nu)) + (-(x-2*np.pi)/(2*nu))*np.exp(-(x-2*np.pi)**2/(4*nu))

# Set initial condition
u = np.zeros(Nx)
u = -2*nu*phi_x(x,nu)/phi(x,nu) + 4

# Time stepping
def spatial_deriv(u):
    # First derivative (central difference)
    du_dx = np.zeros_like(u)
    du_dx[1:-1] = (u[2:] - u[:-2])/(2*dx)
    du_dx[0] = (u[1] - u[-1])/(2*dx)
    du_dx[-1] = (u[0] - u[-2])/(2*dx)
    
    # Second derivative
    d2u_dx2 = np.zeros_like(u)
    d2u_dx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2])/dx**2
    d2u_dx2[0] = (u[1] - 2*u[0] + u[-1])/dx**2
    d2u_dx2[-1] = (u[0] - 2*u[-1] + u[-2])/dx**2
    
    return du_dx, d2u_dx2

# RK4 time stepping
def rhs(u):
    du_dx, d2u_dx2 = spatial_deriv(u)
    return -u*du_dx + nu*d2u_dx2

def rk4_step(u, dt):
    k1 = rhs(u)
    k2 = rhs(u + 0.5*dt*k1)
    k3 = rhs(u + 0.5*dt*k2)
    k4 = rhs(u + dt*k3)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# Time evolution
t = 0
u_sol = np.zeros((nt+1, Nx))
u_sol[0] = u

for n in range(nt):
    u = rk4_step(u, dt)
    t += dt
    u_sol[n+1] = u

# Plot results
plt.figure(figsize=(10,6))
plt.plot(x, u_sol[0], 'b-', label='t=0')
plt.plot(x, u_sol[-1], 'r--', label=f't={t}')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True)
plt.show()