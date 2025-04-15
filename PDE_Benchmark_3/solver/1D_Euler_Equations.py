import numpy as np
import matplotlib.pyplot as plt

# Define FUNCTIONS
def compute_source_term(x, t):
    rho_src = -np.exp(-t)*np.sin(np.pi*x)
    u_src = -np.exp(-t)*np.cos(np.pi*x)
    p_src = -np.exp(-t)*(1+np.sin(np.pi*x))

    return rho_src, u_src, p_src

def compute_initial_and_boundary_conditions(x):
    rho = np.sin(np.pi*x)
    u = np.cos(np.pi*x)
    p = 1 + np.sin(np.pi*x)

    return rho, u, p

# SOLVER FUNCTION
def solve_1d_euler_equations(nx, dt, tf):
    dx = 1.0/nx
    x = np.linspace(dx/2, 1-dx/2, nx)  # spatial grid points
    
    # initial conditions
    rho, u, p = compute_initial_and_boundary_conditions(x)
    
    for n in range(int(tf/dt)):
        t = n*dt
        
        rho_src, u_src, p_src = compute_source_term(x, t)
        
        u_half = 0.5*(u[:-1] + u[1:])
        rho_new = rho - dt/dx*(rho*u)[1:] + (rho*u)[:-1] + dt*rho_src
        u_new = u - dt/dx*(rho*u*u + p)[1:] + (rho*u*u + p)[:-1] + dt*u_src
        p_new = p - dt/dx*((rho*u*u + 2*p)*u)[1:] + ((rho*u*u + 2*p)*u)[:-1] + dt*p_src

        rho, u, p = rho_new, u_new, p_new
    
    return x, rho, u, p

# MAIN
nx = 100  # number of grid points
dt = 0.001  # time step
tf = 1.0  # final time

x, rho, u, p = solve_1d_euler_equations(nx, dt, tf)

rho_exact, u_exact, p_exact = compute_initial_and_boundary_conditions(x)

plt.figure()
plt.plot(x, rho, label='Numerical')
plt.plot(x, rho_exact, label='Exact')
plt.legend()
plt.show()