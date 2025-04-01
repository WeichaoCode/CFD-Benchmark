import numpy as np

# Problem parameters
Lx = 10.0  # Domain length
Nx = 101   # Number of spatial points
c = 1.0    # Convection speed
epsilon = 5e-4  # Damping factor

# Domain setup
x = np.linspace(-5, 5, Nx)
dx = x[1] - x[0]

# Time parameters
t_final = 2.0  # Final time
dt = 0.1 * dx / np.abs(c)  # CFL condition
Nt = int(t_final / dt)

# Initialize solution array
u = np.exp(-x**2)  # Initial condition

# Adams-Bashforth method setup
u_prev = u.copy()
u_prev2 = u.copy()

# Time-stepping loop
for n in range(Nt):
    # First step: Explicit Euler 
    if n == 0:
        # Compute spatial derivatives with periodic BC
        u_x = np.zeros_like(u)
        u_xx = np.zeros_like(u)
        
        # Central difference for first derivatives
        u_x[1:-1] = (u[2:] - u[:-2]) / (2*dx)
        u_x[0] = (u[1] - u[-1]) / (2*dx)  # Periodic BC
        u_x[-1] = (u[0] - u[-2]) / (2*dx)  # Periodic BC
        
        # Central difference for second derivatives
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        u_xx[0] = (u[1] - 2*u[0] + u[-1]) / (dx**2)  # Periodic BC
        u_xx[-1] = (u[0] - 2*u[-1] + u[-2]) / (dx**2)  # Periodic BC
        
        # Explicit Euler update
        u_new = u + dt * (-c * u_x + epsilon * u_xx)
        
        # Update history
        u_prev2 = u.copy()
        u_prev = u_new.copy()
        u = u_new.copy()
    
    # Adams-Bashforth method for subsequent steps
    else:
        # Compute spatial derivatives with periodic BC
        u_x = np.zeros_like(u)
        u_xx = np.zeros_like(u)
        
        # Central difference for first derivatives
        u_x[1:-1] = (u[2:] - u[:-2]) / (2*dx)
        u_x[0] = (u[1] - u[-1]) / (2*dx)  # Periodic BC
        u_x[-1] = (u[0] - u[-2]) / (2*dx)  # Periodic BC
        
        # Central difference for second derivatives
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        u_xx[0] = (u[1] - 2*u[0] + u[-1]) / (dx**2)  # Periodic BC
        u_xx[-1] = (u[0] - 2*u[-1] + u[-2]) / (dx**2)  # Periodic BC
        
        # Compute spatial derivatives for previous time steps
        u_prev_x = np.zeros_like(u_prev)
        u_prev_x[1:-1] = (u_prev[2:] - u_prev[:-2]) / (2*dx)
        u_prev_x[0] = (u_prev[1] - u_prev[-1]) / (2*dx)
        u_prev_x[-1] = (u_prev[0] - u_prev[-2]) / (2*dx)
        
        # Adams-Bashforth update
        u_new = u + dt * (1.5 * (-c * u_x + epsilon * u_xx) - 0.5 * (-c * u_prev_x + epsilon * u_xx))
        
        # Update history
        u_prev2 = u_prev.copy()
        u_prev = u.copy()
        u = u_new.copy()

# Save final solution
save_values = ['u']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_1D_Linear_Convection_adams.npy', u)