import numpy as np

# Parameters
c = 1.0
epsilon_values = [0, 5e-4]
x_start, x_end = -5, 5
Nx = 101
dx = (x_end - x_start) / (Nx - 1)
x = np.linspace(x_start, x_end, Nx)

# Initial condition
u_initial = np.exp(-x**2)

# Time-stepping parameters
CFL = 0.5
dt = CFL * dx / c
t_final = 2.0
Nt = int(t_final / dt)

# Save values
save_values = ['u_epsilon_0', 'u_epsilon_5e-4']

# Predictor-Corrector method
def predictor_corrector(u, epsilon):
    u_new = np.copy(u)
    for n in range(Nt):
        # Predictor step
        u_star = np.copy(u)
        for i in range(1, Nx-1):
            u_star[i] = u[i] - dt * c * (u[i+1] - u[i-1]) / (2*dx) + dt * epsilon * (u[i+1] - 2*u[i] + u[i-1]) / (dx**2)
        
        # Periodic boundary conditions
        u_star[0] = u_star[-2]
        u_star[-1] = u_star[1]
        
        # Corrector step
        for i in range(1, Nx-1):
            u_new[i] = 0.5 * (u[i] + u_star[i] - dt * c * (u_star[i+1] - u_star[i-1]) / (2*dx) + dt * epsilon * (u_star[i+1] - 2*u_star[i] + u_star[i-1]) / (dx**2))
        
        # Periodic boundary conditions
        u_new[0] = u_new[-2]
        u_new[-1] = u_new[1]
        
        # Update solution
        u[:] = u_new[:]
    
    return u

# Solve for each epsilon value
for epsilon, save_name in zip(epsilon_values, save_values):
    u = np.copy(u_initial)
    u_final = predictor_corrector(u, epsilon)
    np.save(save_name, u_final)