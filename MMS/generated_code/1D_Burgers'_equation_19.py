import math
import matplotlib.pyplot as plt

# Parameters
nx = 100  # number of spatial points
nt = 5000  # number of time steps
dx = 2.0 / (nx - 1)  # spatial step size
dt = 2.0 / nt  # time step size
nu = 0.07  # viscosity

# Initialize arrays
x = [i * dx for i in range(nx)]
u = [math.sin(math.pi * xi) for xi in x]  # initial condition
u_new = [0] * nx

# Stability check (von Neumann analysis)
stability_param = nu * dt / (dx * dx)
if stability_param > 0.5:
    print(f"Warning: Solution might be unstable! Stability parameter = {stability_param}")
    print("Please reduce dt or increase dx")

# Function for the forcing term
def forcing_term(x, t):
    return (-math.pi * math.pi * nu * math.exp(-t) * math.sin(math.pi * x) + 
            math.exp(-t) * math.sin(math.pi * x) - 
            math.pi * math.exp(-2*t) * math.sin(math.pi * x) * math.cos(math.pi * x))

# Time stepping
t = 0
plot_times = [0, 0.5, 1.0, 2.0]  # Times at which to plot
solutions_to_plot = []

while t <= 2.0:
    # Store solutions at plot times
    if abs(t - plot_times[0]) < dt/2:
        solutions_to_plot.append((t, u[:]))
        plot_times.pop(0)
        if not plot_times:
            plot_times.append(float('inf'))
    
    # FTCS scheme
    for i in range(1, nx-1):
        # Spatial derivatives
        du_dx = (u[i+1] - u[i-1]) / (2*dx)
        d2u_dx2 = (u[i+1] - 2*u[i] + u[i-1]) / (dx*dx)
        
        # Update solution
        u_new[i] = (u[i] + 
                    dt * (-u[i] * du_dx + 
                          nu * d2u_dx2 + 
                          forcing_term(x[i], t)))
    
    # Apply boundary conditions
    u_new[0] = 0
    u_new[-1] = 0
    
    # Update solution and time
    u = u_new[:]
    t += dt

# Plotting
plt.figure(figsize=(10, 6))
for t, sol in solutions_to_plot:
    plt.plot(x, sol, label=f't = {t:.2f}')

plt.xlabel('x')
plt.ylabel('u')
plt.title("1D Burgers' Equation - FTCS Method")
plt.legend()
plt.grid(True)
plt.show()