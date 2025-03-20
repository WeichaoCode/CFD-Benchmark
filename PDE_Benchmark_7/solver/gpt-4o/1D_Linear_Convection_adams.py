import numpy as np
import matplotlib.pyplot as plt

def initialize_wave(N_x, domain):
    x = np.linspace(domain[0], domain[1], N_x)
    u0 = np.exp(-x**2)
    return x, u0

def central_diff(u, dx, epsilon):
    nx = len(u)
    dudx = np.zeros_like(u)
    
    for i in range(nx):
        dudx[i] = (u[(i+1) % nx] - u[i-1]) / (2 * dx)  # periodic boundary condition
        d2udx2 = (u[(i+1) % nx] - 2*u[i] + u[i-1]) / dx**2  # central difference for second derivative
        dudx[i] = -c * dudx[i] + epsilon * d2udx2
    
    return dudx

def simulate_convection(N_x, domain, T, c, epsilon, CFL):
    # Initialize variables
    x, u = initialize_wave(N_x, domain)
    dx = (domain[1] - domain[0]) / (N_x - 1)
    dt = CFL * dx / abs(c)
    N_t = int(T / dt)
    
    u_history = [u.copy()]
    
    # Explicit Euler for the first time step
    dudx = central_diff(u, dx, epsilon)
    u_new = u + dt * dudx
    u_old = u.copy()
    u[:] = u_new
    
    # Time integration using 2-step Adams-Bashforth Method
    for n in range(1, N_t):
        dudx = central_diff(u, dx, epsilon)
        u_new = u + dt * (3/2 * dudx - 1/2 * central_diff(u_old, dx, epsilon))
        
        u_old = u.copy()
        u[:] = u_new
        
        # Store the solution at every 100 steps for visualization
        if n % 100 == 0:
            u_history.append(u.copy())

    return x, u, u_history

def plot_solution(x, u_history, title):
    plt.figure(figsize=(10, 6))
    for i in range(len(u_history)):
        plt.plot(x, u_history[i], label=f"Step {i*100}")
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Wave Amplitude u(x, t)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define Parameters
N_x = 101
domain = (-5, 5)
T = 3.0
c = 1.0
CFL = 0.5

# Case 1: Undamped (epsilon = 0)
epsilon_undamped = 0.0
x, u_final_undamped, u_history_undamped = simulate_convection(N_x, domain, T, c, epsilon_undamped, CFL)
plot_solution(x, u_history_undamped, "Undamped Wave Propagation")

# Case 2: Damped (epsilon = 5e-4)
epsilon_damped = 5e-4
x, u_final_damped, u_history_damped = simulate_convection(N_x, domain, T, c, epsilon_damped, CFL)
plot_solution(x, u_history_damped, "Damped Wave Propagation")

# Save final solutions
np.save("/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/u_1D_Linear_Convection_adams.npy", u_final_undamped)