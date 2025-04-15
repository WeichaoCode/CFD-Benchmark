import numpy as np
import matplotlib.pyplot as plt

def linear_convection_diffusion(c, epsilon, x, dx, dt, T, initial_condition):
    N_x = len(x)
    N_t = int(T / dt)
    
    u = initial_condition.copy()
    u_new = np.zeros_like(u)
    
    # Function to compute spatial derivatives
    def compute_f(u):
        # Periodic boundary conditions using roll
        du_dx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        d2u_dx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        return -c * du_dx + epsilon * d2u_dx2
    
    # Initialize time
    t = 0.0
    
    # Compute f at initial time
    f_n = compute_f(u)
    
    # First time step using Explicit Euler
    u_new = u + dt * f_n
    t += dt
    
    # Store previous f
    f_prev = f_n.copy()
    u_prev = u.copy()
    u = u_new.copy()
    
    # Time integration using 2-step Adams-Bashforth
    for n in range(1, N_t):
        f_n = compute_f(u)
        u_new = u + (dt / 2) * (3 * f_n - f_prev)
        t += dt
        
        # Update previous values
        f_prev = f_n.copy()
        u_prev = u.copy()
        u = u_new.copy()
        
    return u

def main():
    # Parameters
    x_min = -5.0
    x_max = 5.0
    N_x = 101
    c = 1.0
    epsilon_cases = [0.0, 5e-4]
    CFL = 0.5  # CFL number for stability
    T = 2.0  # Total time
    
    # Discretize spatial domain
    x = np.linspace(x_min, x_max, N_x)
    dx = x[1] - x[0]
    
    # Initial condition
    u0 = np.exp(-x**2)
    
    for epsilon in epsilon_cases:
        if epsilon == 0.0:
            case = 'undamped'
        else:
            case = 'damped'
        
        # Determine time step based on CFL condition
        if epsilon > 0:
            dt_conv = CFL * dx / c
            dt_diff = CFL * dx**2 / (2 * epsilon)
            dt = min(dt_conv, dt_diff)
        else:
            dt = CFL * dx / c
        
        print(f"Running simulation for {case} case with epsilon={epsilon}")
        print(f"Time step dt={dt:.5f}")
        
        # Run simulation
        u_final = linear_convection_diffusion(c, epsilon, x, dx, dt, T, u0)
        
        # Save the final solution
        filename = f"u_final_{case}.npy"
        np.save(filename, u_final)
        print(f"Final solution saved to {filename}")
        
        # Visualization
        plt.plot(x, u_final, label=f'Epsilon={epsilon}')
    
    # Plot initial condition
    plt.plot(x, u0, 'k--', label='Initial Condition')
    plt.xlabel('x')
    plt.ylabel('u(x, T)')
    plt.title('Final Wave Profile')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()