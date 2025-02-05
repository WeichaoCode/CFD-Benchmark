import numpy as np
import matplotlib.pyplot as plt

def solve_diffusion_equation():
    # Physical and numerical parameters
    nu = 0.3  # viscosity
    x_start, x_end = 0, 2
    t_start, t_end = 0, 2

    # Mesh parameters
    nx = 100  # spatial points
    nt = 200  # temporal points

    # Grid generation
    dx = (x_end - x_start) / (nx - 1)
    dt = (t_end - t_start) / (nt - 1)

    x = np.linspace(x_start, x_end, nx)
    t = np.linspace(t_start, t_end, nt)

    # Stability check (von Neumann analysis)
    stability_condition = nu * dt / (dx**2)
    print(f"Stability condition (should be <= 0.5): {stability_condition}")
    if stability_condition > 0.5:
        raise ValueError("Numerical scheme is unstable!")

    # Initialize solution matrix
    u = np.zeros((nt, nx))

    # Initial condition
    u[0, :] = np.sin(np.pi * x)

    # Boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0

    # Solution using First Order Upwind
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            forcing_term = np.pi * nu * np.exp(-t[n]) * np.sin(np.pi * x[i]) - np.exp(-t[n]) * np.sin(np.pi * x[i])
            u[n+1, i] = u[n, i] + dt * (
                nu * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / (dx**2) + forcing_term
            )

    # Plotting
    time_steps = [0, nt//4, nt//2, nt-1]
    plt.figure(figsize=(10, 6))
    for idx in time_steps:
        plt.plot(x, u[idx, :], label=f't = {t[idx]:.2f}')

    plt.title('1D Diffusion Equation - First Order Upwind')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    plt.show()

    return u

# Run the simulation
solution = solve_diffusion_equation()