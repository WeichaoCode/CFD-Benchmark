import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 41
nt = 20
nu = 0.3
sigma = 0.2

# Spatial discretization
dx = 2 / (nx - 1)
x = np.linspace(0, 2, nx)

# Temporal discretization
dt = sigma * dx**2 / nu

# Initial condition
u = np.ones(nx)
u[int(0.5 / dx):int(1 / dx + 1)] = 2

# Function to solve the 1D diffusion equation using FDM
def diffusion_1d(u, nt, dt, dx, nu):
    un = np.zeros(nx)  # Temporary array to store new values
    for n in range(nt):
        un = u.copy()
        for i in range(1, nx - 1):
            u[i] = un[i] + nu * dt / dx**2 * (un[i + 1] - 2 * un[i] + un[i - 1])
        
        # Apply Dirichlet boundary conditions
        u[0] = 1
        u[-1] = 0
    return u

# Solve the diffusion equation
u_final = diffusion_1d(u, nt, dt, dx, nu)

# Visualizing the evolution of the solution
def plot_solution(x, u_initial, u_final):
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_initial, label='Initial Condition', color='blue')
    plt.plot(x, u_final, label=f'Solution at t={nt*dt:.2f}', color='red')
    plt.xlabel('Spatial coordinate x')
    plt.ylabel('Quantity u')
    plt.title('1D Diffusion Equation Solution')
    plt.legend()
    plt.grid(True)
    plt.show()

# Initial condition before solving
u_initial = np.ones(nx)
u_initial[int(0.5 / dx):int(1 / dx + 1)] = 2

# Plotting the results
plot_solution(x, u_initial, u_final)

# Save the final solution to a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/u_1D_Diffusion.npy', u_final)