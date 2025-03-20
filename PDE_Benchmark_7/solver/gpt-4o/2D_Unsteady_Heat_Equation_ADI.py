import numpy as np
import matplotlib.pyplot as plt

# Define the computational domain and parameters
Lx = Ly = 2.0  # Length of domain in x and y directions
nx = ny = 41  # Number of grid points in x and y directions
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
alpha = 1.0  # Thermal diffusivity
t_max = 3.0  # Maximum simulation time
Q0 = 200.0  # Source term coefficient
sigma = 0.1  # Parameter for the Gaussian source

# Stability parameter r
dt = 0.01  # Initial time step guess, will be adjusted based on stability
beta = dx / dy  # Ratio of grid spacing
r = alpha * dt / (2 * dx**2)

# Calculate the source term q(x, y)
x = np.linspace(-Lx/2, Lx/2, nx)
y = np.linspace(-Ly/2, Ly/2, ny)
X, Y = np.meshgrid(x, y)
q = Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Initialize temperature field
T = np.zeros((nx, ny))
T_new = np.zeros_like(T)
T_half = np.zeros_like(T)

# Time stepping
time = 0.0
while time < t_max:
    # Intermediate step: Implicit in x, explicit in y
    # Loop over y-direction (implicit in x-direction)
    for j in range(1, ny - 1):
        # Solve tridiagonal system
        A = np.zeros((nx, nx))
        B = np.zeros((nx))
        
        for i in range(1, nx - 1):
            A[i][i-1] = A[i][i+1] = -r
            A[i][i] = 1 + 2 * r
            B[i] = r * beta**2 * (T[i, j+1] - 2 * T[i, j] + T[i, j-1]) + T[i, j] + 0.5 * dt * q[i, j]
        
        A[0][0] = A[-1][-1] = 1  # Boundary conditions
        B[0] = B[-1] = 0
        
        T_half[:, j] = np.linalg.solve(A, B)

    # Full step: Implicit in y, explicit in x
    # Loop over x-direction (implicit in y-direction)
    for i in range(1, nx - 1):
        # Solve tridiagonal system
        A = np.zeros((ny, ny))
        B = np.zeros((ny))

        for j in range(1, ny - 1):
            A[j][j-1] = A[j][j+1] = -r * beta**2
            A[j][j] = 1 + 2 * r * beta**2
            B[j] = r * (T_half[i+1, j] - 2 * T_half[i, j] + T_half[i-1, j]) + T_half[i, j] + 0.5 * dt * q[i, j]

        A[0][0] = A[-1][-1] = 1  # Boundary conditions
        B[0] = B[-1] = 0

        T_new[i, :] = np.linalg.solve(A, B)

    # Update temperature and time
    T = np.copy(T_new)
    time += dt

# Save the computed solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/T_2D_Unsteady_Heat_Equation_ADI.npy', T)

# Visualize the temperature field
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Temperature Distribution at t = {:.2f} s'.format(t_max))
plt.show()