import numpy as np
import matplotlib.pyplot as plt

# Parameters
h = 0.1  # meters
nx = ny = 80
dx = dy = h / (nx - 1)
mu = 1e-3  # Pa·s
rho = 1  # kg/m³
dP_dz = -3.2  # Pa/m

# Coefficients
a_E = a_W = mu * dy / dx
a_N = a_S = mu * dx / dy
a_P = a_E + a_W + a_N + a_S
S_u = dP_dz * dx * dy

# Initialize velocity field
w = np.zeros((ny, nx))
w_new = np.zeros_like(w)

# Iterative solver parameters
max_iterations = 10000
tolerance = 1e-6

for iteration in range(max_iterations):
    # Update interior points
    w_new[1:-1,1:-1] = (a_E * w[1:-1,2:] +
                         a_W * w[1:-1,0:-2] +
                         a_N * w[2:,1:-1] +
                         a_S * w[0:-2,1:-1] -
                         S_u) / a_P

    # Compute the maximum difference
    diff = np.max(np.abs(w_new - w))
    
    # Check for convergence
    if diff < tolerance:
        print(f'Converged after {iteration+1} iterations.')
        break
    
    # Update the velocity field
    w[:, :] = w_new[:, :]

else:
    print('Maximum iterations reached without convergence.')

# Create grid for plotting
x = np.linspace(0, h, nx)
y = np.linspace(0, h, ny)
X, Y = np.meshgrid(x, y)

# Plot velocity field
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, w, 50, cmap='viridis')
plt.colorbar(contour, label='Velocity w (m/s)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Velocity Distribution in Square Duct')
plt.tight_layout()
plt.show()

# Save the velocity field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/w_2D_Diffusion_FVM.npy', w)