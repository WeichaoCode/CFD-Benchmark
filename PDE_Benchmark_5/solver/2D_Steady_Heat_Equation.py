import numpy as np
import matplotlib.pyplot as plt

# Define domain
x_length = 5
y_length = 4
delta_x = 0.1
delta_y = 0.1
x_points = int(x_length / delta_x)
y_points = int(y_length / delta_y)
T = np.zeros((x_points, y_points))

# Boundary Conditions
T[:, 0] = 10     # AB side
T[:, -1] = 40    # EF side
T[0, :] = 0      # CD side
T[-1, :] = 20    # GH side

# Simulation Parameters
convergence_criterion = 1e-6
difference = 1
iterations = 0
max_iterations = 1000

# Jacobi iteration
while difference > convergence_criterion and iterations < max_iterations:
    T_new = T.copy()
    for i in range(1, x_points-1):
        for j in range(1, y_points-1):
            T_new[i, j] = 0.5 * (delta_y**2 * (T[i-1, j] + T[i+1, j]) +
                                  delta_x**2 * (T[i, j-1] + T[i, j+1])) / (delta_x**2 + delta_y**2)

    # Compute Maximum Difference
    difference = abs((T_new - T)).max()
    # Update Temperature values
    T = T_new.copy()
    iterations += 1

# Plot the temperature distribution
fig = plt.figure(figsize=(6, 5))
plt.title("Steady State Heat Equation")
plt.imshow(T.T, extent=[0, x_length, 0, y_length], origin='lower')
plt.colorbar(label="Temperature (°C)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()