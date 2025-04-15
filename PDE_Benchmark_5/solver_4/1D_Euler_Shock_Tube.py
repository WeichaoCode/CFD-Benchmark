import numpy as np
from matplotlib import pyplot as plt

# Define parameters
L = 1.0
T = 0.2
nx = 100
nt = 100
γ = 1.4
CFL = 0.4

# Discretize space and time
dx = L / nx
dt = CFL * dx

# Initialize primitive and conserved variables
u_prim_left = [1.0, 0.0, 1.0]
u_prim_right = [0.125, 0.0, 0.1]

u_prim = np.zeros((3, nx))
u_con = np.zeros_like(u_prim)

u_prim[:, :nx//2] = np.array(u_prim_left)[:, None]
u_prim[:, nx//2:] = np.array(u_prim_right)[:, None]

ρ, u, p = u_prim[0], u_prim[1], u_prim[2]
u_con[0], u_con[1], u_con[2] = ρ, ρ * u, ρ * (0.5 * u ** 2 + p / (γ - 1))

# Iterate using MacCormack method
u_pred = np.zeros_like(u_con)
u_corr = np.zeros_like(u_con)

for n in range(nt):
    F = np.zeros_like(u_con)
    F[0] = u_con[1]
    F[1] = u_con[1] ** 2 / u_con[0] + (γ - 1) * (u_con[2] - 0.5 * u_con[1] ** 2 / u_con[0])
    F[2] = (u_con[2] + (γ - 1) * (u_con[2] - 0.5 * u_con[1] ** 2 / u_con[0])) * u_con[1] / u_con[0]

    u_pred[:, :-1] = u_con[:, :-1] - dt / dx * (F[:, 1:] - F[:, :-1])
    u_pred[:, -1] = u_con[:, -1]

    F_pred = np.zeros_like(u_pred)
    F_pred[0] = u_pred[1]
    F_pred[1] = u_pred[1] ** 2 / u_pred[0] + (γ - 1) * (u_pred[2] - 0.5 * u_pred[1] ** 2 / u_pred[0])
    F_pred[2] = (u_pred[2] + (γ - 1) * (u_pred[2] - 0.5 * u_pred[1] ** 2 / u_pred[0])) * u_pred[1] / u_pred[0]

    u_corr[:, 1:] = 0.5 * (u_con[:, 1:] + u_pred[:, 1:] - dt / dx * (F_pred[:, 1:] - F_pred[:, :-1]))
    u_corr[:, 0] = u_con[:, 0]
    u_con = u_corr

# Convert back to primitive variables
ρ = u_con[0]
u = u_con[1] / ρ
p = (γ - 1) * (u_con[2] - 0.5 * ρ * u ** 2)

# Visualize the results
x = np.linspace(0.0, L, nx)
plt.figure(figsize=(6, 6))

plt.subplot(3, 1, 1)
plt.plot(x, ρ, 'k-')
plt.title('Density')

plt.subplot(3, 1, 2)
plt.plot(x, u, 'k-')
plt.title('Velocity')

plt.subplot(3, 1, 3)
plt.plot(x, p, 'k-')
plt.title('Pressure')

plt.tight_layout()
plt.show()