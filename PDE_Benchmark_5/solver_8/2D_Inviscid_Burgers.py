import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Lx, Ly = 2.0, 2.0
T = 1.0
nx, ny, nt = 101, 101, 101
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = T / (nt - 1)

# Ensure CFL stability
assert dt <= min(dx, dy) / np.sqrt(2.0), "CFL condition not satisfied"

# Set up grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.linspace(0, T, nt)

X, Y = np.meshgrid(x, y)

# Set initial conditions
U = np.zeros((ny, nx))
V = np.zeros((ny, nx))

U[int(0.5 / dy):int(1.5 / dy), int(0.5 / dx):int(1.5 / dx)] = 2
V[int(0.5 / dy):int(1.5 / dy), int(0.5 / dx):int(1.5 / dx)] = 2

# Finite difference scheme
for n in range(nt - 1):
    Un = U.copy()
    Vn = V.copy()
    
    U[1:, 1:] = (Un[1:, 1:] 
                 - (Un[1:, 1:] * dt / dx * (Un[1:, 1:] - Un[1:, :-1])) 
                 - (Vn[1:, 1:] * dt / dy * (Un[1:, 1:] - Un[:-1, 1:])))
    
    V[1:, 1:] = (Vn[1:, 1:] 
                 - (Un[1:, 1:] * dt / dx * (Vn[1:, 1:] - Vn[1:, :-1])) 
                 - (Vn[1:, 1:] * dt / dy * (Vn[1:, 1:] - Vn[:-1, 1:])))
    
    # Apply boundary conditions
    U[0, :], U[-1, :], U[:, 0], U[:, -1] = 1, 1, 1, 1
    V[0, :], V[-1, :], V[:, 0], V[:, -1] = 1, 1, 1, 1

# Plotting
fig = plt.figure(figsize=(11,7), dpi=100)
plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3])
plt.show()