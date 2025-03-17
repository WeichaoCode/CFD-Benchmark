import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate 2D plot
def plot_2D_solution(X, Y, u, iteration):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, u[:])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('U')
    plt.title('2D Diffusion - Iteration: {}'.format(iteration))
    plt.show()

# Function for Explicit Euler Method
def Explicit_Euler(u, nt, dt, dx, dy, nu):
    for n in range(nt):
        un = u.copy()
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] + nu * dt / dx**2 *
                         (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         nu * dt / dy**2 *
                         (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))
        plot_2D_solution(X, Y, u, n)
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
    return u

# Define parameters
nx = 31
ny = 31
nt = 50
nu = 0.05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)
sigma = .25
dt = sigma * dx * dy / nu

# Initialize variables
u = np.ones((ny, nx))
u[int(.5 / dy): int(1 / dy + 1),int(.5 / dx): int(1 / dx + 1)]=2 

# Apply Explicit Euler Method
u = Explicit_Euler(u, nt, dt, dx, dy, nu)

# Save final solution to .npy file
np.save('final_u.npy', u)