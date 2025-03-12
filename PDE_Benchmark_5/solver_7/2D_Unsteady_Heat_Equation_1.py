import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha = 1
Q_0 = 200
sigma = 0.1
dt = 0.01 # An arbitrary small time step
max_time = 0.5 # Arbitrary final time

# Spatial discretization
nx, ny = 100, 100
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Make sure the time step meets the CFL condition
dt = min(0.25*dx*dx/alpha, 0.25*dy*dy/alpha)

# Initialize solution: the initial condition.
T = np.zeros((nx, ny))

# Heat source
def q_source(x, y):
    return Q_0 * np.exp(- (x**2 + y**2) / (2*sigma**2))

q = q_source(*np.meshgrid(x, y, indexing='ij'))

def solve_heat_eq(T, dt):
    for _ in np.arange(0, max_time, dt):
        T_new = T + dt*( alpha*(np.roll(T, -1, 0) - 2*T + np.roll(T, 1, 0))/dx/dx
                         + alpha*(np.roll(T, -1, 1) - 2*T + np.roll(T, 1, 1))/dy/dy
                         + q)
        T_new[0] = T_new[-1] = 0 # Boundary condition
        T_new[:, 0] = T_new[:, -1] = 0 # Boundary condition
        T = T_new
    return T

T = solve_heat_eq(T, dt)

# Creating the x and y dimensions
X, Y = np.meshgrid(x, y)

plt.figure()
plt.contourf(X, Y, T, cmap='hot')
plt.title('Heat equation solution')
plt.colorbar(label='Temperature (\u00b0C)')
plt.show()