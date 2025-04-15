import math

# Constants
nu = 0.3
pi = math.pi
T = 2.0
x_range = [0, 2]
N = 100  # number of space steps
M = 100  # number of time steps
h = (x_range[1] - x_range[0]) / N  # space step size
k = T / M  # time step size

# Initialize solutions
u = [[0 for _ in range(N+1)] for _ in range(M+1)]
x = [i*h for i in range(N+1)]
t = [j*k for j in range(M+1)]

# Initial condition
for i in range(N+1):
    u[0][i] = math.sin(pi * x[i])

# Boundary conditions
for j in range(M+1):
    u[j][0] = 0
    u[j][N] = 0

# Finite difference scheme - Beam-Warming method
for j in range(M):
    for i in range(2, N):
        u[j+1][i] = u[j][i] - 0.5*k/h*(3*u[j][i] - 4*u[j][i-1] + u[j][i-2]) \
                     + k**2/(2*h**2)*(u[j][i] - 2*u[j][i-1] + u[j][i-2]) \
                     - k*nu*pi**2*math.exp(-t[j])*math.sin(pi*x[i]) \
                     + k*math.exp(-t[j])*math.sin(pi*x[i])

# Von Neumann stability analysis
stability = k / h**2
print("Stability (should be <= 0.5 for stability): ", stability)

# Print the solution at key time steps
print("Solution at t = 0: ", u[0])
print("Solution at t = T/4: ", u[M//4])
print("Solution at t = T/2: ", u[M//2])
print("Solution at t = T: ", u[M])