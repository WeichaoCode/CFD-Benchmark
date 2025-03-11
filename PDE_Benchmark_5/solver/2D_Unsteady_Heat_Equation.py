import numpy as np
import matplotlib.pyplot as plt

# Grid Parameters
N = 100
dx = 2.0/N
dt = (dx**2)/(4*1.0) # Stability condition for explicit method
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

# Time parameters
t_end = 1.0
t_steps = int(t_end/dt)
t = np.linspace(0, t_end, t_steps)

# Thermo-physical Property
alpha = 1.0 # Thermal diffusivity

# Initial and Boundary Conditions
T = np.zeros((N, N))

T_boundary = 0
T.fill(T_boundary) 

# Heat Source term
Q_0 = 200
sigma = 0.1
q = Q_0*np.exp( - (X**2 + Y**2) / (2*sigma**2) )

# Simulation
for n in range(t_steps):
    T_new = T.copy()
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha*dt*((T[:-2, 1:-1] - 2*T[1:-1, 1:-1] + T[2:, 1:-1])/dx**2 + (T[1:-1,:-2] - 2*T[1:-1, 1:-1] + T[1:-1, 2:])/dx**2) + dt*q[1:-1, 1:-1]
    T = T_new

# Visualization
plt.contourf(X, Y, T)
plt.title("2D Heat Equation: Explicit Method")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Temperature")
plt.show()