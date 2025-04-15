import math

# Define physical properties
nu = 0.07
pi = math.pi

# Define spatial and temporal domains
x = [i*0.01 for i in range(201)]
t = [i*0.01 for i in range(201)]

# Initialize solution matrix
u = [[0]*len(t) for _ in range(len(x))]

# Set initial conditions
for i in range(len(x)):
    u[i][0] = math.sin(pi * x[i])

# Set boundary conditions
for j in range(len(t)):
    u[0][j] = 0
    u[-1][j] = 0

# Apply finite difference: First Order Upwind method
for j in range(len(t)-1):
    for i in range(1, len(x)-1):
        u[i][j+1] = u[i][j] - (0.01/(2*0.01))*(u[i+1][j]**2 - u[i-1][j]**2) \
                     + nu*((u[i+1][j]-2*u[i][j]+u[i-1][j])/(0.01**2)) \
                     - pi**2 * nu * math.exp(-t[j]) * math.sin(pi * x[i]) \
                     + math.exp(-t[j]) * math.sin(pi * x[i]) \
                     - pi * math.exp(-2*t[j]) * math.sin(pi * x[i]) * math.cos(pi * x[i])

# Von Neumann stability analysis
for j in range(len(t)-1):
    for i in range(1, len(x)-1):
        r = 0.01 * u[i][j] / (2*0.01)
        if abs(r) > 1:
            print("The solution is unstable.")
            exit(1)

# Plot the solution at key time steps
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.title('1D Burgers equation - FOU method')
plt.plot(x, [u[i][0] for i in range(len(x))], label='t = 0')
plt.plot(x, [u[i][len(t)//4] for i in range(len(x))], label='t = T/4')
plt.plot(x, [u[i][len(t)//2] for i in range(len(x))], label='t = T/2')
plt.plot(x, [u[i][-1] for i in range(len(x))], label='t = T')
plt.legend()
plt.show()