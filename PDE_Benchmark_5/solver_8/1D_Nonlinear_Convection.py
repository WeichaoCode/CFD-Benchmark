import numpy as np
import matplotlib.pyplot as plt

# 1. Define parameters
L = 2.0   # length of the domain
T = 1.0   # time of simulation
nx = 101  # number of spatial discretization points
nt = 100  # number of temporal discretization points
dx = L / (nx-1)  # spatial discretization size
dt = T / (nt-1)  # temporal discretization size

# 2. Discretize space and time
x = np.linspace(0, L, nx)
u = np.ones(nx)
CFL = 0.8  # CFL number should be less than 1 for stability
dt = CFL * dx  # adjust dt to ensure numerical stability

# 3. Set up initial wave profile
mask = np.where((x >= 0.5) & (x <= 1))
u[mask] = 2  # 1 for where x < 0.5 & x > 1 , 2 for where 0.5<= x <= 1

# 4. Finite difference scheme
for n in range(nt):
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - dt/dx * un[i] * (un[i] - un[i-1])

# 5. Plot the wave evolution
plt.plot(x, u)
plt.ylim(1, 2.5)
plt.xlabel('X', fontsize=14)
plt.ylabel('u', fontsize=14)
plt.title('1D Nonlinear Convection', fontsize=14)
plt.grid(True)
plt.show()