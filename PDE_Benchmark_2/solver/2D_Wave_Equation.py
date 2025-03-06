import numpy as np
import matplotlib.pyplot as plt

# define the manufactured solution
def u_exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(t)

# source term derived from the manufactured solution
def source(x, y, t):
    return 2*np.pi**2*np.exp(-t)*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(t) - \
           2*np.exp(-t)*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(t)

# set the parameters and grid
nx, ny, nt = 101, 101, 100  # grid resolution
T = 1.0                     # end time
c = 1.0                     # wave speed
xmin, xmax = 0, 1           # x-domain
ymin, ymax = 0, 1           # y-domain
x, dx = np.linspace(xmin, xmax, nx, retstep=True)
y, dy = np.linspace(ymin, ymax, ny, retstep=True)
t, dt = np.linspace(0, T, nt, retstep=True)

# initialise wave field
u = np.zeros((nx, ny, nt+1))
u[:, :, 0] = u_exact(x.reshape(-1, 1), y, 0)  # initial condition
u[:, :, 1] = u[:, :, 0] + dt*u_exact(x.reshape(-1, 1), y, dt)  # forward Euler step

# check CFL conditions
assert dt <= dx/(np.sqrt(2.0)*c), "CFL condition not met!"

for n in range(1, nt):
    for j in range(1, nx-1):
        for k in range(1, ny-1):
            # apply finite difference scheme
            u[j, k, n+1] = 2*u[j, k, n] - u[j, k, n-1] + \
                           (c**2*dt**2/dx**2)*(u[j+1, k, n] - 2*u[j, k, n] + u[j-1, k, n]) + \
                           (c**2*dt**2/dy**2)*(u[j, k+1, n] - 2*u[j, k, n] + u[j, k-1, n]) + \
                           dt**2*source(x[j], y[k], t[n])

# visualize the solution, exact solution and absolute error
plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
plt.contourf(x, y, u_exact(x.reshape(-1, 1), y, T), cmap="viridis")
plt.title("Exact Solution")
plt.subplot(1,3,2)
plt.contourf(x, y, u[:,:,-1], cmap="viridis")
plt.title("Numerical Solution")
plt.subplot(1,3,3)
plt.contourf(x, y, np.abs(u[:,:,-1]-u_exact(x.reshape(-1, 1), y, T)), cmap="viridis")
plt.title("Absolute Error")
plt.show()