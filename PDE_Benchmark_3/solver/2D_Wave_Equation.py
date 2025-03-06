import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def solve_2d_wave_equation(c, Nx, Ny, Nt, Lx=1, Ly=1, T=1):
    # Step 1: Define grid
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = T / (Nt - 1)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    
    # Step 2: Check CFL condition
    if c * dt / max(dx, dy) > 1/np.sqrt(2):
        raise ValueError("CFL condition not met")
        
    # Step 3: Compute source term
    def f(x, y, t):
        return c**2 * (np.pi**2 * (np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(t) * np.exp(-t)) -
                       2 * np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(t)) - \
               (2 * np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(t))
    
    # Step 4: Compute initial and boundary conditions
    u = np.zeros((Nx, Ny, Nt))
    u[..., 0] = np.sin(np.pi * x[:, None]) * np.sin(np.pi * y[None, :])  # initial condition
    u[[0, -1], :, :] = 0   # boundary condition on x = 0 and x = Lx
    u[:, [0, -1], :] = 0   # boundary condition on y = 0 and y = Ly
    
    # Step 5: Time-stepping using central difference
    for n in range(Nt - 1):
        uxx = (u[2:, 1:-1, n] - 2*u[1:-1, 1:-1, n] + u[:-2, 1:-1, n]) / (dx**2)
        uyy = (u[1:-1, 2:, n] - 2*u[1:-1, 1:-1, n] + u[1:-1,:-2, n]) / (dy**2)
        u[1:-1, 1:-1, n+1] = 2*(1 - c**2*(dt**2)*(1/dx**2 + 1/dy**2))*u[1:-1, 1:-1, n] - u[1:-1, 1:-1, n-1]\
                           + c**2*dt**2*(uxx + uyy) + dt**2*f(x[1:-1, None], y[None, 1:-1], n*dt)

    return x, y, u

# Step 6: Compute exact solution
def exact_sol(x, y, T):
    t = np.linspace(0, T, len(x))
    u_exact = np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(t)
    return u_exact

# Step 7: Calculate errors & plot
x, y, u = solve_2d_wave_equation(1, 50, 50, 100, T=1)
error = np.abs(u - exact_sol(x[None, :, None], y[:, None, None], 1))
X, Y, T = np.meshgrid(x, y, range(u.shape[2]))
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, error[..., -1], cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('Error')
plt.show()