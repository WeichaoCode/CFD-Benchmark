import numpy as np
import matplotlib.pyplot as plt

# Set parameters
Lx, Ly = 1.0, 1.0  # domain dimensions
nx, ny = 50, 50  # number of grid points
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # grid spacing

# Initialize the solution
p = np.zeros((ny, nx))

# Set boundary conditions
p[0, :] = 0  # p = 0 at x = 0
p[-1, :] = np.linspace(0, Ly, nx)  # p = y at x = Lx
p[:, 0] = p[:, -1]  # ∂p/∂y = 0 at y = 0, y = Ly

# Solver parameters
maxiter = 20000
rtol = 1e-6

def laplace_2d(p, dx, dy, rtol, maxiter):
    p_n = p.copy()  
    conv = []  
    diff = rtol + 1  
    ite = 0  
    while diff > rtol and ite < maxiter:
        pn = p.copy()  
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, :-2] + pn[1:-1, 2:]) +
                         dx**2 * (pn[:-2, 1:-1] + pn[2:, 1:-1])) /
                        (2 * (dx**2 + dy**2)))  
        diff = np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2))  
        conv.append(diff)  
        ite += 1  
    return p, ite, conv  

# Solve the equation
p, ites, conv = laplace_2d(p, dx, dy, rtol, maxiter)

# Plot the solution
plt.figure(figsize=(8, 5))
plt.contourf(p)
plt.colorbar()
plt.title('Steady-state distribution of p(x, y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()