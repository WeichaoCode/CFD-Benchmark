import numpy as np
import matplotlib.pyplot as plt

def burgers_eq(L=2*np.pi, T=1.0, nx=101, nt=100, nu=0.07):
    # Step sizes
    dx = L / (nx - 1)
    dt = T / nt

    # Initialize grid
    x = np.linspace(0, L, nx)
    u = -2 * nu * (np.exp(-x**2 / (4 * nu)) + np.exp(-(x - 2 * np.pi)**2 / (4 * nu))) / (np.exp(-x**2 / (4 * nu)) + np.exp(-(x - 2 * np.pi)**2 / (4 * nu))) + 4
    
    un = np.empty_like(u)
    for n in range(nt):  # time steps
        un = u.copy()
        u[1:-1] = un[1:-1] - un[1:-1] * dt / dx * (un[1:-1] - un[:-2]) + nu * dt / dx**2 * \
                (un[2:] - 2 * un[1:-1] + un[:-2])
        u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 * \
               (un[1] - 2 * un[0] + un[-2])
        u[-1] = u[0]
        
    u_analytical = -2 * nu * (np.exp(-(x - 4 * T)**2 / (4 * nu * (T + 1))) + np.exp(-(x - 4 * T - 2 * np.pi)**2 / (4 * nu * (T + 1)))) \
                   / (np.exp(-(x - 4 * T)**2 / (4 * nu * (T + 1))) + np.exp(-(x - 4 * T - 2 * np.pi)**2 / (4 * nu * (T + 1)))) + 4

    return x, u, u_analytical

x, u, u_analytical = burgers_eq()

plt.figure(figsize=(11, 7), dpi=100)
plt.plot(x, u, marker='o', lw=2, label='Computational')
plt.plot(x, u_analytical, label='Analytical')
plt.xlim([0, 2 * np.pi])
plt.ylim([0, 10])
plt.legend()
plt.show()