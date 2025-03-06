import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import diff as psdiff

def solve_1d_korteweg_de_vries(N, T, L):
    dt = T/N
    dx = L/N
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    t = np.linspace(0, T, N, endpoint=False)
    
    u = np.empty((N, N), dtype=complex)

    # Initialization
    u_init = np.exp(-t) * np.sin(np.pi * x)
    u[:, 0] = u_init

    # Boundary conditions
    u[0, :] = u[-1, :]
    u[-1, :] = u[0, :]

    # Source Term
    f = -np.exp(-t)*(np.pi*np.sin(np.pi*x) - 6*np.pi*np.sin(np.pi*x)**2 - np.pi**3*np.sin(np.pi*x))

    for n in range(N-1):
        F = np.exp(-t[n+1]) * f
        un = np.exp(-t[n+1]) * u[:,n]

        # Time marching
        # 3. order derivative w/ periodic bc. = fourier psdiff method
        uxxx = psdiff(un, period=L, order=3)
        u_x = psdiff(un, period=L, order=1)

        dudt = -6*un*u_x - uxxx

        # Euler method
        u[:, n+1] = u[:, n] + dt*dudt

    return u

# Solve PDE
u = solve_1d_korteweg_de_vries(100, 0.01, 1)

# Plot numerical solution
plt.imshow(abs(u), extent=[-0.5, 0.5, 0, 0.01], origin='lower', aspect='auto')
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('t')
plt.show()