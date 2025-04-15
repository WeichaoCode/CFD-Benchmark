import numpy as np
import matplotlib.pyplot as plt

def burgers_equation(Lx=2.0, Ly=2.0, T=1.0, nx=101, ny=101, nt=101):
    # define spatial and temporal grids
    Δx, Δy = Lx / (nx - 1), Ly / (ny - 1)
    Δt = T / (nt - 1)
    x, y, t = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), np.linspace(0, T, nt)

    # initialize velocity fields
    u, v = np.zeros((ny, nx)), np.zeros((ny, nx))

    # ensure stability using the CFL condition
    Δt = min(Δt, Δx, Δy)

    # time iteration
    for n in range(nt - 1):
        un, vn = u.copy(), v.copy()
        
        # Finite difference scheme
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                
                u[i, j] = un[i, j] - un[i, j] * Δt / Δx * (un[i, j] - un[i - 1, j]) - vn[i, j] * Δt / Δy * (un[i, j] - un[i, j - 1])

                v[i, j] = vn[i, j] - un[i, j] * Δt / Δx * (vn[i, j] - vn[i - 1, j]) - vn[i, j] * Δt / Δy * (vn[i, j] - vn[i, j - 1])

    return u, v, x, y, t


# Solve the problem
u, v, x, y, t = burgers_equation()

# generate quiver plot
X, Y = np.meshgrid(x, y)
plt.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])
plt.show()