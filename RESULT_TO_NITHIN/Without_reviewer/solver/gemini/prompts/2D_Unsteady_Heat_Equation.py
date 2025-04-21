import numpy as np

def solve_heat_equation():
    # Parameters
    alpha = 0.01
    Q_0 = 200
    sigma = 0.1
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    t_final = 3
    nx = 50
    ny = 50
    nt = 150

    # Grid
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = t_final / (nt - 1)

    # Initialize temperature field
    T = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            T[i, j] = 1 + 200 * np.exp(-(x[j]**2 + y[i]**2) / (2 * sigma**2))

    # Boundary conditions
    T[:, 0] = 1
    T[:, -1] = 1
    T[0, :] = 1
    T[-1, :] = 1

    # Time loop
    for n in range(1, nt):
        T_new = T.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                q = Q_0 * np.exp(-(x[j]**2 + y[i]**2) / (2 * sigma**2))
                T_new[i, j] = T[i, j] + alpha * dt * (
                    (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dx**2 +
                    (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dy**2
                ) + dt * q
        
        # Boundary conditions
        T_new[:, 0] = 1
        T_new[:, -1] = 1
        T_new[0, :] = 1
        T_new[-1, :] = 1
        
        T = T_new

    # Save the final temperature field
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/T_2D_Unsteady_Heat_Equation.npy', T)

solve_heat_equation()