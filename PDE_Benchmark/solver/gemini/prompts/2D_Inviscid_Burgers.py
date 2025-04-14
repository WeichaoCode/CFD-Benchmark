import numpy as np

def solve_pde():
    # Parameters
    nx = 50
    ny = 50
    nt = 100
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = 0.004
    
    # Initialize u and v
    u = np.ones((ny, nx))
    v = np.ones((ny, nx))
    
    # Initial conditions
    for i in range(ny):
        for j in range(nx):
            x = j * dx
            y = i * dy
            if 0.5 <= x <= 1 and 0.5 <= y <= 1:
                u[i, j] = 2
                v[i, j] = 2
    
    # Boundary conditions
    u[:, 0] = 1
    u[:, -1] = 1
    u[0, :] = 1
    u[-1, :] = 1
    
    v[:, 0] = 1
    v[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1
    
    # Time loop
    for n in range(nt):
        u_old = u.copy()
        v_old = v.copy()
        
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u[i, j] = u_old[i, j] - dt * (u_old[i, j] * (u_old[i, j] - u_old[i, j - 1]) / dx + v_old[i, j] * (u_old[i, j] - u_old[i - 1, j]) / dy)
                v[i, j] = v_old[i, j] - dt * (u_old[i, j] * (v_old[i, j] - v_old[i, j - 1]) / dx + v_old[i, j] * (v_old[i, j] - v_old[i - 1, j]) / dy)
        
        # Boundary conditions
        u[:, 0] = 1
        u[:, -1] = 1
        u[0, :] = 1
        u[-1, :] = 1
        
        v[:, 0] = 1
        v[:, -1] = 1
        v[0, :] = 1
        v[-1, :] = 1
    
    # Save the final solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Inviscid_Burgers.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/v_2D_Inviscid_Burgers.npy', v)

solve_pde()