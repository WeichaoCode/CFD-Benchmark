import numpy as np
from scipy.linalg import solve

def solve_pde():
    # Problem parameters
    Re_tau = 395.0
    mu = 1.0 / Re_tau
    rho = 1.0

    # Domain
    ny = 100
    y = np.linspace(0.0, 2.0, ny)
    dy = y[1] - y[0]

    # Initialize velocity
    u = np.zeros(ny)

    # Turbulence model (example: simple mixing length model)
    def mixing_length_viscosity(y):
        kappa = 0.41
        l_mix = np.minimum(kappa * y, 0.09 * 2.0)  # Limit mixing length by boundary layer thickness
        l_mix = np.minimum(l_mix, kappa * (2.0 - y))
        
        # Numerical derivative with central difference
        dudy = np.zeros_like(y)
        dudy[1:-1] = (u[2:] - u[:-2]) / (2 * dy)
        dudy[0] = (u[1] - u[0]) / dy
        dudy[-1] = (u[-1] - u[-2]) / dy
        
        mut = rho * l_mix**2 * np.abs(dudy)
        return mut

    # Iterative solver (simple fixed-point iteration)
    max_iter = 100
    tolerance = 1e-6
    
    for iteration in range(max_iter):
        # Compute turbulent viscosity
        mu_t = mixing_length_viscosity(y)
        mu_eff = mu + mu_t

        # Discretization (central difference)
        A = np.zeros((ny, ny))
        b = np.zeros(ny)

        # Interior points
        for i in range(1, ny - 1):
            mu_eff_minus = (mu_eff[i] + mu_eff[i-1])/2
            mu_eff_plus = (mu_eff[i] + mu_eff[i+1])/2
            A[i, i-1] = (mu_eff_minus) / dy**2
            A[i, i] = -((mu_eff_plus + mu_eff_minus) / dy**2)
            A[i, i+1] = (mu_eff_plus) / dy**2
            b[i] = -1.0

        # Boundary conditions
        A[0, 0] = 1.0
        b[0] = 0.0
        A[ny-1, ny-1] = 1.0
        b[ny-1] = 0.0

        # Solve the linear system
        u_new = solve(A, b)

        # Check for convergence
        error = np.max(np.abs(u_new - u))
        if error < tolerance:
            break

        # Update solution
        u = u_new.copy()
    
    # Save the solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)

solve_pde()