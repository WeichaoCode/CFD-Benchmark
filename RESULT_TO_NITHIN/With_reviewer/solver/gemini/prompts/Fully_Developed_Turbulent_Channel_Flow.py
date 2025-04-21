import numpy as np

def solve_channel_flow():
    # Problem parameters
    Re_tau = 395.0
    kappa = 0.42
    A = 25.4
    mu = 1.0 / Re_tau

    # Domain
    y_min = 0.0
    y_max = 2.0
    ny = 101
    y = np.linspace(y_min, y_max, ny)
    dy = y[1] - y[0]

    # Initial conditions
    u = np.zeros(ny)
    mu_t = np.zeros(ny)
    
    # Iteration parameters
    max_iter = 1000
    tolerance = 1e-6

    # Finite difference matrix
    def create_matrix(mu_eff):
        diag = -2.0 * mu_eff[1:-1] / dy**2
        upper_diag = (mu_eff[2:] / dy**2) + (0.5 / dy) * (mu_eff[2:] - mu_eff[1:-1]) / dy
        lower_diag = (mu_eff[0:-2] / dy**2) - (0.5 / dy) * (mu_eff[1:-1] - mu_eff[0:-2]) / dy
        
        main_diag = np.diag(diag)
        upper_diag_shifted = np.diag(upper_diag[:-1], k=1)
        lower_diag_shifted = np.diag(lower_diag[1:], k=-1)
        
        A = main_diag + upper_diag_shifted + lower_diag_shifted
        return A

    # Iteration loop
    for iteration in range(max_iter):
        # Cess turbulence model
        y_plus = y * Re_tau
        cess_term = 0.5 * (1 + (1/9) * kappa**2 * Re_tau**2 * (2*y - y**2)**2 * (3 - 4*y + 2*y**2)**2 * (1 - np.exp(-y_plus/A))**2)**0.5 - 0.5
        mu_eff = mu * (1 + cess_term)

        # Create matrix
        A = create_matrix(mu_eff)

        # Source term
        b = -np.ones(ny - 2)

        # Solve linear system
        u_new = np.zeros(ny)
        u_new[1:-1] = np.linalg.solve(A, b)

        # Boundary conditions
        u_new[0] = 0.0
        u_new[-1] = 0.0

        # Convergence check
        residual = np.linalg.norm(u_new - u)
        if residual < tolerance:
            break

        # Update solution
        u = u_new.copy()

    # Save the solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)

solve_channel_flow()