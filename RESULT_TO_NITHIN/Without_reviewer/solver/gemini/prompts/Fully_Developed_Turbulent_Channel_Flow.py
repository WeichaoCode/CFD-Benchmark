import numpy as np

def solve_channel_flow():
    # Problem parameters
    Re_tau = 395.0
    kappa = 0.42
    A = 25.4
    mu = 1.0 / Re_tau

    # Domain
    ny = 101
    y = np.linspace(0, 2, ny)
    dy = y[1] - y[0]

    # Initial conditions
    u = np.zeros(ny)
    mu_t = np.zeros(ny)

    # Boundary conditions
    u[0] = 0.0
    u[-1] = 0.0

    # Iteration parameters
    max_iter = 1000
    tolerance = 1e-6

    # Finite difference coefficients
    diag = np.zeros(ny)
    lower_diag = np.zeros(ny - 1)
    upper_diag = np.zeros(ny - 1)
    rhs = np.zeros(ny)

    # Iteration loop
    for iter in range(max_iter):
        # Compute effective viscosity
        y_plus = y * Re_tau
        mu_eff_over_mu = 0.5 * (1 + (1/9) * kappa**2 * Re_tau**2 * (2*y - y**2)**2 * (3 - 4*y + 2*y**2)**2 * (1 - np.exp(-y_plus/A))**2)**0.5 - 0.5
        mu_eff = mu * (1 + mu_eff_over_mu)

        # Assemble finite difference matrix
        for i in range(1, ny - 1):
            dmu_eff_dy = (mu_eff[i+1] - mu_eff[i-1]) / (2*dy)
            diag[i] = -2 * mu_eff[i] / dy**2
            lower_diag[i-1] = mu_eff[i] / dy**2 - dmu_eff_dy / (2*dy)
            upper_diag[i] = mu_eff[i] / dy**2 + dmu_eff_dy / (2*dy)
            rhs[i] = -1.0

        # Apply boundary conditions
        diag[0] = 1.0
        lower_diag[0] = 0.0
        rhs[0] = 0.0
        diag[-1] = 1.0
        upper_diag[-1] = 0.0
        rhs[-1] = 0.0

        # Solve tridiagonal system
        diag[1:] -= lower_diag * upper_diag[:-1] / diag[:-1]
        rhs[1:] -= lower_diag * rhs[:-1] / diag[:-1]
        u_new = np.zeros(ny)
        u_new[-1] = rhs[-1] / diag[-1]
        for i in range(ny - 2, -1, -1):
            u_new[i] = (rhs[i] - upper_diag[i] * u_new[i+1]) / diag[i]

        # Check convergence
        error = np.max(np.abs(u_new - u))
        if error < tolerance:
            break

        # Update solution
        u = u_new.copy()

    # Save the solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/mu_eff_Fully_Developed_Turbulent_Channel_Flow.npy', mu_eff)

solve_channel_flow()