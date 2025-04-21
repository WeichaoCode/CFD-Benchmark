import numpy as np

def solve_convection_diffusion():
    # Problem parameters
    rho = 1.0
    u = 2.5
    Gamma = 0.1
    nx = 5
    x_start = 0.0
    x_end = 1.0

    # Grid generation
    dx = (x_end - x_start) / nx
    x = np.linspace(x_start + dx/2, x_end - dx/2, nx)

    # Initialize solution
    phi = np.zeros(nx)

    # Discretization coefficients (Central Differencing Scheme)
    aW = np.zeros(nx)
    aE = np.zeros(nx)
    aP = np.zeros(nx)
    b = np.zeros(nx)

    # Interior nodes
    for i in range(1, nx - 1):
        aW[i] = Gamma / dx + rho * u / 2
        aE[i] = Gamma / dx - rho * u / 2
        aP[i] = aW[i] + aE[i]

    # Boundary nodes
    aW[0] = 0.0
    aE[0] = Gamma / dx - rho * u / 2
    aP[0] = aW[0] + aE[0] + (Gamma / dx + rho * u / 2)
    b[0] = (Gamma / dx + rho * u / 2) * 1.0

    aW[nx-1] = Gamma / dx + rho * u / 2
    aE[nx-1] = 0.0
    aP[nx-1] = aW[nx-1] + aE[nx-1] + (Gamma / dx - rho * u / 2)
    b[nx-1] = (Gamma / dx - rho * u / 2) * 0.0

    # TDMA solver
    def tdma(a, b, c, d):
        n = len(d)
        x = np.zeros(n)

        for i in range(1, n):
            factor = a[i] / b[i-1]
            b[i] -= factor * c[i-1]
            d[i] -= factor * d[i-1]

        x[n-1] = d[n-1] / b[n-1]
        for i in range(n-2, -1, -1):
            x[i] = (d[i] - c[i] * x[i+1]) / b[i]
        return x

    # Prepare coefficients for TDMA
    a = np.copy(aW)
    c = np.copy(aE)
    d = np.copy(b)
    b = np.copy(aP)

    phi = tdma(a, b, c, d)

    # Save the solution
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/phi_1D_Convection_Diffusion_Phi.npy', phi)

solve_convection_diffusion()