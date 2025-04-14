import numpy as np
import scipy.fft as fft

def kdv_burgers_solver():
    # Domain parameters
    L = 10.0  # Domain length
    T = 10.0  # Total simulation time
    nx = 512  # Spatial resolution 
    nt = 1000  # Temporal resolution

    # Physical parameters
    a = 1e-4  # Diffusion coefficient 
    b = 2e-4  # Dispersion coefficient
    n = 20.0  # Initial condition parameter

    # Grid setup
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]
    dt = T / nt

    # Initial condition 
    u0 = 0.5/n * np.log(1 + np.cosh(n)**2 / np.cosh(n*(x - 0.2*L))**2)

    # Spectral method for solving KdV-Burgers equation
    u = u0.copy()
    k = 2*np.pi*fft.fftfreq(nx, dx)

    # Time integration using spectral method
    for _ in range(nt):
        # Compute nonlinear term in spectral space
        u_hat = fft.fft(u)
        
        # Compute derivative in spectral space
        du_dx_hat = 1j * k * u_hat
        du_dx = np.real(fft.ifft(du_dx_hat))
        
        # Nonlinear term computation with safe scaling
        nonlinear_term = np.clip(u, -1e10, 1e10) * np.clip(du_dx, -1e10, 1e10)
        nonlinear_hat = -0.5j * k * fft.fft(nonlinear_term)
        
        # Linear terms in spectral space
        linear = -a * k**2 + b * k**3
        
        # Solve using exponential time differencing
        u_hat = u_hat + dt * (nonlinear_hat + linear * u_hat)
        u = np.real(fft.ifft(u_hat))

    # Save final solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_1D_KdV_Burgers_Equation.npy', u)

# Run solver
kdv_burgers_solver()