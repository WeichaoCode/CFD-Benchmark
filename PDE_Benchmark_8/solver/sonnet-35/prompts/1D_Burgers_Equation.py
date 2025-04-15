import numpy as np
import scipy.fftpack as fftpack
import scipy.integrate as integrate

# Problem parameters
nu = 0.07
Lx = 2 * np.pi
Nx = 256
dx = Lx / Nx
x = np.linspace(0, Lx, Nx, endpoint=False)

# Time parameters
t_start = 0
t_end = 0.14 * np.pi
Nt = 1000
dt = (t_end - t_start) / Nt

# Initial condition
def phi(x):
    return np.exp(-x**2 / (4*nu)) + np.exp(-(x - Lx)**2 / (4*nu))

def initial_condition(x):
    dphidx = np.gradient(phi(x), dx)
    return -2 * nu / phi(x) * dphidx + 4

# Initial velocity field
u = initial_condition(x)

# FFT-based solver for periodic advection-diffusion equation
def solve_burgers_equation(u, nu, dt, Nt):
    # Wavenumbers
    k = 2 * np.pi * fftpack.fftfreq(Nx, d=dx)
    
    # Time-stepping using spectral method
    u_hat = fftpack.fft(u)
    
    for _ in range(Nt):
        # Nonlinear term in spectral space
        nonlinear_term = -0.5j * k * fftpack.fft(u**2)
        
        # Linear diffusion term
        linear_term = -nu * k**2 * u_hat
        
        # Explicit RK4 time integration
        k1 = nonlinear_term + linear_term
        k2 = nonlinear_term + linear_term
        k3 = nonlinear_term + linear_term
        k4 = nonlinear_term + linear_term
        
        u_hat += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return np.real(fftpack.ifft(u_hat))

# Solve the equation
u_final = solve_burgers_equation(u, nu, dt, Nt)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/u_final_1D_Burgers_Equation.npy', u_final)