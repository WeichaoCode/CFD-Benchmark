import numpy as np
import scipy.fftpack as fftpack

# Problem parameters
nu = 0.07
Lx = 2 * np.pi
nx = 256
nt = 400
dt = 0.14 * np.pi / nt
x = np.linspace(0, Lx, nx, endpoint=False)

# Initial condition
def initial_phi(x):
    return np.exp(-x**2 / (4*nu)) + np.exp(-(x - Lx)**2 / (4*nu))

def initial_condition(x):
    phi = initial_phi(x)
    u0 = -2 * nu * np.gradient(phi, x) / phi + 4
    return u0

# Initial velocity field
u = initial_condition(x)

# Solve using Fourier spectral method
def nonlinear_term(u):
    # Use central difference with periodic boundary conditions
    du_dx = np.zeros_like(u)
    du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * (Lx/nx))
    du_dx[0] = (u[1] - u[-1]) / (2 * (Lx/nx))
    du_dx[-1] = (u[0] - u[-2]) / (2 * (Lx/nx))
    
    # Compute nonlinear term with scaled multiplication to avoid overflow
    return -0.5 * np.multiply(u, du_dx, dtype=np.float64)

# Time integration
for _ in range(nt):
    # Compute nonlinear term in physical space
    N = nonlinear_term(u)
    
    # Transform to spectral space
    u_hat = fftpack.fft(u)
    N_hat = fftpack.fft(N)
    
    # Compute linear diffusion term in spectral space
    k = fftpack.fftfreq(nx, d=Lx/nx) * 2 * np.pi
    linear_term = nu * k**2 * u_hat
    
    # RK4 time integration with careful type handling
    k1_hat = dt * (linear_term + N_hat)
    k1 = np.real(fftpack.ifft(k1_hat))
    
    k2_hat = dt * (linear_term + fftpack.fft(nonlinear_term(u + 0.5*k1))) 
    k2 = np.real(fftpack.ifft(k2_hat))
    
    k3_hat = dt * (linear_term + fftpack.fft(nonlinear_term(u + 0.5*k2)))
    k3 = np.real(fftpack.ifft(k3_hat))
    
    k4_hat = dt * (linear_term + fftpack.fft(nonlinear_term(u + k3)))
    k4 = np.real(fftpack.ifft(k4_hat))
    
    # Update solution with type-safe addition
    u = u + (k1 + 2*k2 + 2*k3 + k4) / 6

# Ensure final solution is real and bounded
u = np.clip(np.real(u), -1e10, 1e10)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/u_1D_Burgers_Equation.npy', u)