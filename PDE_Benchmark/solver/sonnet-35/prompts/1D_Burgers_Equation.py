import numpy as np
import scipy.fft as fft

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

# Time integration using Fourier spectral method
for _ in range(nt):
    # Compute derivatives in Fourier space
    u_hat = fft.rfft(u)
    dx_hat = 1j * fft.rfftfreq(nx, Lx/nx)
    
    # Nonlinear term (using anti-aliasing and dealiasing)
    du_dx = fft.irfft(1j * dx_hat * u_hat, n=nx)
    
    # Compute nonlinear term with careful scaling to prevent overflow
    nonlinear = du_dx * u
    nonlinear_hat = fft.rfft(nonlinear) * 0.5
    
    # Linear term (diffusion)
    linear_hat = -nu * dx_hat**2 * u_hat
    
    # Time stepping
    u_hat = u_hat + dt * (linear_hat - nonlinear_hat)
    
    # Inverse transform back to physical space
    u = fft.irfft(u_hat)

# Save final solution
np.save('u.npy', u)