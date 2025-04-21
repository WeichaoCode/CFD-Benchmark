import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

# Grid parameters
Nx = 64
Nz = 64
Lx = 1.0
Lz = 2.0
dx = Lx/Nx
dz = Lz/Nz
x = np.linspace(0, Lx, Nx, endpoint=False)
z = np.linspace(-1, 1, Nz, endpoint=False)
X, Z = np.meshgrid(x, z, indexing='ij')

# Time parameters
dt = 0.01
t_final = 20.0
nt = int(t_final/dt)

# Physical parameters
nu = 1/(5e4)
D = nu/1

# Wavenumbers
kx = 2*np.pi*fftfreq(Nx, dx)
kz = 2*np.pi*fftfreq(Nz, dz)
KX, KZ = np.meshgrid(kx, kz, indexing='ij')
K2 = KX**2 + KZ**2
K2[0,0] = 1.0

# Initial conditions
u = 0.5*(1 + np.tanh((Z-0.5)/0.1) - np.tanh((Z+0.5)/0.1))
w = 0.01*np.sin(2*np.pi*X)*np.exp(-(Z-0.5)**2/0.1**2) + 0.01*np.sin(2*np.pi*X)*np.exp(-(Z+0.5)**2/0.1**2)
s = u.copy()

def spectral_derivative_x(f):
    f_hat = fft2(f)
    return ifft2(1j*KX*f_hat).real

def spectral_derivative_z(f):
    f_hat = fft2(f)
    return ifft2(1j*KZ*f_hat).real

def solve_pressure(u, w):
    ux = spectral_derivative_x(u)
    uz = spectral_derivative_z(u)
    wx = spectral_derivative_x(w)
    wz = spectral_derivative_z(w)
    
    rhs = np.clip(-2*(ux*wx + uz*wz) + (ux**2 + wz**2), -1e10, 1e10)
    p_hat = -fft2(rhs)/K2
    p_hat[0,0] = 0
    return ifft2(p_hat).real

# Time stepping
for n in range(nt):
    # Compute pressure
    p = solve_pressure(u, w)
    
    # Velocity and pressure gradients
    ux = spectral_derivative_x(u)
    uz = spectral_derivative_z(u)
    wx = spectral_derivative_x(w)
    wz = spectral_derivative_z(w)
    px = spectral_derivative_x(p)
    pz = spectral_derivative_z(p)
    
    # Tracer derivatives
    sx = spectral_derivative_x(s)
    sz = spectral_derivative_z(s)
    
    # Nonlinear terms with clipping to prevent overflow
    Nu = np.clip(-(u*ux + w*uz), -1e10, 1e10)
    Nw = np.clip(-(u*wx + w*wz), -1e10, 1e10)
    Ns = np.clip(-(u*sx + w*sz), -1e10, 1e10)
    
    # Diffusion terms
    u_hat = fft2(u)
    w_hat = fft2(w)
    s_hat = fft2(s)
    
    Lu = nu*ifft2(-K2*u_hat).real
    Lw = nu*ifft2(-K2*w_hat).real
    Ls = D*ifft2(-K2*s_hat).real
    
    # Update fields with clipping
    u_new = u + dt*(Nu - px + Lu)
    w_new = w + dt*(Nw - pz + Lw)
    s_new = s + dt*(Ns + Ls)
    
    # Apply clipping to prevent instabilities
    u = np.clip(u_new, -1e10, 1e10)
    w = np.clip(w_new, -1e10, 1e10)
    s = np.clip(s_new, -1e10, 1e10)

# Save final state
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/p_2D_Shear_Flow_With_Tracer.npy', p)