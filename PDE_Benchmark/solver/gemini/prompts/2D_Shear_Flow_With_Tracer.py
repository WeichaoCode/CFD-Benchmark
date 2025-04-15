import numpy as np
import scipy.fft as fft
import os

def solve_cfd():
    # Parameters
    nx = 64
    nz = 128
    nt = 200
    dt = 0.1
    nu = 1 / (5 * 10**4)
    D = nu / 1
    
    # Domain
    x = np.linspace(0, 1, nx, endpoint=False)
    z = np.linspace(-1, 1, nz, endpoint=False)
    X, Z = np.meshgrid(x, z)
    
    # Initial conditions
    u = 0.5 * (1 + np.tanh((Z - 0.5) / 0.1) - np.tanh((Z + 0.5) / 0.1))
    w = 0.01 * np.sin(2 * np.pi * X) * np.exp(-((Z - 0.5)**2 + (Z + 0.5)**2) / 0.01)
    s = u.copy()
    
    # Fourier space
    kx = 2 * np.pi * fft.fftfreq(nx, d=1/nx)
    kz = 2 * np.pi * fft.fftfreq(nz, d=2/nz)
    KX, KZ = np.meshgrid(kx, kz)
    
    # Initialize Fourier transforms
    u_hat = fft.fft2(u)
    w_hat = fft.fft2(w)
    s_hat = fft.fft2(s)
    
    # Time loop
    for n in range(nt):
        # Nonlinear terms (pseudo-spectral)
        u = np.real(fft.ifft2(u_hat))
        w = np.real(fft.ifft2(w_hat))
        s = np.real(fft.ifft2(s_hat))
        
        N_u = -u * np.real(fft.ifft2(1j * KX * u_hat)) - w * np.real(fft.ifft2(1j * KZ * u_hat))
        N_w = -u * np.real(fft.ifft2(1j * KX * w_hat)) - w * np.real(fft.ifft2(1j * KZ * w_hat))
        N_s = -u * np.real(fft.ifft2(1j * KX * s_hat)) - w * np.real(fft.ifft2(1j * KZ * s_hat))
        
        N_u_hat = fft.fft2(N_u)
        N_w_hat = fft.fft2(N_w)
        N_s_hat = fft.fft2(N_s)
        
        # Pressure term (solve Poisson equation in Fourier space)
        pressure_hat = -(1j * KX * N_u_hat + 1j * KZ * N_w_hat) / (KX**2 + KZ**2 + 1e-12)  # Add small constant to avoid division by zero
        
        # Time step (forward Euler)
        u_hat = u_hat + dt * (-1j * KX * pressure_hat - N_u_hat + nu * (KX**2 + KZ**2) * u_hat)
        w_hat = w_hat + dt * (-1j * KZ * pressure_hat - N_w_hat + nu * (KX**2 + KZ**2) * w_hat)
        s_hat = s_hat + dt * (-N_s_hat + D * (KX**2 + KZ**2) * s_hat)
        
    # Inverse Fourier transform to get the final solution
    u = np.real(fft.ifft2(u_hat))
    w = np.real(fft.ifft2(w_hat))
    s = np.real(fft.ifft2(s_hat))
    
    # Save the variables
    save_dir = '.'  # Current directory
    np.save(os.path.join(save_dir, 'u.npy'), u)
    np.save(os.path.join(save_dir, 'w.npy'), w)
    np.save(os.path.join(save_dir, 's.npy'), s)

if __name__ == "__main__":
    solve_cfd()