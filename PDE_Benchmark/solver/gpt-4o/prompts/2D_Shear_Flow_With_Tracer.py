import numpy as np
from scipy.fftpack import fft2, ifft2

# Parameters
Lx, Lz = 1.0, 2.0  # Domain size
Nx, Nz = 128, 256  # Number of grid points
dx, dz = Lx / Nx, Lz / Nz
x = np.linspace(0, Lx, Nx, endpoint=False)
z = np.linspace(-Lz/2, Lz/2, Nz, endpoint=False)
X, Z = np.meshgrid(x, z)

nu = 1 / 5e4  # Kinematic viscosity
D = nu  # Tracer diffusivity
dt = 0.01  # Time step
T = 20.0  # Final time
Nt = int(T / dt)  # Number of time steps

# Initial conditions
u = 0.5 * (1 + np.tanh((Z - 0.5) / 0.1) - np.tanh((Z + 0.5) / 0.1))
w = 0.01 * np.sin(2 * np.pi * X) * (np.exp(-((Z - 0.5) ** 2) / 0.01) + np.exp(-((Z + 0.5) ** 2) / 0.01))
s = u.copy()

# Wavenumbers for Fourier transform
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
KX, KZ = np.meshgrid(kx, kz)
K2 = KX**2 + KZ**2
K2[0, 0] = 1.0  # Avoid division by zero

# Time-stepping loop
for n in range(Nt):
    # Fourier transform of velocity and tracer
    u_hat = fft2(u)
    w_hat = fft2(w)
    s_hat = fft2(s)

    # Compute nonlinear terms in physical space
    u_x = np.real(ifft2(1j * KX * u_hat))
    u_z = np.real(ifft2(1j * KZ * u_hat))
    w_x = np.real(ifft2(1j * KX * w_hat))
    w_z = np.real(ifft2(1j * KZ * w_hat))
    s_x = np.real(ifft2(1j * KX * s_hat))
    s_z = np.real(ifft2(1j * KZ * s_hat))

    # Nonlinear terms
    nonlinear_u = np.nan_to_num(u * u_x + w * u_z, nan=0.0, posinf=0.0, neginf=0.0)
    nonlinear_w = np.nan_to_num(u * w_x + w * w_z, nan=0.0, posinf=0.0, neginf=0.0)
    nonlinear_s = np.nan_to_num(u * s_x + w * s_z, nan=0.0, posinf=0.0, neginf=0.0)

    # Fourier transform of nonlinear terms
    nonlinear_u_hat = fft2(nonlinear_u)
    nonlinear_w_hat = fft2(nonlinear_w)
    nonlinear_s_hat = fft2(nonlinear_s)

    # Update velocity in Fourier space
    u_hat = (u_hat - dt * nonlinear_u_hat) / (1 + nu * dt * K2)
    w_hat = (w_hat - dt * nonlinear_w_hat) / (1 + nu * dt * K2)

    # Update tracer in Fourier space
    s_hat = (s_hat - dt * nonlinear_s_hat) / (1 + D * dt * K2)

    # Transform back to physical space
    u = np.real(ifft2(u_hat))
    w = np.real(ifft2(w_hat))
    s = np.real(ifft2(s_hat))

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)