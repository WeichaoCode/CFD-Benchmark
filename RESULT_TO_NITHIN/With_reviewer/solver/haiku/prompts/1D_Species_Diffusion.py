import numpy as np

# Parameters
L = 0.1  # domain length [m]
N = 100  # number of control volumes
Gamma = 1e-4  # diffusion coefficient [m^2/s]

# Grid
dx = L/N  # control volume size
x = np.linspace(dx/2, L-dx/2, N)  # cell centers
xf = np.linspace(0, L, N+1)  # face locations

# Initialize arrays
phi = np.zeros(N)  # solution array
a = np.zeros(N)  # coefficient matrix diagonal
b = np.zeros(N-1)  # coefficient matrix lower diagonal
c = np.zeros(N-1)  # coefficient matrix upper diagonal
d = np.zeros(N)  # RHS vector

# Set coefficients
for i in range(1,N-1):
    a[i] = -2*Gamma/dx**2
    b[i] = Gamma/dx**2
    c[i] = Gamma/dx**2
    d[i] = 0

# Boundary conditions
a[0] = 1
d[0] = 10  # phi(0) = 10
a[-1] = 1
d[-1] = 100  # phi(L) = 100

# Solve tridiagonal system
for i in range(1,N):
    w = b[i-1]/a[i-1]
    a[i] = a[i] - w*c[i-1]
    d[i] = d[i] - w*d[i-1]

phi[N-1] = d[N-1]/a[N-1]
for i in range(N-2,-1,-1):
    phi[i] = (d[i] - c[i]*phi[i+1])/a[i]

# Save solution
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/phi_1D_Species_Diffusion.npy', phi)