import numpy as np

# Parameters
L = 0.5  # Length of domain [m]
k = 1000  # Thermal conductivity [W/m.K]
Q = 2e6   # Heat generation [W/m^3]
T_left = 100  # Left boundary temperature [°C]
T_right = 200 # Right boundary temperature [°C]
N = 100   # Number of control volumes

# Grid
dx = L/N
x_faces = np.linspace(0, L, N+1)  # Face locations
x_centers = (x_faces[1:] + x_faces[:-1])/2  # Cell center locations

# Initialize arrays
A = np.zeros((N,N))  # Coefficient matrix
b = np.zeros(N)      # RHS vector

# Build system of equations
for i in range(N):
    # Interior cells
    if i > 0 and i < N-1:
        A[i,i-1] = k/dx**2  # West coefficient
        A[i,i] = -2*k/dx**2 # Center coefficient
        A[i,i+1] = k/dx**2  # East coefficient
        b[i] = -Q
        
    # Left boundary
    elif i == 0:
        A[i,i] = -2*k/dx**2
        A[i,i+1] = k/dx**2
        b[i] = -Q - T_left*k/dx**2
        
    # Right boundary    
    else:
        A[i,i-1] = k/dx**2
        A[i,i] = -2*k/dx**2
        b[i] = -Q - T_right*k/dx**2

# Solve system
T = np.linalg.solve(A,b)

# Save temperature solution
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/T_1D_Heat_Conduction_With_Source.npy', T)