import numpy as np

# Domain parameters
nx = 5  # number of control volumes
dx = 1.0/nx  # control volume size

# Physical parameters
rho = 1.0  # density
u = 2.5    # velocity 
gamma = 0.1  # diffusion coefficient

# Initialize arrays
x = np.linspace(dx/2, 1-dx/2, nx)  # cell centers
phi = np.zeros(nx)
a_e = np.zeros(nx)
a_w = np.zeros(nx)
a_p = np.zeros(nx)
b = np.zeros(nx)

# Calculate coefficients
for i in range(nx):
    # East face
    Fe = rho * u
    De = gamma/dx
    Pe = Fe/De
    if i < nx-1:
        a_e[i] = De * max(0, (1-0.1*abs(Pe))) + max(-Fe, 0)
        
    # West face    
    Fw = rho * u
    Dw = gamma/dx
    Pw = Fw/Dw
    if i > 0:
        a_w[i] = Dw * max(0, (1-0.1*abs(Pw))) + max(Fw, 0)
        
    # Source terms
    if i == 0:  # Left boundary
        b[i] = a_w[i] * 1.0
    elif i == nx-1:  # Right boundary
        b[i] = a_e[i] * 0.0
        
    # Calculate a_p
    a_p[i] = a_w[i] + a_e[i]
    if i == 0:
        a_p[i] += Fe
    elif i == nx-1:
        a_p[i] += Fw

# TDMA solver
def TDMA(a_e, a_w, a_p, b, phi):
    P = np.zeros(len(phi))
    Q = np.zeros(len(phi))
    
    # Forward substitution
    P[0] = a_e[0]/a_p[0]
    Q[0] = b[0]/a_p[0]
    
    for i in range(1, len(phi)):
        denom = a_p[i] - a_w[i]*P[i-1]
        P[i] = a_e[i]/denom
        Q[i] = (b[i] + a_w[i]*Q[i-1])/denom
    
    # Backward substitution
    phi[-1] = Q[-1]
    for i in range(len(phi)-2, -1, -1):
        phi[i] = P[i]*phi[i+1] + Q[i]
    
    return phi

# Solve system
phi = TDMA(a_e, a_w, a_p, b, phi)

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/phi_1D_Convection_Diffusion_Phi.npy', phi)