import numpy as np

# Parameters
r_inner = 0.5
r_outer = 10.0
v_inf = 1.0
nu = 0.005
Nr = 50
Ntheta = 100
r = np.linspace(r_inner, r_outer, Nr)
theta = np.linspace(0, 2 * np.pi, Ntheta)
dr = r[1] - r[0]
dtheta = theta[1] - theta[0]
dt = 0.001
T = 1.0

# Initialize variables
psi = np.zeros((Nr, Ntheta))
omega = np.zeros((Nr, Ntheta))

# Initial conditions
# psi[:, :] = 0.0
# omega[:, :] = 0.0

# Boundary conditions
# Inner boundary
psi[0, :] = 20.0
omega[0, :] = 2.0 * (psi[1, :] - psi[0, :]) / dr**2

# Outer boundary
y = r_outer * np.sin(theta)
psi[-1, :] = v_inf * y + 20.0
omega[-1, :] = 0.0

# Periodic boundary condition is handled implicitly by the spatial discretization

def solve_poisson(omega, dr, dtheta, r):
    psi = np.zeros_like(omega)
    
    # Boundary conditions
    psi[0, :] = 20.0
    y = r[-1] * np.sin(theta)
    psi[-1, :] = v_inf * y + 20.0
    
    # Iterate until convergence
    max_iter = 1000
    tolerance = 1e-6
    
    for _ in range(max_iter):
        psi_old = np.copy(psi)
        
        for i in range(1, Nr - 1):
            for j in range(Ntheta):
                jp1 = (j + 1) % Ntheta
                jm1 = (j - 1) % Ntheta
                
                term1 = (psi[i+1, j] + psi[i-1, j]) / dr**2
                term2 = (psi[i+1, j] - psi[i-1, j]) / (2 * r[i] * dr)
                term3 = (psi[i, jp1] + psi[i, jm1]) / (r[i]**2 * dtheta**2)
                
                psi[i, j] = (0.5 / (1/dr**2 + 1/(r[i]**2 * dtheta**2))) * (term1 - omega[i, j] + term2 + term3)
        
        # Apply boundary conditions again to ensure they are satisfied after each iteration
        psi[0, :] = 20.0
        y = r[-1] * np.sin(theta)
        psi[-1, :] = v_inf * y + 20.0
        
        # Check for convergence
        max_diff = np.max(np.abs(psi - psi_old))
        if max_diff < tolerance:
            break
    
    return psi

# Time loop
t = 0.0
while t < T:
    # Calculate velocity components
    u_r = np.zeros_like(psi)
    u_theta = np.zeros_like(psi)
    
    for i in range(Nr):
        for j in range(Ntheta):
            jp1 = (j + 1) % Ntheta
            jm1 = (j - 1) % Ntheta
            
            u_r[i, j] = (1 / r[i]) * (psi[i, jp1] - psi[i, jm1]) / (2 * dtheta)
            
            if i > 0 and i < Nr - 1:
                u_theta[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (2 * dr)
            elif i == 0:
                u_theta[i, j] = -(psi[1, j] - psi[0, j]) / dr
            else:
                u_theta[i, j] = -(psi[i, j] - psi[i-1, j]) / dr
    
    # Solve vorticity transport equation (Forward Euler)
    omega_new = np.zeros_like(omega)
    for i in range(1, Nr - 1):
        for j in range(Ntheta):
            jp1 = (j + 1) % Ntheta
            jm1 = (j - 1) % Ntheta
            
            domega_dt = -u_r[i, j] * (omega[i+1, j] - omega[i-1, j]) / (2 * dr) - \
                        (u_theta[i, j] / r[i]) * (omega[i, jp1] - omega[i, jm1]) / (2 * dtheta) + \
                        nu * ((omega[i+1, j] - 2 * omega[i, j] + omega[i-1, j]) / dr**2 + \
                              (omega[i+1, j] - omega[i-1, j]) / (2 * r[i] * dr) + \
                              (omega[i, jp1] - 2 * omega[i, j] + omega[i, jm1]) / (r[i]**2 * dtheta**2))
            
            omega_new[i, j] = omega[i, j] + dt * domega_dt
    
    # Boundary conditions for omega
    omega_new[0, :] = 2.0 * (psi[1, :] - psi[0, :]) / dr**2
    omega_new[-1, :] = 0.0
    
    omega = omega_new.copy()
    
    # Solve Poisson equation
    psi = solve_poisson(omega, dr, dtheta, r)
    
    t += dt

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/psi_Flow_Past_Circular_Cylinder.npy', psi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/omega_Flow_Past_Circular_Cylinder.npy', omega)