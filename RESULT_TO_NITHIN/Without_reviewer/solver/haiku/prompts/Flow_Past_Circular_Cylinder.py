import numpy as np

# Parameters
nr = 100  # Number of points in r direction
ntheta = 120  # Number of points in theta direction
r_min, r_max = 0.5, 10.0
dr = (r_max - r_min) / (nr-1)
dtheta = 2*np.pi / ntheta
nu = 0.005
v_inf = 1.0
dt = 0.001
t_final = 2.0
nt = int(t_final/dt)

# Grid
r = np.linspace(r_min, r_max, nr)
theta = np.linspace(0, 2*np.pi, ntheta)
R, THETA = np.meshgrid(r, theta)
X = R * np.cos(THETA)
Y = R * np.sin(THETA)

# Initialize fields
psi = np.zeros((ntheta, nr))
omega = np.zeros((ntheta, nr))

# Time stepping
for n in range(nt):
    # Store old values
    psi_old = psi.copy()
    omega_old = omega.copy()
    
    # Solve Poisson equation for streamfunction
    for _ in range(50):  # Inner iteration
        for i in range(1, ntheta-1):
            for j in range(1, nr-1):
                # Finite difference coefficients for Poisson equation in polar coordinates
                psi[i,j] = ((psi[i+1,j] + psi[i-1,j])/(dtheta**2) + 
                           (psi[i,j+1]*(1 + dr/(2*r[j])) + psi[i,j-1]*(1 - dr/(2*r[j])))/(dr**2))/
                           (2/(dtheta**2) + 2/(dr**2)) + omega[i,j]/(2/(dtheta**2) + 2/(dr**2))
    
        # Boundary conditions for psi
        psi[:,0] = 20  # Inner cylinder
        psi[:,-1] = v_inf * R[:,-1] * np.sin(THETA[:,-1]) + 20  # Outer boundary
        psi[0,:] = psi[-1,:]  # Periodic in theta
        
    # Calculate velocities
    ur = np.zeros_like(psi)
    utheta = np.zeros_like(psi)
    for i in range(1, ntheta-1):
        for j in range(1, nr-1):
            ur[i,j] = (psi[i+1,j] - psi[i-1,j])/(2*dtheta*r[j])
            utheta[i,j] = -(psi[i,j+1] - psi[i,j-1])/(2*dr)
    
    # Solve vorticity transport equation
    for i in range(1, ntheta-1):
        for j in range(1, nr-1):
            # Finite difference for vorticity transport
            omega[i,j] = omega_old[i,j] - dt*(
                ur[i,j]*(omega_old[i,j+1] - omega_old[i,j-1])/(2*dr) +
                utheta[i,j]*(omega_old[i+1,j] - omega_old[i-1,j])/(2*dtheta*r[j])
            ) + nu*dt*(
                (omega_old[i,j+1] - 2*omega_old[i,j] + omega_old[i,j-1])/(dr**2) +
                (omega_old[i+1,j] - 2*omega_old[i,j] + omega_old[i-1,j])/(r[j]**2*dtheta**2) +
                (omega_old[i,j+1] - omega_old[i,j-1])/(2*r[j]*dr)
            )
    
    # Boundary conditions for omega
    omega[:,0] = 2*(psi[:,0] - psi[:,1])/dr**2  # Inner cylinder
    omega[:,-1] = 0  # Outer boundary
    omega[0,:] = omega[-1,:]  # Periodic in theta

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/psi_Flow_Past_Circular_Cylinder.npy', psi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/omega_Flow_Past_Circular_Cylinder.npy', omega)