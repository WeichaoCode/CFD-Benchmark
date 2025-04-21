import numpy as np

# Parameters
gamma = 1.4
nx = 400  # Number of spatial points
dx = 2.0/nx  # Spatial step size
x = np.linspace(-1, 1, nx)
cfl = 0.8
t_end = 0.25

# Initialize arrays
rho = np.zeros(nx)
u = np.zeros(nx)
p = np.zeros(nx)
E = np.zeros(nx)

# Set initial conditions
rho[x < 0] = 1.0
rho[x >= 0] = 0.125
p[x < 0] = 1.0
p[x >= 0] = 0.1
u[:] = 0.0
E[:] = p/(gamma-1)/rho + 0.5*u*u

# Conservative variables
U = np.zeros((3, nx))
U[0] = rho
U[1] = rho*u
U[2] = rho*E

def compute_flux(U):
    # Compute primitive variables
    rho = U[0]
    u = U[1]/rho
    E = U[2]/rho
    p = (gamma-1)*rho*(E - 0.5*u*u)
    
    # Compute flux
    F = np.zeros_like(U)
    F[0] = rho*u
    F[1] = rho*u*u + p
    F[2] = u*(rho*E + p)
    return F

def lax_friedrichs_step(U, dx, dt):
    F = compute_flux(U)
    U_new = np.zeros_like(U)
    
    # Interior points
    U_new[:,1:-1] = 0.5*(U[:,2:] + U[:,:-2]) - 0.5*dt/dx*(F[:,2:] - F[:,:-2])
    
    # Reflective boundary conditions
    U_new[:,0] = U_new[:,1]
    U_new[1,0] = -U_new[1,1]  # Reflect momentum
    U_new[:,-1] = U_new[:,-2]
    U_new[1,-1] = -U_new[1,-2]  # Reflect momentum
    
    return U_new

# Time stepping
t = 0
while t < t_end:
    # Compute time step
    rho = U[0]
    u = U[1]/rho
    E = U[2]/rho
    p = (gamma-1)*rho*(E - 0.5*u*u)
    c = np.sqrt(gamma*p/rho)
    dt = cfl*dx/np.max(np.abs(u) + c)
    
    if t + dt > t_end:
        dt = t_end - t
        
    # Update solution
    U = lax_friedrichs_step(U, dx, dt)
    t += dt

# Compute final primitive variables
rho = U[0]
u = U[1]/rho
E = U[2]/rho
p = (gamma-1)*rho*(E - 0.5*u*u)

# Save results
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/rho_1D_Euler_Shock_Tube.npy', rho)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_1D_Euler_Shock_Tube.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/p_1D_Euler_Shock_Tube.npy', p)