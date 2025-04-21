import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def rhs_momentum(u, w, b, nu, dx, dz, Nx, Nz):
    # Allocate arrays for right-hand side
    u_rhs = np.zeros_like(u)
    w_rhs = np.zeros_like(w)
    
    # Compute derivatives using central differences
    for i in range(1, Nx-1):
        for j in range(1, Nz-1):
            # u-momentum 
            u_adv_x = u[i,j] * (u[i+1,j] - u[i-1,j]) / (2*dx)
            u_adv_z = w[i,j] * (u[i,j+1] - u[i,j-1]) / (2*dz)
            
            u_diff_x = nu * (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / (dx**2)
            u_diff_z = nu * (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / (dz**2)
            
            u_rhs[i,j] = -u_adv_x - u_adv_z + u_diff_x + u_diff_z
            
            # w-momentum
            w_adv_x = u[i,j] * (w[i+1,j] - w[i-1,j]) / (2*dx)
            w_adv_z = w[i,j] * (w[i,j+1] - w[i,j-1]) / (2*dz)
            
            w_diff_x = nu * (w[i+1,j] - 2*w[i,j] + w[i-1,j]) / (dx**2)
            w_diff_z = nu * (w[i,j+1] - 2*w[i,j] + w[i,j-1]) / (dz**2)
            
            w_rhs[i,j] = -w_adv_x - w_adv_z + w_diff_x + w_diff_z + b[i,j]
    
    return u_rhs, w_rhs

@njit
def rhs_buoyancy(u, w, b, kappa, dx, dz, Nx, Nz):
    # Allocate array for buoyancy RHS
    b_rhs = np.zeros_like(b)
    
    # Compute derivatives using central differences
    for i in range(1, Nx-1):
        for j in range(1, Nz-1):
            # Advection terms
            b_adv_x = u[i,j] * (b[i+1,j] - b[i-1,j]) / (2*dx)
            b_adv_z = w[i,j] * (b[i,j+1] - b[i,j-1]) / (2*dz)
            
            # Diffusion terms
            b_diff_x = kappa * (b[i+1,j] - 2*b[i,j] + b[i-1,j]) / (dx**2)
            b_diff_z = kappa * (b[i,j+1] - 2*b[i,j] + b[i,j-1]) / (dz**2)
            
            b_rhs[i,j] = -b_adv_x - b_adv_z + b_diff_x + b_diff_z
    
    return b_rhs

def solve_convection_diffusion():
    # Problem parameters
    Lx, Lz = 4.0, 1.0
    Ra, Pr = 2e6, 1.0
    
    # Numerical parameters
    Nx, Nz = 128, 32
    dx, dz = Lx / (Nx-1), Lz / (Nz-1)
    
    # Derived parameters
    nu = (Ra/Pr)**(-0.5)
    kappa = (Ra*Pr)**(-0.5)
    
    # Time stepping
    dt = 0.001
    t_end = 50.0
    
    # Initialize fields
    x = np.linspace(0, Lx, Nx)
    z = np.linspace(0, Lz, Nz)
    
    # Initial conditions with small random perturbation
    u = np.zeros((Nx, Nz))
    w = np.zeros((Nx, Nz))
    b = np.tile(Lz - z, (Nx, 1)) + 0.01 * np.random.randn(Nx, Nz)
    
    # Periodic boundary conditions
    u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = 0
    w[0,:] = w[-1,:] = w[:,0] = w[:,-1] = 0
    b[0,:] = b[-1,:] = b[:,0] = b[:,-1] = 0
    
    # Time integration
    t = 0
    while t < t_end:
        # RHS computations
        u_rhs, w_rhs = rhs_momentum(u, w, b, nu, dx, dz, Nx, Nz)
        b_rhs = rhs_buoyancy(u, w, b, kappa, dx, dz, Nx, Nz)
        
        # Update fields
        u += dt * u_rhs
        w += dt * w_rhs
        b += dt * b_rhs
        
        t += dt
    
    # Save final solutions
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_2D_Rayleigh_Benard_Convection.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/w_2D_Rayleigh_Benard_Convection.npy', w)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/b_2D_Rayleigh_Benard_Convection.npy', b)

# Run the simulation
solve_convection_diffusion()