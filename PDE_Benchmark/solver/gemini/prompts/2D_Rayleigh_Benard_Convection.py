import numpy as np

def solve_cfd():
    # Problem parameters
    Lx = 4.0
    Lz = 1.0
    Ra = 2e6
    Pr = 1.0
    nu = (Ra/Pr)**(-0.5)
    kappa = (Ra*Pr)**(-0.5)
    
    # Numerical parameters
    nx = 64
    nz = 32
    dt = 0.001
    t_final = 50.0
    
    # Grid
    x = np.linspace(0, Lx, nx)
    z = np.linspace(0, Lz, nz)
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    X, Z = np.meshgrid(x, z)
    
    # Initial conditions
    u = np.zeros_like(X)
    w = np.zeros_like(X)
    b = Lz - Z + 0.01 * np.random.rand(nz, nx)
    
    # Time loop
    t = 0.0
    while t < t_final:
        # Nonlinear terms (Advection)
        u_grad_u = u * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dx) + w * (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dz)
        u_grad_w = u * (np.roll(w, -1, axis=1) - np.roll(w, 1, axis=1)) / (2*dx) + w * (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) / (2*dz)
        u_grad_b = u * (np.roll(b, -1, axis=1) - np.roll(b, 1, axis=1)) / (2*dx) + w * (np.roll(b, -1, axis=0) - np.roll(b, 1, axis=0)) / (2*dz)
        
        # Diffusion terms
        d2u_dx2 = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2
        d2u_dz2 = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dz**2
        d2w_dx2 = (np.roll(w, -1, axis=1) - 2*w + np.roll(w, 1, axis=1)) / dx**2
        d2w_dz2 = (np.roll(w, -1, axis=0) - 2*w + np.roll(w, 1, axis=0)) / dz**2
        d2b_dx2 = (np.roll(b, -1, axis=1) - 2*b + np.roll(b, 1, axis=1)) / dx**2
        d2b_dz2 = (np.roll(b, -1, axis=0) - 2*b + np.roll(b, 1, axis=0)) / dz**2

        # Update velocities and buoyancy (explicit Euler)
        u = u + dt * (-u_grad_u + nu * (d2u_dx2 + d2u_dz2))
        w = w + dt * (-u_grad_w + nu * (d2w_dx2 + d2w_dz2) + b)
        b = b + dt * (-u_grad_b + kappa * (d2b_dx2 + d2b_dz2))
        
        # Boundary conditions
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]
        w[:, 0] = w[:, -2]
        w[:, -1] = w[:, 1]
        b[:, 0] = b[:, -2]
        b[:, -1] = b[:, 1]
        
        u[0, :] = 0
        u[-1, :] = 0
        w[0, :] = 0
        w[-1, :] = 0
        b[0, :] = Lz
        b[-1, :] = 0
        
        t += dt
    
    # Save the final solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Rayleigh_Benard_Convection.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/w_2D_Rayleigh_Benard_Convection.npy', w)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/b_2D_Rayleigh_Benard_Convection.npy', b)
    
if __name__ == "__main__":
    solve_cfd()