import numpy as np

def solve_navier_stokes_2d():
    # Parameters
    nx, ny = 41, 41
    nt = 10
    dx = dy = 2 / 40
    dt = 0.01
    rho = 1.0
    nu = 0.1
    F = 1.0

    # Initialize arrays
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    # Computational loop
    for _ in range(nt):
        # Create temporary arrays for update
        un = u.copy()
        vn = v.copy()
        pn = p.copy()

        # Compute derivatives using central differences
        # u-momentum equation
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] 
                         - un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
                         - vn[1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
                         + nu * dt/dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, 0:-2])
                         + nu * dt/dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[0:-2, 1:-1])
                         + F * dt)

        # v-momentum equation
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] 
                         - un[1:-1, 1:-1] * dt/dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
                         - vn[1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
                         + nu * dt/dx**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
                         + nu * dt/dy**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))

        # Pressure Poisson equation
        for _ in range(50):  # Iterative solution
            pn = p.copy()
            p[1:-1, 1:-1] = (0.25 * (pn[1:-1, 2:] + pn[1:-1, 0:-2] + 
                                      pn[2:, 1:-1] + pn[0:-2, 1:-1]) 
                              - dx**2 / (4 * rho) * (
                                  (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 
                              - dx**2 / (4 * rho) * (
                                  (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2)

        # Apply periodic boundary conditions
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]
        v[:, 0] = v[:, -2]
        v[:, -1] = v[:, 1]
        p[:, 0] = p[:, -2]
        p[:, -1] = p[:, 1]

        # No-slip boundary conditions
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

    # Save final solutions
    np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_2D_Navier_Stokes_Channel.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/v_2D_Navier_Stokes_Channel.npy', v)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/p_2D_Navier_Stokes_Channel.npy', p)

solve_navier_stokes_2d()