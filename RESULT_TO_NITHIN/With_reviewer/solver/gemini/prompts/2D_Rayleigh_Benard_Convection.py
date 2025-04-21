import numpy as np

def solve_cfd():
    # Problem parameters
    Lx = 4.0
    Lz = 1.0
    Ra = 2e6
    Pr = 1.0
    nu = (Ra/Pr)**(-0.5)
    kappa = (Ra*Pr)**(-0.5)
    t_final = 50.0

    # Numerical parameters
    nx = 64
    nz = 32
    dt = 0.001
    nt = int(t_final / dt)

    # Grid
    x = np.linspace(0, Lx, nx)
    z = np.linspace(0, Lz, nz)
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    X, Z = np.meshgrid(x, z)

    # Initial conditions
    u = np.zeros((nz, nx))
    w = np.zeros((nz, nx))
    b = Lz - Z + 0.01 * np.random.rand(nz, nx)

    # Functions for derivatives (central difference)
    def dudx(u):
        du = np.zeros_like(u)
        du[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        du[:, 0] = (u[:, 1] - u[:, -1]) / (2 * dx)  # Periodic BC
        du[:, -1] = (u[:, 0] - u[:, -2]) / (2 * dx) # Periodic BC
        return du

    def dudz(u):
        du = np.zeros_like(u)
        du[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dz)
        du[0, :] = (u[1, :] - u[-1, :]) / (2 * dz)
        du[-1, :] = (u[0, :] - u[-2, :]) / (2 * dz)
        return du

    def d2udx2(u):
        d2u = np.zeros_like(u)
        d2u[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dx**2
        d2u[:, 0] = (u[:, 1] - 2 * u[:, 0] + u[:, -1]) / dx**2
        d2u[:, -1] = (u[:, 0] - 2 * u[:, -1] + u[:, -2]) / dx**2
        return d2u

    def d2udz2(u):
        d2u = np.zeros_like(u)
        d2u[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / dz**2
        d2u[0, :] = (u[1, :] - 2 * u[0, :] + u[-1, :]) / dz**2
        d2u[-1, :] = (u[0, :] - 2 * u[-1, :] + u[-2, :]) / dz**2
        return d2u

    # Time loop
    for n in range(nt):
        # Nonlinear terms
        adv_u = u * dudx(u) + w * dudz(u)
        adv_w = u * dudx(w) + w * dudz(w)
        adv_b = u * dudx(b) + w * dudz(b)

        # Viscous and diffusion terms
        visc_u = nu * (d2udx2(u) + d2udz2(u))
        visc_w = nu * (d2udx2(w) + d2udz2(w))
        diff_b = kappa * (d2udx2(b) + d2udz2(b))

        # Update velocity and buoyancy (explicit Euler)
        u = u - dt * adv_u + dt * visc_u
        w = w - dt * adv_w + dt * visc_w + dt * b
        b = b - dt * adv_b + dt * diff_b

        # Boundary conditions
        u[0, :] = 0
        u[-1, :] = 0
        w[0, :] = 0
        w[-1, :] = 0
        b[0, :] = Lz
        b[-1, :] = 0

    # Save the final solution
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Rayleigh_Benard_Convection.npy', u)
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/w_2D_Rayleigh_Benard_Convection.npy', w)
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/b_2D_Rayleigh_Benard_Convection.npy', b)

solve_cfd()