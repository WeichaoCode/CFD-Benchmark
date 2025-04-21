import numpy as np

def solve_cfd():
    # Parameters
    nx = 64
    nz = 128
    nt = 200
    nu = 1 / (5 * 10000)
    D = nu / 1
    dx = 1 / nx
    dz = 2 / nz
    dt = 0.01

    # Domain
    x = np.linspace(0, 1, nx)
    z = np.linspace(-1, 1, nz)
    X, Z = np.meshgrid(x, z)

    # Initial conditions
    u = 0.5 * (1 + np.tanh((Z - 0.5) / 0.1) - np.tanh((Z + 0.5) / 0.1))
    w = 0.01 * np.sin(2 * np.pi * X) * np.exp(-((Z - 0.5)**2 + (Z + 0.5)**2) / 0.01)
    s = np.copy(u)
    p = np.zeros_like(u)

    # Functions for derivatives (central difference)
    def du_dx(u):
        u_padded = np.pad(u, ((0, 0), (1, 1)), mode='wrap')
        return (u_padded[:, 2:] - u_padded[:, :-2]) / (2 * dx)

    def du_dz(u):
        u_padded = np.pad(u, ((1, 1), (0, 0)), mode='wrap')
        return (u_padded[2:, :] - u_padded[:-2, :]) / (2 * dz)

    def d2u_dx2(u):
        u_padded = np.pad(u, ((0, 0), (1, 1)), mode='wrap')
        return (u_padded[:, 2:] - 2 * u + u_padded[:, :-2]) / (dx**2)

    def d2u_dz2(u):
        u_padded = np.pad(u, ((1, 1), (0, 0)), mode='wrap')
        return (u_padded[2:, :] - 2 * u + u_padded[:-2, :]) / (dz**2)

    # Time loop
    for n in range(nt):
        # Advection terms
        adv_u = u * du_dx(u) + w * du_dz(u)
        adv_w = u * du_dx(w) + w * du_dz(w)
        adv_s = u * du_dx(s) + w * du_dz(s)

        # Diffusion terms
        diff_u = nu * (d2u_dx2(u) + d2u_dz2(u))
        diff_w = nu * (d2u_dx2(w) + d2u_dz2(w))
        diff_s = D * (d2u_dx2(s) + d2u_dz2(s))

        # Update velocities and tracer
        u = u - dt * adv_u - dt * du_dx(p) + dt * diff_u
        w = w - dt * adv_w - dt * du_dz(p) + dt * diff_w
        s = s - dt * adv_s + dt * diff_s

        # Pressure correction (simple approach, not solving Poisson equation)
        div_uw = du_dx(u) + du_dz(w)
        p = p - dt * div_uw

    # Save the final solution
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)
    np.save('/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Shear_Flow_With_Tracer.npy', p)

solve_cfd()