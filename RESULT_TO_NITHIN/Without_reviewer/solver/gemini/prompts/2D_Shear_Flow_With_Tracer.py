import numpy as np

def solve_cfd():
    # Parameters
    nx = 64
    nz = 128
    nt = 200
    dt = 0.1
    nu = 1 / (5 * 10**4)
    D = nu / 1
    
    # Domain
    x = np.linspace(0, 1, nx)
    z = np.linspace(-1, 1, nz)
    X, Z = np.meshgrid(x, z)
    
    # Initial conditions
    u = 0.5 * (1 + np.tanh((Z - 0.5) / 0.1) - np.tanh((Z + 0.5) / 0.1))
    w = 0.01 * np.sin(2 * np.pi * X) * np.exp(-((Z - 0.5)**2 + (Z + 0.5)**2) / 0.01)
    s = u.copy()
    p = np.zeros_like(u)
    
    # Finite difference functions (periodic boundary conditions)
    def laplacian(f):
        f_xx = (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / (x[1] - x[0])**2
        f_zz = (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / (z[1] - z[0])**2
        return f_xx + f_zz

    def gradient(f):
        f_x = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * (x[1] - x[0]))
        f_z = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * (z[1] - z[0]))
        return f_x, f_z

    def divergence(u, w):
        u_x = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * (x[1] - x[0]))
        w_z = (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) / (2 * (z[1] - z[0]))
        return u_x + w_z

    def advection(u, w, f):
        f_x = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * (x[1] - x[0]))
        f_z = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * (z[1] - z[0]))
        return u * f_x + w * f_z

    # Time loop
    for n in range(nt):
        # Momentum equation
        u_new = u + dt * (-advection(u, w, u) - gradient(p)[0] + nu * laplacian(u))
        w_new = w + dt * (-advection(u, w, w) - gradient(p)[1] + nu * laplacian(w))
        
        # Tracer equation
        s_new = s + dt * (-advection(u, w, s) + D * laplacian(s))

        # Incompressibility constraint (Pressure Poisson equation)
        rhs = divergence(u_new, w_new) / dt
        
        # Solve pressure poisson equation using FFT
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=x[1]-x[0])
        kz = 2 * np.pi * np.fft.fftfreq(nz, d=z[1]-z[0])
        KX, KZ = np.meshgrid(kx, kz)
        
        rhs_hat = np.fft.fft2(rhs)
        p_hat = rhs_hat / (KX**2 + KZ**2 + 1e-10)  # Adding a small constant to avoid division by zero
        p_hat[0, 0] = 0  # Set the DC component to zero
        p = np.fft.ifft2(p_hat).real

        # Correct velocity field
        u = u_new - dt * gradient(p)[0]
        w = w_new - dt * gradient(p)[1]
        s = s_new

    # Save the final solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Shear_Flow_With_Tracer.npy', p)

solve_cfd()