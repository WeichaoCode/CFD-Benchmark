def compute_dphdx(phi, dx, x):
    dphdx = np.zeros_like(x)
    dphdx[:-1] = (phi[1:] - phi[:-1]) / dx
    dphdx[-1] = (phi[0] - phi[-1]) / dx # For periodic boundary conditions
    return dphdx