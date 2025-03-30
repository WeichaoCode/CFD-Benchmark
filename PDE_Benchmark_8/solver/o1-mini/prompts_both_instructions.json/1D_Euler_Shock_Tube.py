import numpy as np

# Parameters
gamma = 1.4
Nx = 81
x_min = -1.0
x_max = 1.0
L = x_max - x_min
dx = L / (Nx - 1)
CFL = 1.0
t_final = 0.25

# Grid
x = np.linspace(x_min, x_max, Nx)

# Initial Conditions
rho = np.where(x < 0, 1.0, 0.125)
u = np.zeros(Nx)
p = np.where(x < 0, 1.0, 0.1)

# Conservative Variables
E = p / ((gamma - 1) * rho) + 0.5 * u**2
U = np.vstack((rho, rho * u, rho * E))

def compute_flux(U):
    rho = U[0]
    u = U[1] / rho
    E = U[2] / rho
    p = (gamma - 1) * rho * (E - 0.5 * u**2)
    F = np.vstack((rho * u, rho * u**2 + p, u * (rho * E + p)))
    return F

def apply_boundary(U):
    U[:,0] = U[:,1]
    U[:,-1] = U[:,-2]
    return U

F = compute_flux(U)

t = 0.0
while t < t_final:
    # Compute time step
    rho = U[0]
    u = U[1] / rho
    E = U[2] / rho
    p = (gamma - 1) * rho * (E - 0.5 * u**2)
    a = np.sqrt(gamma * p / rho)
    dt = CFL * dx / np.max(np.abs(u) + a)
    if t + dt > t_final:
        dt = t_final - t
    # Predictor step
    U_predict = U.copy()
    F = compute_flux(U)
    U_predict[:, :-1] = U[:, :-1] - dt/dx * (F[:,1:] - F[:, :-1])
    U_predict = apply_boundary(U_predict)
    # Compute flux at predictor step
    F_predict = compute_flux(U_predict)
    # Corrector step
    U[:,1:-1] = 0.5 * (U[:,1:-1] + U_predict[:,1:-1] - dt/dx * (F_predict[:,1:-1] - F_predict[:,0:-2]))
    U = apply_boundary(U)
    t += dt

# Final primitive variables
rho = U[0]
u = U[1] / rho
E = U[2] / rho
p = (gamma - 1) * rho * (E - 0.5 * u**2)
F = compute_flux(U)

# Save variables
np.save('rho.npy', rho)
np.save('u.npy', u)
np.save('p.npy', p)
np.save('U.npy', U)
np.save('F.npy', F)