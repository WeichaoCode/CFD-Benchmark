import numpy as np

gamma = 1.4

# Domain setup
x_min = -1.0
x_max = 1.0
Nx = 400
dx = (x_max - x_min) / Nx
x = np.linspace(x_min + 0.5*dx, x_max - 0.5*dx, Nx)

# Initial conditions
rho = np.where(x < 0, 1.0, 0.125)
u = np.zeros(Nx)
p = np.where(x < 0, 1.0, 0.1)
E = p / ((gamma - 1) * rho) + 0.5 * u**2

# Conservative variables
U = np.vstack((rho, rho * u, rho * E))

# Time setup
t = 0.0
t_final = 0.25
CFL = 0.5

def compute_flux(U):
    rho = U[0]
    u = U[1] / rho
    E = U[2] / rho
    p = (gamma - 1) * (rho * E - 0.5 * rho * u**2)
    F = np.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = u * (rho * E + p)
    return F

def primitive(U):
    rho = U[0]
    u = U[1] / rho
    E = U[2] / rho
    p = (gamma - 1) * (rho * E - 0.5 * rho * u**2)
    return rho, u, p

def HLLC_flux(U_L, U_R):
    rho_L, u_L, p_L = primitive(U_L)
    rho_R, u_R, p_R = primitive(U_R)

    E_L = U_L[2] / rho_L
    E_R = U_R[2] / rho_R

    # Compute sound speeds
    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    # Compute wave speeds
    S_L = np.minimum(u_L - a_L, u_R - a_R)
    S_R = np.maximum(u_L + a_L, u_R + a_R)

    # Compute S_M
    numerator = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
    denominator = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    S_M = numerator / denominator

    # Compute fluxes
    F_L = compute_flux(U_L)
    F_R = compute_flux(U_R)

    # Compute U_star_L
    U_star_L = np.zeros_like(U_L)
    U_star_L[0] = rho_L * (S_L - u_L) / (S_L - S_M)
    U_star_L[1] = U_star_L[0] * S_M
    U_star_L[2] = U_star_L[0] * (
        E_L + (S_M - u_L) * (S_M + p_L / (rho_L * (S_L - u_L)))
    )

    # Compute U_star_R
    U_star_R = np.zeros_like(U_R)
    U_star_R[0] = rho_R * (S_R - u_R) / (S_R - S_M)
    U_star_R[1] = U_star_R[0] * S_M
    U_star_R[2] = U_star_R[0] * (
        E_R + (S_M - u_R) * (S_M + p_R / (rho_R * (S_R - u_R)))
    )

    # Initialize flux
    F_HLLC = np.zeros_like(U_L)

    # Condition for flux based on wave speeds
    mask1 = S_L > 0
    mask2 = (S_L <= 0) & (S_M > 0)
    mask3 = S_M <= 0

    F_HLLC[:, mask1] = F_L[:, mask1]
    F_HLLC[:, mask2] = F_L[:, mask2] + S_L[mask2] * (U_star_L[:, mask2] - U_L[:, mask2])
    F_HLLC[:, mask3] = F_R[:, mask3]

    return F_HLLC

while t < t_final:
    rho, u, p = primitive(U)
    a = np.sqrt(gamma * p / rho)
    max_speed = np.max(np.abs(u) + a)
    dt = CFL * dx / max_speed
    if t + dt > t_final:
        dt = t_final - t
    # Apply boundary conditions (reflective)
    U_ext = np.hstack((U[:,0:1], U, U[:,-1:]))
    U_ext[1,0] = -U_ext[1,1]
    U_ext[0,0] = U_ext[0,1]
    U_ext[2,0] = U_ext[2,1]
    U_ext[1,-1] = -U_ext[1,-2]
    U_ext[0,-1] = U_ext[0,-2]
    U_ext[2,-1] = U_ext[2,-2]
    # Compute fluxes at interfaces
    F_half = HLLC_flux(U_ext[:, :-1], U_ext[:,1:])
    # Update U
    U = U - (dt / dx) * (F_half[:,1:] - F_half[:,:-1])
    t += dt

# Compute final primitive variables
rho, u, p = primitive(U)

# Save variables
np.save('rho.npy', rho)
np.save('u.npy', u)
np.save('p.npy', p)