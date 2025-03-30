import numpy as np

def solve_pde(epsilon, x_start=-5, x_end=5, N_x=101, c=1, T_final=1.0):
    dx = (x_end - x_start) / (N_x - 1)
    x = np.linspace(x_start, x_end, N_x)
    u = np.exp(-x**2)
    
    CFL = 0.8
    if epsilon > 0:
        dt_conv = dx / c
        dt_diff = dx**2 / (2 * epsilon)
        dt = CFL * min(dt_conv, dt_diff)
    else:
        dt = CFL * (dx / c)
    
    Nt = int(T_final / dt) + 1
    dt = T_final / Nt
    
    def compute_f(u):
        u_forward = np.roll(u, -1)
        u_backward = np.roll(u, 1)
        DU_DX = (u_forward - u_backward) / (2 * dx)
        D2U_DX2 = (u_forward - 2 * u + u_backward) / dx**2
        return -c * DU_DX + epsilon * D2U_DX2
    
    f_prev = compute_f(u)
    u_new = u + dt * f_prev
    
    f_current = compute_f(u_new)
    
    for _ in range(2, Nt+1):
        u_next = u_new + (dt / 2) * (3 * f_current - f_prev)
        f_prev, f_current = f_current, compute_f(u_next)
        u_new = u_next
    
    return u_new

u_undamped = solve_pde(epsilon=0)
u_damped = solve_pde(epsilon=5e-4)

np.save('u_undamped.npy', u_undamped)
np.save('u_damped.npy', u_damped)