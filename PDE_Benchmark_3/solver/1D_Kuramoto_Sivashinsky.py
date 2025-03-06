import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def solve_1d_kuramoto_sivashinsky(n_grid=100, max_time=0.05, dt=0.0002):
    # Define PARAMETERS
    n_time = round(max_time / dt)
    x_domain = np.linspace(0, 1, n_grid)
    
    # Compute source term from MMS solution
    t = np.zeros(n_time)
    f = np.zeros((n_time, n_grid))
    for i in range(n_time):
        t[i] = i * dt
        f[i, :] = -(np.exp(-t[i])*np.pi*np.cos(np.pi*x_domain) \
                    + np.exp(-t[i])*(np.pi**2-2*np.pi**4)*np.sin(np.pi*x_domain) \
                    + (-np.pi**2)*np.exp(-2*t[i])*np.sin(2*np.pi*x_domain))

    # compute the initial and boundary conditions from MMS
    u = np.zeros((n_time, n_grid))
    u[0, :] = np.sin(np.pi*x_domain)

    # Apply an upwind scheme using central difference
    for i in range(n_time - 1):
        u_xx = (np.roll(u[i, :], -1) + np.roll(u[i, :], 1) - 2*u[i, :]) / 2
        u_xxxx = (np.roll(u[i, :], -2) - 4*np.roll(u[i, :], -1) + 6*u[i, :] - 4*np.roll(u[i, :], 1) + np.roll(u[i, :], 2)) / 4
        u_x = (np.roll(u[i, :], -1) - u[i, :]) / 2
        u[i+1, :] = u[i, :] + dt * (f[i, :] + u_xx + u_xxxx + 0.5 * u_x**2)
        
    return u, x_domain, t

# Run simulation
u, x_domain, t = solve_1d_kuramoto_sivashinsky()

# Compute exact solution for comparison
u_exact = np.sin(np.pi*x_domain) * np.exp(-t[:, None])

# Error analysis and plot numerical, exact solution and error
err = np.abs(u_exact - u)
print('maximum error:', np.max(err))

plt.figure(figsize=(10,5))
plt.plot(x_domain, u[-1, :], label='Numerical solution')
plt.plot(x_domain, u_exact[-1, :], label='Exact solution')
plt.title("Comparing numerical and exact solutions at t={}".format(t[-1]))
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(x_domain, err[-1, :])
plt.title("Error at t={}".format(t[-1]))
plt.show()