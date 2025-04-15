import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters
L  = 1.0  # size of the domain
Nx = 50   # number of grid points in x and y direction
Nt = 500  # number of time steps
dt = 0.01 # time step
c  = 1.0  # speed of sound

x, dx = np.linspace(0, L, Nx, retstep=True)
y = x.copy()
t = np.linspace(0, dt*Nt, Nt+1)

# Step 2: Check CFL condition for stability
assert c*dt/dx < 1.0, "CFL condition violated: reduce dt or increase dx"

# Step 3: Compute source term from MMS solution
def compute_source_mms(x, y, t):
    rho = np.exp(-t) * np.sin(np.pi*x)[:,None] * np.sin(np.pi*y)[None,:]
    u = np.exp(-t) * np.sin(np.pi*x)[:,None] * np.cos(np.pi*y)[None,:]
    v = np.exp(-t) * np.cos(np.pi*x)[:,None] * np.sin(np.pi*y)[None,:]
    p = np.exp(-t) * (1 + np.sin(np.pi*x)[:,None] * np.sin(np.pi*y)[None,:])
    E = p/(c**2) + 0.5*rho*(u**2 + v**2)
    return rho, u, v, E

# Step 4: Compute the initial and boundary conditions from MMS
rho, u, v, E = compute_source_mms(x, y, t[0])

# Step 5: Solve the PDE using finite difference
for j in range(1, Nt+1):
    rhoN, uN, vN, EN = compute_source_mms(x, y, t[j])

    u_interior = u[1:-1,1:-1]
    v_interior = v[1:-1,1:-1]
    p_interior = (c**2)*(E[1:-1,1:-1] - 0.5*rho[1:-1,1:-1]*(u_interior**2 + v_interior**2))

    # compute interior with central/upwind scheme
    rho[1:-1,1:-1] = rhoN[1:-1,1:-1] -   dt/dx * \
        (uN[2:,1:-1] * rhoN[2:,1:-1] - uN[:-2,1:-1] * rhoN[:-2,1:-1]) - \
        dt/dx * (vN[1:-1,2:] * rhoN[1:-1,2:] - vN[1:-1,:-2] * rhoN[1:-1,:-2])

# Step 6: Compute exact solution for comparison
rho_exact, u_exact, v_exact, E_exact = compute_source_mms(x, y, t[-1])
errors = np.sqrt((rho - rho_exact)**2)

# Step 7: Error analysis and create plots
print('RMS error:', np.sqrt(np.mean(errors**2)))

plt.figure()
plt.subplot(1,2,1)
plt.pcolormesh(x, y, rho)
plt.title('Numerical solution')

plt.subplot(1,2,2)
plt.pcolormesh(x, y, rho_exact)
plt.title('Exact solution')

plt.figure()
plt.pcolormesh(x, y, errors)
plt.title('Errors')
plt.show()