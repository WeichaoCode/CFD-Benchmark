import numpy as np
import matplotlib.pyplot as plt

# Constants
pi = np.pi
nu = 0.1
rho = 1.0
dt = 0.01
dx = dy = 0.1
x = y = np.arange(0, 1+dx, dx)
t = np.arange(0, 1+dt, dt)
nx, ny, nt = len(x), len(y), len(t)

# MMS solutions
def u_exact(x, y, t): return np.exp(-t) * np.sin(pi*x) * np.sin(pi*y)
def v_exact(x, y, t): return np.exp(-t) * np.cos(pi*x) * np.cos(pi*y)
def p_exact(x, y, t): return np.exp(-t) * np.cos(pi*x) * np.cos(pi*y)

# Initialize variables
u, v, p = np.zeros((nx, ny, nt)), np.zeros((nx, ny, nt)), np.zeros((nx, ny, nt))
u[:,:,0] = u_exact(x[:,None], y[None,:], t[0])
v[:,:,0] = v_exact(x[:,None], y[None,:], t[0])
p[:,:,0] = p_exact(x[:,None], y[None,:], t[0])

# Time stepping
for k in range(nt-1):
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            # Compute derivatives
            du_dx = (u[i+1,j,k] - u[i-1,j,k]) / (2*dx)
            du_dy = (u[i,j+1,k] - u[i,j-1,k]) / (2*dy)
            dv_dx = (v[i+1,j,k] - v[i-1,j,k]) / (2*dx)
            dv_dy = (v[i,j+1,k] - v[i,j-1,k]) / (2*dy)
            dp_dx = (p[i+1,j,k] - p[i-1,j,k]) / (2*dx)
            dp_dy = (p[i,j+1,k] - p[i,j-1,k]) / (2*dy)
            d2u_dx2 = (u[i+1,j,k] - 2*u[i,j,k] + u[i-1,j,k]) / (dx**2)
            d2u_dy2 = (u[i,j+1,k] - 2*u[i,j,k] + u[i,j-1,k]) / (dy**2)
            d2v_dx2 = (v[i+1,j,k] - 2*v[i,j,k] + v[i-1,j,k]) / (dx**2)
            d2v_dy2 = (v[i,j+1,k] - 2*v[i,j,k] + v[i,j-1,k]) / (dy**2)
            # Update variables
            u[i,j,k+1] = u[i,j,k] + dt*(-u[i,j,k]*du_dx - v[i,j,k]*du_dy + nu*(d2u_dx2 + d2u_dy2) - dp_dx/rho)
            v[i,j,k+1] = v[i,j,k] + dt*(-u[i,j,k]*dv_dx - v[i,j,k]*dv_dy + nu*(d2v_dx2 + d2v_dy2) - dp_dy/rho)
            p[i,j,k+1] = p[i,j,k] + dt*(-rho*(du_dx**2 + 2*du_dy*dv_dx + dv_dy**2))

# Compute exact solution
u_exact = u_exact(x[:,None,None], y[None,:,None], t[None,None,:])
v_exact = v_exact(x[:,None,None], y[None,:,None], t[None,None,:])
p_exact = p_exact(x[:,None,None], y[None,:,None], t[None,None,:])

# Compute error
error_u = np.abs(u - u_exact)
error_v = np.abs(v - v_exact)
error_p = np.abs(p - p_exact)

# Plot solution and error at final time
plt.figure(figsize=(12,8))
plt.subplot(231); plt.imshow(u[:,:,-1], origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='jet'); plt.title('Numerical u')
plt.subplot(232); plt.imshow(v[:,:,-1], origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='jet'); plt.title('Numerical v')
plt.subplot(233); plt.imshow(p[:,:,-1], origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='jet'); plt.title('Numerical p')
plt.subplot(234); plt.imshow(error_u[:,:,-1], origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='jet'); plt.title('Error u')
plt.subplot(235); plt.imshow(error_v[:,:,-1], origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='jet'); plt.title('Error v')
plt.subplot(236); plt.imshow(error_p[:,:,-1], origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='jet'); plt.title('Error p')
plt.tight_layout()
plt.show()