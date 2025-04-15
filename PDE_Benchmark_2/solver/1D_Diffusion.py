import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import spsolve

# Grid & Time Parameters
Lx = 1     # domain length
nx = 101   # number of points in domain
dt = 0.01  # time step
t_end = 0.5
x = np.linspace(0, Lx, nx) # grid points
dx = x[1] - x[0]  # grid spacing

nu = 0.1  # diffusion coefficient
a_diag = -2./dx**2
c_diag = 1./dt+2*nu/dx**2

# Source function derived from the MMS
f = lambda x, t: np.exp(-t)*(-np.pi**2*np.sin(np.pi*x) - np.pi*np.cos(np.pi*x)) + nu*np.exp(-t)*np.pi**2*np.sin(np.pi*x)

# MMS solution
u_exact = lambda x, t: np.exp(-t)*np.sin(np.pi*x)

# Initial and boundary conditions
u0 = u_exact(x, 0)
lbc = lambda t: u_exact(0, t) 
rbc = lambda t: u_exact(Lx, t) 

# Solution matrix
u = np.zeros((len(t), nx))
u[0, :] = u0  # set initial condition

# Arrays for tridiagonal matrix
abc_data = [np.ones(nx-2), a_diag*np.ones(nx-2), np.ones(nx-2)]
abc_diags = np.array([-1, 0, 1])

# Solve each time step
t = 0.
while t < t_end:
    # Source term + old solution/dt
    rhs = np.zeros(nx, np.float_)
    rhs[1:-1] = f(x[1:-1], t) + u_old[1:-1]/dt + nu*u_old[1:-1]/dx**2
    rhs[0] = lbc(t)
    rhs[-1] = rbc(t)
    A = dia_matrix((abc_data, abc_diags), shape=(nx-2, nx-2)).tocsc()
    u = spsolve(A, rhs[1:-1])
    
    # Update solution
    u_old = np.copy(u)
    t += dt

# Exact solution
u_exact = u_exact(x, t_end)

# Error estimation (absolute error)
error = np.abs(u - u_exact)

# Plot solutions and error
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x, t, u, rstride=1, cstride=1, color='b')
ax.set_title("Numerical Solution")

ax2 = fig.add_subplot(122)
ax2.plot(x, error)
ax2.set_title("Absolute Error")

plt.show()