import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import diff as psdiff

# Constants
pi = np.pi
exp = np.exp
sin = np.sin
cos = np.cos

# Grid resolution
Nx, Ny = 100, 100
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y)

# Time step size
dt = 0.01
t_end = 1.0
t = np.arange(0, t_end, dt)

# MMS solution
def MMS(t):
    rho = exp(-t) * sin(pi*X) * sin(pi*Y)
    u = exp(-t) * sin(pi*X) * cos(pi*Y)
    v = exp(-t) * cos(pi*X) * sin(pi*Y)
    p = exp(-t) * (1 + sin(pi*X) * sin(pi*Y))
    E = p/(gamma-1) + 0.5*rho*(u**2 + v**2)
    return rho, u, v, E

# Source terms
def source_terms(t):
    rho, u, v, E = MMS(t)
    S_rho = -rho + psdiff(rho*u, period=1.0, axis=0) + psdiff(rho*v, period=1.0, axis=1)
    S_u = -rho*u + psdiff(rho*u**2 + p, period=1.0, axis=0) + psdiff(rho*u*v, period=1.0, axis=1)
    S_v = -rho*v + psdiff(rho*u*v, period=1.0, axis=0) + psdiff(rho*v**2 + p, period=1.0, axis=1)
    S_E = -E + psdiff((E+p)*u, period=1.0, axis=0) + psdiff((E+p)*v, period=1.0, axis=1)
    return S_rho, S_u, S_v, S_E

# Time integration (Euler method)
def solve():
    rho, u, v, E = MMS(0)
    for _t in t:
        S_rho, S_u, S_v, S_E = source_terms(_t)
        rho += dt * S_rho
        u += dt * S_u
        v += dt * S_v
        E += dt * S_E
    return rho, u, v, E

# Solve PDE
rho, u, v, E = solve()

# Compute MMS solution
rho_mms, u_mms, v_mms, E_mms = MMS(t_end)

# Compute absolute error
error_rho = np.abs(rho - rho_mms)
error_u = np.abs(u - u_mms)
error_v = np.abs(v - v_mms)
error_E = np.abs(E - E_mms)

# Plot numerical solution and error
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.contourf(X, Y, rho)
plt.title('Numerical solution for rho')
plt.colorbar()

plt.subplot(222)
plt.contourf(X, Y, error_rho)
plt.title('Error in rho')
plt.colorbar()

plt.subplot(223)
plt.contourf(X, Y, E)
plt.title('Numerical solution for E')
plt.colorbar()

plt.subplot(224)
plt.contourf(X, Y, error_E)
plt.title('Error in E')
plt.colorbar()

plt.tight_layout()
plt.show()