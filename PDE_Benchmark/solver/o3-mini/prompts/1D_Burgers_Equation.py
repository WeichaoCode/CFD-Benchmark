#!/usr/bin/env python3
import numpy as np

# Parameters
nu = 0.07
L = 2*np.pi
T_final = 0.14 * np.pi
Nx = 256
x = np.linspace(0, L, Nx, endpoint=False)
dx = L / Nx

# Initial condition:
# phi = exp(-x^2/(4*nu)) + exp(-((x-2*pi)^2)/(4*nu))
phi = np.exp(-x**2/(4*nu)) + np.exp(-((x - L)**2)/(4*nu))
# dphi/dx = -x/(2*nu)*exp(-x^2/(4*nu)) - (x-2*pi)/(2*nu)*exp(-((x-2*pi)^2)/(4*nu))
dphi_dx = -x/(2*nu)*np.exp(-x**2/(4*nu)) - (x - L)/(2*nu)*np.exp(-((x - L)**2)/(4*nu))
u = -2*nu/phi * dphi_dx + 4

# Define derivative functions with periodic BC
def ddx(u):
    return (np.roll(u, -1) - np.roll(u, 1))/(2*dx)

def d2dx2(u):
    return (np.roll(u, -1) - 2*u + np.roll(u, 1))/(dx**2)

# Right-hand side of the PDE: ∂u/∂t = - u ∂u/∂x + nu ∂²u/∂x²
def rhs(u):
    return - u * ddx(u) + nu * d2dx2(u)

# Determine time step based on CFL conditions
# CFL for convection: dt_convection ~ dx / max(|u|)
# CFL for diffusion: dt_diffusion ~ dx**2/(2*nu)
umax = np.max(np.abs(u))
dt_conv = dx / (umax + 1e-10)
dt_diff = dx**2 / (2*nu)
dt = 0.5 * min(dt_conv, dt_diff)

# Adjust number of time steps to reach T_final
Nt = int(np.ceil(T_final/dt))
dt = T_final / Nt  # re-adjust dt exactly

# Time integration using RK4
for n in range(Nt):
    k1 = rhs(u)
    k2 = rhs(u + 0.5*dt*k1)
    k3 = rhs(u + 0.5*dt*k2)
    k4 = rhs(u + dt*k3)
    u = u + dt*(k1 + 2*k2 + 2*k3 + k4)/6

# Save final solution as a 1D array in "u.npy"
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_1D_Burgers_Equation.npy', u)