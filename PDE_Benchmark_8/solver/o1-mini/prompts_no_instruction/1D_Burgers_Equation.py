import numpy as np

nu = 0.07
nx = 101
nt = 100
dx = 2 * np.pi / (nx - 1)
dt = dx * nu
x = np.linspace(0, 2 * np.pi, nx)

phi = np.exp(-x**2 / (4 * nu)) + np.exp(-(x - 2 * np.pi)**2 / (4 * nu))
phi_x = (np.roll(phi, -1) - np.roll(phi, 1)) / (2 * dx)
u = -2 * nu / phi * phi_x + 4

for _ in range(nt):
    du_dx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    d2u_dx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    u = u + dt * (-u * du_dx + nu * d2u_dx2)

np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/u_1D_Burgers_Equation.npy', u)