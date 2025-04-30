import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

nx = 41
ny = 41
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
nt = 1000
dt = 0.3777 / nt
nu = 0.05

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

u = np.ones((ny, nx))
u[int(ny / 2 - 0.25):int(ny / 2 + 0.25), int(nx / 2 - 0.25):int(nx / 2 + 0.25)] = 2

for n in range(nt):
    u_old = u.copy()
    A = diags([ -nu * (1 / (dx**2) + 1 / (dy**2)), 
                nu / (dx**2), 
                nu / (dy**2), 
                -nu * (1 / (dx**2) + 1 / (dy**2))], 
               [-1, 0, 1, 0])
    b = u_old * (1 / dt)
    u = spsolve(A.toarray(), b.flatten())
    u = u.reshape((ny, nx))

np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemma/prompts/u_2D_Diffusion.npy', u)