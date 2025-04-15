"""
2D diffusion equation
---------------------------------------------------------------
- Exercise 10 - Solving a 2D diffusion equation via the Finite Volume Method
- Finite Volume Method
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def plot2d(x, y, u, figtitle):
    # This function yields a pretty 3D plot of a 2D field. It takes three
    # arguments: the x array of dimension (1,nx), y arrary of dim (1,ny), and
    # solution array of dimension (ny,nx)

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, u, cmap=cm.viridis, rstride=2, cstride=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(figtitle)


def FVM(h, N, mu, dPdz, MAX, TOL):
    dx = h / N
    dy = h / N
    L2norm = np.ones(1)

    ## Define initial conditions information

    w = np.zeros((N, N))
    wp1 = w.copy()
    aE = mu * dy / dx
    aW = mu * dy / dx
    aN = mu * dx / dy
    aS = mu * dx / dy
    Su = dPdz * dx * dy

    l = 0
    while (L2norm[l] > TOL):
        if (l > MAX):
            raise Exception('System not converged after MAX iterations')

        # north boundary
        aB = 2 * mu * dx / dy
        aP = aE + aW + aN + aB
        wp1[0, 1:-1] = (aE * w[0, 2:] + aW * w[0, :-2] + aN * w[1, 1:-1] - Su) / aP
        # south boundary
        aP = aE + aW + aS + aB
        wp1[-1, 1:-1] = (aE * w[-1, 2:] + aW * w[-1, :-2] + aS * w[-1, 1:-1] - Su) / aP
        # west boundary
        aB = 2 * mu * dy / dx
        aP = aE + aN + aS + aB
        wp1[1:-1, 0] = (aE * w[1:-1, 1] + aN * w[2:, 0] + aS * w[:-2, 0] - Su) / aP
        # east boundary
        aP = aW + aN + aS + aB
        wp1[1:-1, -1] = (aW * w[1:-1, -1] + aN * w[2:, -1] + aS * w[:-2, -1] - Su) / aP

        # internal cells
        aP = aE + aW + aN + aS
        wp1[1:-1, 1:-1] = (aW * w[1:-1, :-2] + aE * w[1:-1, 2:] + aN * w[2:, 1:-1] + aS * w[:-2, 1:-1] - Su) / aP

        # fill in corner cells as the mean of adjacent cells
        wp1[0, 0] = 0.5 * (wp1[1, 0] + wp1[0, 1])
        wp1[-1, 0] = 0.5 * (wp1[-2, 0] + wp1[-1, 1])
        wp1[0, -1] = 0.5 * (wp1[0, -2] + wp1[1, -1])
        wp1[-1, -1] = 0.5 * (wp1[-1, -2] + wp1[-2, -1])

        # compute residual L2 norm
        L2norm = np.append(L2norm, (np.sum((w - wp1) ** 2)) ** 0.5)

        w = wp1.copy()
        l += 1
    return w, L2norm


# Domain length
h = 0.1
mu = 1E-3
dPdz = -3.2
MAX = 50000
TOL = 1e-3
N = np.linspace(10, 80, 10)
N = N.astype(int)
wmax = np.zeros(10)

for l in range(0, 10):
    w, L2norm = FVM(h, N[l], mu, dPdz, MAX, TOL)
    wmax[l] = w[int(N[l] / 2) - 1, int(N[l] / 2) - 1]
    print("N = ", N[l], "wmax = ", wmax[l])

plot2d(np.linspace(0, h, N[-1]), np.linspace(0, h, N[-1]), w, '2D diffusion equation')
plt.show()

##########################################################################
import os

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "results")

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# Get the current Python file name without extension
python_filename = os.path.splitext(os.path.basename(__file__))[0]

# Define the file name dynamically
output_file_w = os.path.join(OUTPUT_FOLDER, f"w_{python_filename}_SA.npy")

# Save the array u in the results folder
np.save(output_file_w, w)
