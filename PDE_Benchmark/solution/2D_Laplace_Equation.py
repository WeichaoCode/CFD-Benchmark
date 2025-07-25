"""
2D Laplace Equation
---------------------------------------------------------------
- Step 9: 2D Laplace Equation
- REF: https://github.com/barbagroup/CFDPython
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""
import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D


def plot2D(x, y, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = numpy.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')


def laplace2d(p, y, dx, dy, l1norm_target):
    l1norm = 1
    pn = numpy.empty_like(p)

    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy ** 2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
                          dx ** 2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) /
                         (2 * (dx ** 2 + dy ** 2)))

        p[:, 0] = 0  # p = 0 @ x = 0
        p[:, -1] = y  # p = y @ x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1
        l1norm = (numpy.sum(numpy.abs(p[:]) - numpy.abs(pn[:])) /
                  numpy.sum(numpy.abs(pn[:])))

    return p


##variable declarations
nx = 121
ny = 121
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

##initial conditions
p = numpy.zeros((ny, nx))  # create a XxY vector of 0's

##plotting aids
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 1, ny)

##boundary conditions
p[:, 0] = 0  # p = 0 @ x = 0
p[:, -1] = y  # p = y @ x = 2
p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1

plot2D(x, y, p)

p = laplace2d(p, y, dx, dy, 1e-4)

plot2D(x, y, p)

pyplot.show()

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
output_file = os.path.join(OUTPUT_FOLDER, f"p_{python_filename}.npy")

# Save the array u in the results folder
numpy.save(output_file, p)