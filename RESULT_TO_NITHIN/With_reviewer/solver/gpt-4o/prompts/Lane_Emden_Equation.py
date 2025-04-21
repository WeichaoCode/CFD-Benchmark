import numpy as np
from scipy.optimize import fsolve

# Parameters
n = 3.0
R0 = 5.0
r_min = 0.0
r_max = 1.0
num_points = 100

# Discretization
r = np.linspace(r_min, r_max, num_points)
dr = r[1] - r[0]

# Initial guess for f
f_initial = R0**(2/(n-1)) * (1 - r**2)**2

# Boundary conditions
f_initial[-1] = 0.0  # f(r=1) = 0

# Function to solve
def lane_emden(f):
    dfdr = np.gradient(f, dr)
    d2fdr2 = np.gradient(dfdr, dr)
    return d2fdr2 + f**n

# Solve the Lane-Emden equation
f_solution = fsolve(lane_emden, f_initial)

# Save the solution
np.save('/PDE_Benchmark/results/prediction/gpt-4o/prompts/f_solution_Lane_Emden_Equation.npy', f_solution)