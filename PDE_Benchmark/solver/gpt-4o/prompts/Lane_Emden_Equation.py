import numpy as np
from scipy.optimize import fsolve

# Parameters
n = 3.0
R0 = 5.0
r_min = 0.0
r_max = 1.0
num_points = 100

# Discretize the domain
r = np.linspace(r_min, r_max, num_points)
dr = r[1] - r[0]

# Initial guess for f(r)
f_initial = R0**(2/(n-1)) * (1 - r**2)**2

# Define the Lane-Emden equation in finite difference form
def lane_emden(f):
    dfdr = np.zeros_like(f)
    d2fdr2 = np.zeros_like(f)
    
    # Central difference for the second derivative
    for i in range(1, len(r) - 1):
        dfdr[i] = (f[i+1] - f[i-1]) / (2 * dr)
        d2fdr2[i] = (f[i+1] - 2*f[i] + f[i-1]) / (dr**2)
    
    # Regularity condition at r=0
    d2fdr2[0] = 2 * (f[1] - f[0]) / (dr**2)
    
    # Boundary condition at r=1
    f[-1] = 0
    
    # Lane-Emden equation
    return d2fdr2 + f**n

# Solve the Lane-Emden equation using fsolve
f_solution = fsolve(lane_emden, f_initial)

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/f_solution_Lane_Emden_Equation.npy', f_solution)