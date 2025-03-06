import numpy as np
import matplotlib.pyplot as plt

def solve_1d_diffusion(n, T_end, nu, L=1):
    """
    Solver for 1D diffusion using FDM and Implicit Scheme.
        n - grid size
        T_end - final time
        nu - diffusion coefficient
        L - size of the domain
    """
    # Define parameters
    dx = L / (n - 1)
    dt = 0.001
    steps = int(T_end / dt)
    
    # Ensure stability with CFL condition
    assert dt <= dx**2 / (2*nu), "Unstable: adjust dt or grid size"
    

    # Space and time discretization
    x = np.linspace(0, L, n)
    t = np.linspace(0, T_end, steps+1)

    # Compute the source term
    f = lambda x,t: np.exp(-t) * (nu * np.pi**2 - 1) * np.sin(np.pi*x)

    # Initial and boundary conditions
    u = np.zeros((steps+1, n))
    u[:,0] = u[:,n-1] = 0 #Neumann BCs

    # Solve PDE with finite differences
    I, J = np.eye(n), np.eye(n, k=-1) + np.eye(n, k=1) - 2*np.eye(n)
    J[0,1], J[n-1,n-2] = 2, 2 # Adjust for Neumann BCs

    for i in range(steps):
        b = u[i,:] + dt * f(x, t[i]) # Compute rhs
        b[0] = b[n-1] = 0 # Boundary conditions
        u[i+1,:] = np.linalg.solve(I + nu*dt/dx**2*J, b)

    return x, t, u

# Parameters
n = 50
T_end = 1
nu = 0.05

# Solve
x, t, u_numeric = solve_1d_diffusion(n, T_end, nu)

# Exact solution for comparison
u_exact = np.exp(-t[-1]) * np.sin(np.pi*x)

# Error analysis
error = np.abs(u_exact - u_numeric[-1])
print(f"Max error: {np.max(error)}")

# Plots
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(x, u_numeric[-1], label="Numeric")
plt.plot(x, u_exact, '--', label="Exact")
plt.ylabel("u")
plt.xlabel("x")
plt.title("Solution at final time")
plt.legend()

plt.subplot(122)
plt.plot(x, error)
plt.ylabel("Error")
plt.xlabel("x")
plt.title("Error at final time")

plt.tight_layout()
plt.show()