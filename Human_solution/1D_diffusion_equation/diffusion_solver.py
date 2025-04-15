import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


class DiffusionSolverCN:
    def __init__(self, nx, nt, dx, dt, nu):
        self.nx = nx  # Number of spatial points
        self.nt = nt  # Number of time steps
        self.dx = dx  # Spatial step size
        self.dt = dt  # Time step size
        self.nu = nu  # Diffusivity
        self.u = np.zeros((nx, nt))  # Solution array
        self.x = np.linspace(0, (nx - 1) * dx, nx)  # Spatial grid
        self.t = np.linspace(0, (nt - 1) * dt, nt)  # Time grid

    def source_term(self, i, n):
        """ Given source term from MMS method """
        x, t = self.x[i], self.t[n]
        return (np.pi ** 2 * self.nu * np.exp(-t) * np.sin(np.pi * x) -
                np.exp(-t) * np.sin(np.pi * x))

    def initialize(self, initial_condition):
        """ Set the initial condition. """
        self.u[:, 0] = initial_condition

    def solve(self):
        """ Solve using Crank-Nicholson method with source term. """
        r = self.nu * self.dt / (2 * self.dx ** 2)  # Diffusion coefficient

        # Tridiagonal coefficient matrix A (Implicit part)
        A = np.zeros((self.nx, self.nx))
        for i in range(1, self.nx - 1):
            A[i, i - 1] = -r
            A[i, i] = 1 + 2 * r
            A[i, i + 1] = -r
        A[0, 0] = A[-1, -1] = 1  # Dirichlet BCs
        A_inv = np.linalg.inv(A)  # Precompute inverse (for small nx)

        # Time-stepping loop
        for n in range(self.nt - 1):
            b = np.copy(self.u[:, n])  # Right-hand side vector

            for i in range(1, self.nx - 1):
                # Diffusion term (explicit part)
                diffusion = r * (self.u[i - 1, n] - 2 * self.u[i, n] + self.u[i + 1, n])

                # Source term (MMS)
                source = self.dt / 2 * (self.source_term(i, n) + self.source_term(i, n + 1))

                # Right-hand side
                b[i] += diffusion + source

            # Solve Au^{n+1} = b
            self.u[:, n + 1] = np.dot(A_inv, b)

    def get_solution(self):
        """ Return computed solution. """
        return self.u


def analytical_solution(x, t):
    """ Analytical solution of 1D diffusion equation (Manufactured Solution). """
    return np.exp(-t) * np.sin(np.pi * x)


# Parameters
nx = 100  # Number of spatial points
nt = 200  # Number of time steps
dx = 0.02  # Spatial step size
dt = 0.001  # Time step size
nu = 0.1  # Diffusivity

# Initialize solver
solver = DiffusionSolverCN(nx, nt, dx, dt, nu)

# Define spatial and temporal grid
x = np.linspace(0, (nx - 1) * dx, nx)
t = np.linspace(0, (nt - 1) * dt, nt)

# Initial condition
initial_condition = analytical_solution(x, 0)
solver.initialize(initial_condition)

# Solve numerically
solver.solve()
numerical_solution = solver.get_solution()

# Compute the analytical solution at all time steps
analytical_sol = np.zeros((nx, nt))
for n in range(nt):
    analytical_sol[:, n] = analytical_solution(x, t[n])

# Compute absolute error
error = np.abs(numerical_solution - analytical_sol)

# ===================== PLOTTING =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Heatmap of the numerical solution
c1 = axes[0].imshow(numerical_solution, extent=[0, nt * dt, 0, nx * dx], origin="lower", aspect="auto", cmap="jet")
fig.colorbar(c1, ax=axes[0])
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Space")
axes[0].set_title("Numerical Solution (Crank-Nicholson)")

# Heatmap of the analytical solution
c2 = axes[1].imshow(analytical_sol, extent=[0, nt * dt, 0, nx * dx], origin="lower", aspect="auto", cmap="jet")
fig.colorbar(c2, ax=axes[1])
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Space")
axes[1].set_title("Analytical Solution")

# Heatmap of absolute error
c3 = axes[2].imshow(error, extent=[0, nt * dt, 0, nx * dx], origin="lower", aspect="auto", cmap="inferno")
fig.colorbar(c3, ax=axes[2])
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Space")
axes[2].set_title("Absolute Error (|Numerical - Analytical|)")

plt.tight_layout()
plt.savefig("/opt/CFD-Benchmark/Human_solution/1D_diffusion_equation/image.png")
plt.show()

# ===================== LINE PLOTS =====================
plt.figure(figsize=(10, 6))
time_steps_to_plot = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]  # Select different time steps

for n in time_steps_to_plot:
    plt.plot(x, numerical_solution[:, n], label=f"Numerical (t={t[n]:.2f})", linestyle="--")
    plt.plot(x, analytical_sol[:, n], label=f"Analytical (t={t[n]:.2f})", linestyle="-")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Comparison of Numerical vs Analytical Solution")
plt.legend()
plt.grid()
plt.show()
