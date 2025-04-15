import numpy as np
import matplotlib.pyplot as plt


class NonlinearConvectionFOU:
    def __init__(self, nx, nt, dx, dt):
        self.nx = nx  # Number of spatial points
        self.nt = nt  # Number of time steps
        self.dx = dx  # Spatial step size
        self.dt = dt  # Time step size
        self.u = np.zeros((nx, nt))  # Solution array
        self.x = np.linspace(0, (nx - 1) * dx, nx)  # Spatial grid
        self.t = np.linspace(0, (nt - 1) * dt, nt)  # Time grid

    def source_term(self, i, n):
        """ Given source term from MMS method """
        x, t = self.x[i], self.t[n]
        return (np.pi * np.exp(-2 * t) * np.cos(np.pi * x) * np.sin(np.pi * x) -
                np.exp(-t) * np.sin(np.pi * x))

    def initialize(self, initial_condition):
        """ Set the initial condition. """
        self.u[:, 0] = initial_condition

    def solve(self):
        """ Solve using First-Order Upwind (FOU) Scheme. """
        for n in range(self.nt - 1):
            for i in range(1, self.nx):  # Upwind: use i-1 term
                convection = (self.u[i, n] * (self.u[i, n] - self.u[i - 1, n])) / self.dx
                source = self.dt * self.source_term(i, n)
                self.u[i, n + 1] = self.u[i, n] - self.dt * convection + source

    def get_solution(self):
        """ Return computed solution. """
        return self.u


def analytical_solution(x, t):
    """ Analytical solution for 1D nonlinear convection with source. """
    return np.exp(-t) * np.sin(np.pi * x)


# Parameters
nx = 100  # Number of spatial points
nt = 200  # Number of time steps
dx = 0.02  # Spatial step size
dt = 0.001  # Time step size

# Initialize solver
solver = NonlinearConvectionFOU(nx, nt, dx, dt)

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
axes[0].set_title("Numerical Solution (FOU)")

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
