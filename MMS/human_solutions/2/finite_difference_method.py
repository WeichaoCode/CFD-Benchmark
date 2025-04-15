import math
import numpy as np
import matplotlib.pyplot as plt

"""
Create a class for finite difference method, difficult to ensure stability.
"""


class solve_1D_unsteady_flow:
    def __init__(self, nx, nt, T, L, c, nu, equation, method):
        """
        Initialize the solver for 1D unsteady flow.

        Parameters:
            nx: Number of spatial points
            nt: Number of time steps
            T: Final simulation time [0, T]
            L: Length of spatial domain [0, L]
            c: Wave speed
            nu: Viscosity (if applicable)
            equation: Type of equation (e.g., "linear convection")
            method: Numerical method (e.g., "FTCS", "FOU", "Leapfrog", etc.)
        """
        self.x_start, self.x_end = 0.0, L
        self.t_start, self.t_end = 0.0, T
        self.wave_speed = c
        self.viscosity = nu
        self.equation = equation
        self.method = method
        self.nx, self.nt = nx, nt
        self.dx, self.dt = None, None
        self.x, self.t, self.u = None, None, None

    def solve_cfd_problems(self):
        """Solve the specified CFD problem."""
        if self.equation == "linear convection":
            return self._solve_linear_convection()
        elif self.equation == "nonlinear convection":
            return self._solve_nonlinear_convection()
        elif self.equation == "diffusion equation":
            return self._solve_diffusion_equation()
        elif self.equation == "burgers equation":
            return self._solve_burgers_equation()
        else:
            raise NotImplementedError(f"The equation '{self.equation}' is not implemented yet.")

    def _initialize_grid(self):
        """Initialize grid, time steps, and solution array."""
        self.dx = (self.x_end - self.x_start) / (self.nx - 1)
        self.dt = (self.t_end - self.t_start) / (self.nt - 1)
        self.x = np.linspace(self.x_start, self.x_end, self.nx)
        self.t = np.linspace(self.t_start, self.t_end, self.nt)
        # Initialize the solution array with shape (Nx, Nt)
        self.u = np.zeros((self.nx, self.nt))
        # Set the initial condition
        self.u[:, 0] = np.sin(np.pi * self.x)
        # Set the boundary condition
        self.u[0, :], self.u[-1, :] = 0.0, 0.0

    def _source_term(self, n, i, equation):
        source_term = None
        if equation == "linear convection":
            source_term = np.pi * self.wave_speed * np.exp(-self.t[n]) * np.cos(np.pi * self.x[i]) - np.exp(
                -self.t[n]) * np.sin(
                np.pi * self.x[i])
        elif equation == "nonlinear convection":
            source_term = np.pi * np.exp(-2 * self.t[n]) * np.cos(np.pi * self.x[i]) * np.sin(
                np.pi * self.x[i]) - np.exp(
                -self.t[n]) * np.sin(np.pi * self.x[i])
        return source_term

    def _solve_linear_convection(self):
        """This funtion is written by human as an code example for LLM
        """
        # The following methods are FDM
        if self.method == "FTCS":
            self._initialize_grid()
            # Time-stepping loop
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = (self.u[i, n] - 0.5 * self.wave_speed * self.dt / self.dx *
                                        (self.u[i + 1, n] - self.u[i - 1, n]) + self.dt * self._source_term(n, i,
                                                                                                            self.equation))
        elif self.method == "FOU":
            self._initialize_grid()
            # Time-stepping loop
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = (self.u[i, n] - self.wave_speed * self.dt / self.dx *
                                        (self.u[i, n] - self.u[i - 1, n]) + self.dt * self._source_term(n, i,
                                                                                                        self.equation))
        elif self.method == "Leapfrog":
            self._initialize_grid()
            # First step using Forward Euler (! this is import or when compute u[:, 1] will use u[:, -1] which is wrong)
            for i in range(1, self.nx - 1):
                self.u[i, 1] = (self.u[i, 0] - 0.5 * self.wave_speed * self.dt / self.dx *
                                (self.u[i + 1, 0] - self.u[i - 1, 0]) + self.dt * self._source_term(0, i))
            # Time-stepping loop
            for n in range(1, self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = (self.u[i, n - 1] - self.wave_speed * self.dt / self.dx *
                                        (self.u[i + 1, n] - self.u[i - 1, n]) + 2 * self.dt * self._source_term(n, i,
                                                                                                                self.equation))
        elif self.method == "Lax-Friedrichs":
            self._initialize_grid()
            # Time-stepping loop
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = ((self.u[i - 1, n] + self.u[i + 1, n]) / 2 - 0.5 * self.wave_speed *
                                        self.dt / self.dx * (self.u[i + 1, n] - self.u[i - 1, n]) + self.dt *
                                        self._source_term(n, i, self.equation))
        # elif self.method == "Lax-Wendroff":
        #     dx = (self.x_end - self.x_start) / (self.nx - 1)
        #     dt = (self.t_end - self.t_start) / (self.nt - 1)
        #     x = np.linspace(self.x_start, self.x_end, self.nx)
        #     t = np.linspace(self.t_start, self.t_end, self.nt)
        #     # Initialize the solution array with shape (Nx, Nt)
        #     u = np.zeros((self.nx, self.nt))
        #     # Set the initial condition
        #     u[:, 0] = np.sin(np.pi * x)
        #     # Set the boundary condition
        #     u[0, :], u[-1, :] = 0.0, 0.0
        #     # Time-stepping loop
        #     # note the source term computation is complex, may avoid using this method
        #     for n in range(self.nt - 1):
        #         for i in range(1, self.nx - 1):
        #             u[i, n + 1] = u[i, n] - 0.5 * self.wave_speed * dt / dx * (u[i + 1, n] - u[i - 1, n]) \
        #                           + 0.5 * (self.wave_speed * dt / dx) ** 2 * (
        #                                   u[i + 1, n] - 2 * u[i, n] + u[i - 1, n]) + dt * (
        #                                   np.pi * self.wave_speed * np.exp(-t[n]) * np.cos(np.pi * x[i])
        #                                   - np.exp(-t[n]) * np.sin(np.pi * x[i])) + (dt ** 2) / 2 * (self.wave_speed * (
        #                     -np.pi * self.wave_speed * np.exp(-t[n]) * np.pi * np.sin(np.pi * x[i]) - np.exp(
        #                 -t[n]) * np.pi * np.cos(np.pi * x[i])) - np.pi * self.wave_speed * np.exp(-t[n]) * np.cos(
        #                 np.pi * x[i])
        #                                                                                              + np.exp(
        #                         -t[n]) * np.sin(np.pi * x[i]))
        #     self.u = u
        # elif self.method == "Beam-Warming":
        #     dx = (self.x_end - self.x_start) / (self.nx - 1)
        #     dt = (self.t_end - self.t_start) / (self.nt - 1)
        #     x = np.linspace(self.x_start, self.x_end, self.nx)
        #     t = np.linspace(self.t_start, self.t_end, self.nt)
        #     # Initialize the solution array with shape (Nx, Nt)
        #     u = np.zeros((self.nx, self.nt))
        #     # Set the initial condition
        #     u[:, 0] = np.sin(np.pi * x)
        #     # Set the boundary condition
        #     u[0, :], u[-1, :] = 0.0, 0.0
        #     # we use dt * source term for simplicity (avoid computation)
        #     # First time step using Forward Euler (n=0 → n=1)
        #     for i in range(1, self.nx - 1):
        #         u[i, 1] = u[i, 0] - self.wave_speed * dt / dx * (u[i, 0] - u[i - 1, 0]) \
        #                   + dt * (np.pi * self.wave_speed * np.exp(-t[0]) * np.cos(np.pi * x[i])
        #                           - np.exp(-t[0]) * np.sin(np.pi * x[i]))
        #
        #     # Time-stepping loop using Beam-Warming (n ≥ 1)
        #     for n in range(1, self.nt - 1):
        #         for i in range(2, self.nx - 1):  # Beam-Warming needs u[i-2]
        #             source_term = dt * (np.pi * self.wave_speed * np.exp(-t[n]) * np.cos(np.pi * x[i])
        #                                 - np.exp(-t[n]) * np.sin(np.pi * x[i]))
        #
        #             u[i, n + 1] = u[i, n] - (3 / 2) * self.wave_speed * dt / dx * (u[i, n] - u[i - 1, n]) \
        #                           + (1 / 2) * self.wave_speed * dt / dx * (u[i - 1, n] - u[i - 2, n]) \
        #                           + (self.wave_speed ** 2 * dt ** 2) / (2 * dx ** 2) * (
        #                                   u[i, n] - 2 * u[i - 1, n] + u[i - 2, n]) \
        #                           + source_term  # Corrected source term
        #     self.u = u
        # The following methods are FVM
        else:
            raise NotImplementedError(f"The numerical method '{self.method} is not implemented yet.")

    def _solve_nonlinear_convection(self):
        """
        Solve the 1D nonlinear convection equation using the specified finite difference method.
        The equation is: ∂u/∂t + u ∂u/∂x = S(x,t)
        This code generated by GPT-4o
        """
        if self.method == "FTCS":
            self._initialize_grid()
            # Time-stepping loop
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = (self.u[i, n] - 0.5 * self.dt / self.dx * (self.u[i + 1, n] - self.u[i - 1, n])
                                        * self.u[i, n] + self.dt * self._source_term(n, i, self.equation))

        elif self.method == "FOU":
            self._initialize_grid()
            # Time-stepping loop (First-Order Upwind)
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = (self.u[i, n] - self.dt / self.dx * (self.u[i, n] - self.u[i - 1, n])
                                        * self.u[i, n] + self.dt * self._source_term(n, i, self.equation))

        elif self.method == "Leapfrog":
            self._initialize_grid()
            # First step using Forward Euler
            for i in range(1, self.nx - 1):
                self.u[i, 1] = (self.u[i, 0] - 0.5 * self.dt / self.dx * (self.u[i + 1, 0] - self.u[i - 1, 0])
                                * self.u[i, 0] + self.dt * self._source_term(0, i, self.equation))

            # Time-stepping loop
            for n in range(1, self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = (self.u[i, n - 1] - self.dt / self.dx * (self.u[i + 1, n] - self.u[i - 1, n])
                                        * self.u[i, n] + 2 * self.dt * self._source_term(n, i, self.equation))

        elif self.method == "Lax-Friedrichs":
            self._initialize_grid()
            # Time-stepping loop
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = (0.5 * (self.u[i - 1, n] + self.u[i + 1, n]) -
                                        0.5 * self.dt / self.dx * (self.u[i + 1, n] - self.u[i - 1, n])
                                        * self.u[i, n] + self.dt * self._source_term(n, i, self.equation))

        else:
            raise NotImplementedError(f"The numerical method '{self.method}' is not implemented yet.")

    def _solve_diffusion_equation(self):
        if self.method == "FTCS":
            dx = (self.x_end - self.x_start) / (self.nx - 1)
            dt = (self.t_end - self.t_start) / (self.nt - 1)
            x = np.linspace(self.x_start, self.x_end, self.nx)
            t = np.linspace(self.t_start, self.t_end, self.nt)
            # Initialize the solution array with shape (Nx, Nt)
            u = np.zeros((self.nx, self.nt))
            # Set the initial condition
            u[:, 0] = np.sin(np.pi * x)
            # Set the boundary condition
            u[0, :], u[-1, :] = 0.0, 0.0
            # Time-stepping loop
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    u[i, n + 1] = u[i, n] + self.viscosity * dt / (dx ** 2) * (u[i - 1, n] - 2 * u[i, n] + u[i + 1, n]) \
                                  + dt * (np.pi ** 2 * self.viscosity * np.exp(-t[n]) * np.sin(np.pi * x[i]) -
                                          np.exp(-t[n]) * np.sin(np.pi * x[i]))
            self.u = u
        elif self.method == "FOU":
            pass
        elif self.method == "Leapfrog":
            pass
        elif self.method == "Lax-Friedrichs":
            pass
        elif self.method == "Lax-Wendroff":
            pass
        elif self.method == "Beam-Warming":
            pass
        else:
            raise NotImplementedError(f"The numerical method '{self.method} is not implemented yet.")

    def _solve_burgers_equation(self):
        if self.method == "FTCS":
            dx = (self.x_end - self.x_start) / (self.nx - 1)
            dt = (self.t_end - self.t_start) / (self.nt - 1)
            x = np.linspace(self.x_start, self.x_end, self.nx)
            t = np.linspace(self.t_start, self.t_end, self.nt)
            # Initialize the solution array with shape (Nx, Nt)
            u = np.zeros((self.nx, self.nt))
            # Set the initial condition
            u[:, 0] = np.sin(np.pi * x)
            # Set the boundary condition
            u[0, :], u[-1, :] = 0.0, 0.0
            # Time-stepping loop
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    u[i, n + 1] = ((u[i, n] - 0.5 * dt / dx * u[i, n] * (u[i + 1, n] - u[i - 1, n])
                                    + self.viscosity * dt / (dx ** 2) * (u[i + 1, n] - 2 * u[i, n] + u[i - 1, n])) +
                                   dt * (np.pi * np.exp(-2 * t[n]) * np.cos(np.pi * x[i]) * np.sin(np.pi * x[i]) -
                                         np.exp(-t[n]) * np.sin(np.pi * x[i]) +
                                         np.pi ** 2 * self.viscosity * np.exp(-t[n]) * np.sin(np.pi * x[i])))
            self.u = u
        elif self.method == "FOU":
            pass
        elif self.method == "Leapfrog":
            pass
        elif self.method == "Lax-Friedrichs":
            pass
        elif self.method == "Lax-Wendroff":
            pass
        elif self.method == "Beam-Warming":
            pass
        else:
            raise NotImplementedError(f"The numerical method '{self.method} is not implemented yet.")


def compare_with_exact(u, x, t, equation, method, nx, nt, save_image=False):
    """
    Compare numerical solution u with exact solution u_exact = e^(-t) * sin(pi * x).

    Parameters:
        u (ndarray): Numerical solution array of shape (nt, nx).
        x (ndarray): Spatial grid points of shape (nx,).
        t (ndarray): Temporal grid points of shape (nt,).
        T (float): Final simulation time.

    Returns:
        None (Displays plots and prints L2 norm and MSE error)
    """
    nx, nt = u.shape  # Shape of numerical solution
    dx = x[1] - x[0]  # Spatial step size

    # Define exact solution function
    def u_exact(t, x):
        return np.exp(-t) * np.sin(np.pi * x)

    # Compute the exact solution at key time steps
    t_indices = [0, nt // 4, nt // 2, nt - 1]  # Indices for t = 0, T/4, T/2, T
    t_labels = ["t = 0", f"t = T/4", "t = T/2", "t = T"]

    # Compute errors
    errors = []
    mse_errors = []

    plt.figure(figsize=(8, 6))
    for i, idx in enumerate(t_indices):
        t_val = t[idx]
        u_num = u[:, idx]  # Numerical solution at current time
        u_ex = u_exact(t_val, x)  # Exact solution

        # Compute L2 error and MSE
        l2_error = np.sqrt(np.sum((u_num - u_ex) ** 2) * dx)
        mse_error = np.mean((u_num - u_ex) ** 2)

        errors.append(l2_error)
        mse_errors.append(mse_error)

        # Plot numerical and exact solutions
        plt.plot(x, u_num, linestyle='-', marker='o', label=f"Numerical {t_labels[i]}")
        plt.plot(x, u_ex, linestyle='--', label=f"Exact {t_labels[i]}")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.title(f"{equation}: ({method},nx:{nx},nt:{nt})")
    plt.grid()
    if save_image:
        plt.savefig(f"/opt/CFD-Benchmark/MMS/human_solutions/images/{equation}_{method}.png")
    plt.show()

    # Print errors
    print("L2 Norm Errors at key time steps:", errors)
    print("MSE Errors at key time steps:", mse_errors)


def numerical_solution_test(equation, method, Nx=101, Nt=101, save_image=False):
    nx, nt, T, L, c, nu = Nx, Nt, 2.0, 2.0, 1.0, 0.3
    dx, dt = 2.0 / (nx - 1), 2.0 / (nt - 1)  # CFL conditions: c*dt/dx <= 1
    # solver_1 = solve_1D_unsteady_flow(nx, nt, L=L, T=T, c=c, nu=None, equation="linear convection", method="FTCS")
    solver_1 = solve_1D_unsteady_flow(nx, nt, L=L, T=T, c=c, nu=nu, equation=equation, method=method)
    solver_1.solve_cfd_problems()
    u = solver_1.u
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    compare_with_exact(u, x, t, equation, method, nx, nt, save_image=save_image)


# test_numerical_solution("linear convection", "FTCS")

# test_numerical_solution("nonlinear convection", "FTCS", Nx=101, Nt=601, save_image=True)

# test_numerical_solution("diffusion equation", "FTCS", Nx=51, Nt=1001, save_image=True)

# test_numerical_solution("burgers equation", "FTCS", Nx=51, Nt=1001, save_image=True)
for method in ["FTCS", "FOU", "Leapfrog", "Lax-Friedrichs"]:
    numerical_solution_test(equation="nonlinear convection", method=method)
"""
Create a class for finite volume method
"""

