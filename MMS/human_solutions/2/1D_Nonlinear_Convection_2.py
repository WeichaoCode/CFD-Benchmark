import math
import numpy as np
import matplotlib.pyplot as plt


class solve_1D_unsteady_flow:
    def __init__(self, nx, nt, T, L, c, nu, equation, method):
        """

        Initialize the solver for 1D unsteady flow.

        Parameters:
            nx:
            nt:
            T: final time, simulation time is [0, T]
            L: spatial domain, x ranges from [0, L]
        """
        self.x_start = 0.0
        self.x_end = L
        self.t_start = 0.0
        self.t_end = T
        self.wave_speed = c
        self.viscosity = nu
        self.equation = equation
        self.method = method
        self.u = None
        self.nx = nx
        self.nt = nt

    def solve_cfd_problems(self):
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

    def _solve_linear_convection(self):
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
                    u[i, n + 1] = u[i, n] - 0.5 * self.wave_speed * dt / dx * (u[i + 1, n] - u[i - 1, n]) \
                                  + dt * (np.pi * self.wave_speed * np.exp(-t[n]) * np.cos(np.pi * x[i])
                                          - np.exp(-t[n]) * np.sin(np.pi * x[i]))
            self.u = u
        elif self.method == "FOU":
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
                    u[i, n + 1] = u[i, n] - self.wave_speed * dt / dx * (u[i, n] - u[i - 1, n]) + dt * (
                            np.pi * self.wave_speed * np.exp(-t[n]) * np.cos(np.pi * x[i])
                            - np.exp(-t[n]) * np.sin(np.pi * x[i]))
            self.u = u
        elif self.method == "Leapfrog":
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
            # First step using Forward Euler (! this is import or when compute u[:, 1] will use u[:, -1] which is wrong)
            for i in range(1, self.nx - 1):
                u[i, 1] = u[i, 0] - 0.5 * self.wave_speed * dt / dx * (u[i + 1, 0] - u[i - 1, 0]) \
                          + dt * (np.pi * self.wave_speed * np.exp(-t[0]) * np.cos(np.pi * x[i])
                                  - np.exp(-t[0]) * np.sin(np.pi * x[i]))
            # Time-stepping loop
            for n in range(1, self.nt - 1):
                for i in range(1, self.nx - 1):
                    u[i, n + 1] = u[i, n - 1] - self.wave_speed * dt / dx * (u[i + 1, n] - u[i - 1, n]) + 2 * dt * (
                            np.pi * self.wave_speed * np.exp(-t[n]) * np.cos(np.pi * x[i])
                            - np.exp(-t[n]) * np.sin(np.pi * x[i]))
            self.u = u
        elif self.method == "Lax-Friedrichs":
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
                    u[i, n + 1] = (u[i - 1, n] + u[i + 1, n]) / 2 - 0.5 * self.wave_speed * dt / dx * (
                            u[i + 1, n] - u[i - 1, n]) + dt * (
                                          np.pi * self.wave_speed * np.exp(-t[n]) * np.cos(np.pi * x[i])
                                          - np.exp(-t[n]) * np.sin(np.pi * x[i]))
            self.u = u
        elif self.method == "Lax-Wendroff":
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
            # note the source term computation is complex, may avoid using this method
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    u[i, n + 1] = u[i, n] - 0.5 * self.wave_speed * dt / dx * (u[i + 1, n] - u[i - 1, n]) \
                                  + 0.5 * (self.wave_speed * dt / dx) ** 2 * (
                                          u[i + 1, n] - 2 * u[i, n] + u[i - 1, n]) + dt * (
                                          np.pi * self.wave_speed * np.exp(-t[n]) * np.cos(np.pi * x[i])
                                          - np.exp(-t[n]) * np.sin(np.pi * x[i])) + (dt ** 2) / 2 * (self.wave_speed * (
                            -np.pi * self.wave_speed * np.exp(-t[n]) * np.pi * np.sin(np.pi * x[i]) - np.exp(
                        -t[n]) * np.pi * np.cos(np.pi * x[i])) - np.pi * self.wave_speed * np.exp(-t[n]) * np.cos(
                        np.pi * x[i])
                                                                                                     + np.exp(
                                -t[n]) * np.sin(np.pi * x[i]))
            self.u = u
        elif self.method == "Beam-Warming":
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
            # we use dt * source term for simplicity (avoid computation)
            # we first use forward Euler to compute u[1, :] since j start from 2, has j - 2 term
            # Time-stepping loop
            for n in range(self.nt - 1):
                u[1, n] = np.exp(t[n]) * np.sin(np.pi * x[1])
                for i in range(2, self.nx - 1):  # Beam-Warming needs u[i-2]
                    u[i, n + 1] = u[i, n] - (3 / 2) * self.wave_speed * dt / dx * (u[i, n] - u[i - 1, n]) \
                                  + (1 / 2) * self.wave_speed * dt / dx * (u[i - 1, n] - u[i - 2, n]) \
                                  + (self.wave_speed ** 2 * dt ** 2) / (2 * dx ** 2) * (
                                              u[i, n] - 2 * u[i - 1, n] + u[i - 2, n]) \
                                  + dt * (np.pi * self.wave_speed * np.exp(-t[0]) * np.cos(np.pi * x[i])
                                  - np.exp(-t[0]) * np.sin(np.pi * x[i]))
            self.u = u
        else:
            raise NotImplementedError(f"The numerical method '{self.method} is not implemented yet.")

    def _solve_nonlinear_convection(self):
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
                    u[i, n + 1] = u[i, n] - 0.5 * dt / dx * (u[i + 1, n] - u[i - 1, n]) * u[i, n] \
                                  + dt * (np.pi * np.exp(-2 * t[n]) * np.cos(np.pi * x[i]) * np.sin(np.pi * x[i])
                                          - np.exp(-t[n]) * np.sin(np.pi * x[i]))
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
    plt.title(f"Comparison of Numerical and Exact Solution ({method},nx:{nx},nt:{nt})")
    plt.grid()
    if save_image:
        plt.savefig(f"/opt/CFD-Benchmark/MMS/human_solutions/images/{equation}_{method}.png")
    plt.show()

    # Print errors
    print("L2 Norm Errors at key time steps:", errors)
    print("MSE Errors at key time steps:", mse_errors)


def numerical_solution_test(equation, method, Nx=101, Nt=101, save_image=False):
    nx, nt, T, L, c, nu = Nx, Nt, 2.0, 2.0, 1.0, 0.3
    if equation == "linear convection":
        if method == "FTCS":
            nx, nt = 101, 601
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

numerical_solution_test(equation="linear convection", method="Beam-Warming", Nx=401, Nt=401)
