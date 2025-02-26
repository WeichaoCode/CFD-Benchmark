import math
import numpy as np
import matplotlib.pyplot as plt

"""
Create a class for finite difference method, difficult to ensure stability.
"""


class solve_1D_unsteady_flow:
    """
    A solver for 1D unsteady flow problems using various numerical methods.
    Supports different PDEs, initial conditions, and boundary conditions.
    """

    def __init__(self, nx, nt, T, L, c, nu, equation, method, ic_type="mms", bc_type="mms"):
        """
        Initialize the solver for 1D unsteady flow.

        Args:
            nx (int): Number of spatial points.
            nt (int): Number of time steps.
            T (float): Final simulation time [0, T].
            L (float): Length of spatial domain [0, L].
            c (float): Wave speed.
            nu (float): Viscosity (if applicable).
            equation (str): Type of equation (e.g., "linear convection").
            method (str): Numerical method (e.g., "FTCS", "FOU", "Leapfrog").
            ic_type (str): Type of initial condition. Options: "sin", "gaussian", "step", "constant".
            bc_type (str): Type of boundary condition. Options: "dirichlet", "neumann", "periodic".
        """
        self.x_start, self.x_end = 0.0, L
        self.t_start, self.t_end = 0.0, T
        self.wave_speed = c
        self.viscosity = nu
        self.equation = equation
        self.method = method
        self.ic_type = ic_type.lower()
        self.bc_type = bc_type.lower()

        self.nx, self.nt = nx, nt
        self.dx, self.dt = None, None
        self.x, self.t, self.u = None, None, None

    def solve_cfd_problems(self):
        """Solve the specified CFD problem."""
        if self.equation == "linear convection":
            return self.solve_linear_convection()
        elif self.equation == "nonlinear convection":
            return self.solve_nonlinear_convection()
        elif self.equation == "diffusion equation":
            return self.solve_diffusion_equation()
        elif self.equation == "burgers equation":
            return self.solve_burgers_equation()
        else:
            raise NotImplementedError(f"The equation '{self.equation}' is not implemented yet.")

    def _initialize_grid(self):
        """Initialize grid, time steps, and solution array."""
        self.dx = (self.x_end - self.x_start) / (self.nx - 1)
        self.dt = (self.t_end - self.t_start) / (self.nt - 1)
        self.x = np.linspace(self.x_start, self.x_end, self.nx)
        self.t = np.linspace(self.t_start, self.t_end, self.nt)

        # Initialize the solution array
        self.u = np.zeros((self.nx, self.nt))

        # Apply initial and boundary conditions
        self._set_initial_conditions()
        self._apply_boundary_conditions()

    def _set_initial_conditions(self):
        """
        Sets the initial condition (IC) based on the chosen type.
        Supports:
        - "sin" (default): Sine wave initial condition.
        - "gaussian": Gaussian distribution.
        - "step": Step function (shock tube type).
        - "constant": Uniform initial condition.
        """
        if self.ic_type == "sin":
            self.u[:, 0] = np.sin(np.pi * self.x)
        elif self.ic_type == "gaussian":
            self.u[:, 0] = np.exp(-100 * (self.x - 0.5) ** 2)  # Gaussian bump centered at x=0.5
        elif self.ic_type == "step":
            self.u[:, 0] = np.where(self.x < 0.5, 1.0, 0.0)  # Step function with discontinuity at x=0.5
        elif self.ic_type == "constant":
            self.u[:, 0] = 1.0  # Constant initial value
        elif self.ic_type == "mms":
            self.u[:, 0] = np.sin(np.pi * self.x)  # use manufactured solution
        else:
            raise ValueError(f"Unsupported initial condition type: {self.ic_type}")

    def _apply_boundary_conditions(self):
        """
        Applies boundary conditions (BC) based on the chosen type.
        Supports:
        - "dirichlet" (default): Fixed value at boundaries (e.g., u=0).
        - "neumann": Zero-gradient at boundaries (du/dx = 0).
        - "periodic": Periodic boundary conditions.
        """
        if self.bc_type == "dirichlet":
            self.u[0, :] = 0.0  # Left boundary
            self.u[-1, :] = 0.0  # Right boundary
        elif self.bc_type == "neumann":
            self.u[0, :] = self.u[1, :]  # Zero gradient at left boundary
            self.u[-1, :] = self.u[-2, :]  # Zero gradient at right boundary
        elif self.bc_type == "periodic":
            self.u[0, :] = self.u[-2, :]  # Wrap around left to right
            self.u[-1, :] = self.u[1, :]  # Wrap around right to left
        elif self.bc_type == "mms":
            self.u[0, :], self.u[-1, :] = 0.0, 0.0  # use manufactured solution
        else:
            raise ValueError(f"Unsupported boundary condition type: {self.bc_type}")

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
        elif equation == "diffusion equation":
            source_term = (np.pi ** 2 * self.viscosity * np.exp(-self.t[n]) *
                           np.sin(np.pi * self.x[i]) - np.exp(-self.t[n]) * np.sin(np.pi * self.x[i]))
        elif equation == "burgers equation":
            source_term = np.pi * np.exp(-2 * self.t[n]) * np.cos(np.pi * self.x[i]) * np.sin(np.pi * self.x[i])
            (- np.exp(-self.t[n]) * np.sin(np.pi * self.x[i]) + np.pi ** 2 * self.viscosity *
             np.exp(-self.t[n]) * np.sin(np.pi * self.x[i]))
        return source_term

    def _check_stability(self):
        """
        Checks the numerical stability conditions and adjusts dx, dt if necessary.
        Ensures:
        - CFL condition for convection: c * dt / dx ≤ 1
        - Diffusion stability condition: nu * dt / dx^2 ≤ 0.5
        """
        # Compute CFL condition for convection
        cfl = self.wave_speed * self.dt / self.dx
        diffusion_stability = self.viscosity * self.dt / self.dx ** 2  # For diffusion

        print(f"Initial Stability Check: CFL = {cfl:.4f}, Diffusion Stability = {diffusion_stability:.4f}")

        # Adjust dt if CFL condition is violated
        if self.equation in ["linear convection", "nonlinear convection"]:
            if cfl > 1.0:
                self.dt = 0.9 * self.dx / self.wave_speed  # Reduce dt to ensure CFL ≤ 1
                print(f"CFL condition violated! Adjusting dt to {self.dt:.6f}")

        # Adjust dt if diffusion stability is violated
        if self.equation == "diffusion equation":
            if diffusion_stability > 0.5:
                self.dt = 0.4 * self.dx ** 2 / self.viscosity  # Reduce dt to ensure stability
                print(f"Diffusion condition violated! Adjusting dt to {self.dt:.6f}")

        # Final stability check after adjustment
        cfl = self.wave_speed * self.dt / self.dx
        diffusion_stability = self.viscosity * self.dt / self.dx ** 2
        print(f"Adjusted Stability Check: CFL = {cfl:.4f}, Diffusion Stability = {diffusion_stability:.4f}")

        if cfl > 1.0 or diffusion_stability > 0.5:
            print("Warning: Stability may still not be guaranteed. Consider reducing dx or increasing nx.")

    def _solve_linear_convection(self):
        """This funtion is written by human as an code example for LLM
        """
        self._initialize_grid()
        self._check_stability()
        # The following methods are FDM
        if self.method == "FOU":
            # Time-stepping loop
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = (self.u[i, n] - self.wave_speed * self.dt / self.dx *
                                        (self.u[i, n] - self.u[i - 1, n]) + self.dt * self._source_term(n, i,
                                                                                                        self.equation))
        elif self.method == "Leapfrog":
            # First step using Forward Euler (! this is import or when compute u[:, 1] will use u[:, -1] which is wrong)
            for i in range(1, self.nx - 1):
                self.u[i, 1] = (self.u[i, 0] - 0.5 * self.wave_speed * self.dt / self.dx *
                                (self.u[i + 1, 0] - self.u[i - 1, 0]) + self.dt * self._source_term(0, i,
                                                                                                    self.equation))
            # Time-stepping loop
            for n in range(1, self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = (self.u[i, n - 1] - self.wave_speed * self.dt / self.dx *
                                        (self.u[i + 1, n] - self.u[i - 1, n]) + 2 * self.dt * self._source_term(n, i,
                                                                                                                self.equation))
        elif self.method == "Lax-Friedrichs":
            # Time-stepping loop
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = ((self.u[i - 1, n] + self.u[i + 1, n]) / 2 - 0.5 * self.wave_speed *
                                        self.dt / self.dx * (self.u[i + 1, n] - self.u[i - 1, n]) + self.dt *
                                        self._source_term(n, i, self.equation))
        # The following methods are FVM
        else:
            raise NotImplementedError(f"The numerical method '{self.method} is not implemented yet.")

    def solve_nonlinear_convection(self):
        """
        Solve the 1D nonlinear convection equation using the specified finite difference method.
        The equation is: ∂u/∂t + u ∂u/∂x = S(x,t)
        This code generated by GPT-4o
        """
        self._initialize_grid()
        self._check_stability()
        if self.method == "Leapfrog":
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
            # Time-stepping loop
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = (0.5 * (self.u[i - 1, n] + self.u[i + 1, n]) -
                                        0.5 * self.dt / self.dx * (self.u[i + 1, n] - self.u[i - 1, n])
                                        * self.u[i, n] + self.dt * self._source_term(n, i, self.equation))

        else:
            raise NotImplementedError(f"The numerical method '{self.method}' is not implemented yet.")

    def solve_diffusion_equation(self):
        self._initialize_grid()
        self._check_stability()
        if self.method == "FTCS":
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    self.u[i, n + 1] = (self.u[i, n] + self.viscosity *
                                        self.dt / (self.dx ** 2) * (self.u[i + 1, n] - 2 *
                                                                    self.u[i, n] + self.u[i - 1, n])
                                        + self.dt * self._source_term(n, i, self.equation))
        else:
            raise NotImplementedError(f"The numerical method '{self.method} is not implemented yet.")

    def solve_burgers_equation(self):
        self._initialize_grid()
        self._check_stability()
        if self.method == "Crank-Nicholson":
            """ Solve using Crank-Nicholson method with source term. """
            r = self.viscosity * self.dt / (2 * self.dx ** 2)  # Diffusion coefficient
            alpha = self.dt / (4 * self.dx)  # Convective term coefficient

            # Tridiagonal coefficient matrix A (Implicit part)
            A = np.zeros((self.nx, self.nx))
            for i in range(1, self.nx - 1):
                A[i, i - 1] = -r
                A[i, i] = 1 + 2 * r
                A[i, i + 1] = -r
            A[0, 0] = A[-1, -1] = 1  # Dirichlet BCs
            A_inv = np.linalg.inv(A)  # Precompute inverse (for small nx)

            x = np.linspace(0, (self.nx - 1) * self.dx, self.nx)

            # Time-stepping loop
            for n in range(self.nt - 1):
                b = np.copy(self.u[:, n])  # Right-hand side vector

                for i in range(1, self.nx - 1):
                    # Nonlinear convection term
                    convection = alpha * (self.u[i + 1, n] ** 2 - self.u[i - 1, n] ** 2)

                    # Diffusion term
                    diffusion = r * (self.u[i - 1, n] - 2 * self.u[i, n] + self.u[i + 1, n])

                    # Source term (MMS)
                    source = self.dt / 2 * (
                                self._source_term(i, n, self.equation)
                                + self._source_term(i, n + 1, self.equation))

                    # Right-hand side
                    b[i] += diffusion - convection + source

                self.u[:, n + 1] = np.dot(A_inv, b)  # Solve Au^{n+1} = b
        else:
            raise NotImplementedError(f"The numerical method '{self.method} is not implemented yet.")
