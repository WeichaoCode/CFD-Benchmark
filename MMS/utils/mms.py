import sympy as sp


class ManufacturedSolutionPDE:
    def __init__(self, equation, manufactured_solution, variables, domain):
        """
        Initialize the PDE solver with a given PDE and manufactured solution.

        Parameters:
        - equation: The PDE to solve (must be written in terms of sympy symbols)
        - manufactured_solution: The exact solution chosen by the user
        - variables: A tuple of symbolic variables (e.g., (x, t) for 1D time-dependent PDEs)
        - domain: Dictionary defining the domain, boundary, and initial conditions (e.g., {'x': (0,1), 't': (0,2)})
        """
        self.equation = equation
        self.solution = manufactured_solution
        self.variables = variables
        self.domain = domain
        self.source_term = None
        self.boundary_conditions = {}
        self.initial_conditions = None

    def compute_source_term(self):
        """ Compute the source term f(x, t) required for the PDE to satisfy the manufactured solution. """
        # Compute the residual when manufactured solution is substituted into the PDE
        residual = self.equation.subs(self.variables[-1], self.solution)  # Replace u in PDE with exact solution
        self.source_term = -sp.simplify(residual)  # Rearrange to compute source term
        return self.source_term

    def extract_boundary_conditions(self):
        """ Derive the boundary conditions from the manufactured solution. """
        for var in self.variables[:-1]:  # Ignore time variable
            if var in self.domain:
                x_min, x_max = self.domain[var]
                self.boundary_conditions[var] = {
                    f"{var}={x_min}": self.solution.subs(var, x_min),
                    f"{var}={x_max}": self.solution.subs(var, x_max)
                }
        return self.boundary_conditions

    def extract_initial_conditions(self):
        """ Derive the initial conditions for time-dependent PDEs. """
        if "t" in self.domain:
            t0 = self.domain["t"][0]  # Initial time
            self.initial_conditions = self.solution.subs("t", t0)
        return self.initial_conditions

    def generate_pde_conditions(self):
        """ Compute source term, boundary, and initial conditions for the manufactured solution. """
        self.compute_source_term()
        self.extract_boundary_conditions()
        self.extract_initial_conditions()
        return {
            "PDE": self.equation,
            "Source Term": self.source_term,
            "Boundary Conditions": self.boundary_conditions,
            "Initial Condition": self.initial_conditions
        }


# Example Usage
if __name__ == "__main__":
    # Define variables
    x, y, t = sp.symbols('x y t')
    u = sp.Function('u')(x, y, t)
    v = sp.Function('u')(x, y, t)

    # Define a PDE (Heat Equation)
    nu = sp.Symbol('nu')
    c = sp.Symbol('c')
    # pde = sp.diff(u, t) + u * sp.diff(u, x)
    # pde = sp.diff(u, t) - nu * sp.diff(u, x, x)
    # pde = sp.diff(u, t) + u * sp.diff(u, x) - nu * sp.diff(u, x, x)
    # pde = sp.diff(u, t) + c * sp.diff(u, x) + c * sp.diff(u, y)
    pde = sp.diff(u, t) + u * sp.diff(u, x)

    # Choose a manufactured solution
    manufactured_solution = sp.exp(-t) * sp.sin(sp.pi * x * y)

    # Define domain
    domain = {"x": (0, 2), "y": (0, 2), "t": (0, 2)}

    # Initialize the Manufactured Solution PDE framework
    pde_solver = ManufacturedSolutionPDE(pde, manufactured_solution, (x, t, u), domain)

    # Compute and print results
    results = pde_solver.generate_pde_conditions()
    for key, value in results.items():
        print(f"\n{key}:")
        sp.pprint(value)
