# import sympy as sp
#
#
# class ManufacturedSolutionPDE:
#     def __init__(self, equation, manufactured_solution, variables, domain):
#         """
#         Initialize the PDE solver with a given PDE and manufactured solution.
#
#         Parameters:
#         - equation: The PDE to solve (must be written in terms of sympy symbols)
#         - manufactured_solution: The exact solution chosen by the user
#         - variables: A tuple of symbolic variables (e.g., (x, t) for 1D time-dependent PDEs)
#         - domain: Dictionary defining the domain, boundary, and initial conditions (e.g., {'x': (0,1), 't': (0,2)})
#         """
#         self.equation = equation
#         self.solution = manufactured_solution
#         self.variables = variables
#         self.domain = domain
#         self.source_term = None
#         self.boundary_conditions = {}
#         self.initial_conditions = None
#
#     def compute_source_term(self):
#         """ Compute the source term f(x, t) required for the PDE to satisfy the manufactured solution. """
#         # Compute the residual when manufactured solution is substituted into the PDE
#         residual = self.equation.subs(self.variables[-1], self.solution)  # Replace u in PDE with exact solution
#         self.source_term = -sp.simplify(residual)  # Rearrange to compute source term
#         return self.source_term
#
#     def extract_boundary_conditions(self):
#         """ Derive the boundary conditions from the manufactured solution. """
#         for var in self.variables[:-1]:  # Ignore time variable
#             if var in self.domain:
#                 x_min, x_max = self.domain[var]
#                 self.boundary_conditions[var] = {
#                     f"{var}={x_min}": self.solution.subs(var, x_min),
#                     f"{var}={x_max}": self.solution.subs(var, x_max)
#                 }
#         return self.boundary_conditions
#
#     def extract_initial_conditions(self):
#         """ Derive the initial conditions for time-dependent PDEs. """
#         if "t" in self.domain:
#             t0 = self.domain["t"][0]  # Initial time
#             self.initial_conditions = self.solution.subs("t", t0)
#         return self.initial_conditions
#
#     def generate_pde_conditions(self):
#         """ Compute source term, boundary, and initial conditions for the manufactured solution. """
#         self.compute_source_term()
#         self.extract_boundary_conditions()
#         self.extract_initial_conditions()
#         return {
#             "PDE": self.equation,
#             "Source Term": self.source_term,
#             "Boundary Conditions": self.boundary_conditions,
#             "Initial Condition": self.initial_conditions
#         }
#
#
# # Example Usage
# if __name__ == "__main__":
#     # Define variables
#     x, y, t = sp.symbols('x y t')
#     u = sp.Function('u')(x, y, t)
#     v = sp.Function('u')(x, y, t)
#
#     # Define a PDE (Heat Equation)
#     nu = sp.Symbol('nu')
#     c = sp.Symbol('c')
#     # pde = sp.diff(u, t) + u * sp.diff(u, x)
#     # pde = sp.diff(u, t) - nu * sp.diff(u, x, x)
#     # pde = sp.diff(u, t) + u * sp.diff(u, x) - nu * sp.diff(u, x, x)
#     # pde = sp.diff(u, t) + c * sp.diff(u, x) + c * sp.diff(u, y)
#     pde = sp.diff(u, t) + u * sp.diff(u, x)
#
#     # Choose a manufactured solution
#     manufactured_solution = sp.exp(-t) * sp.sin(sp.pi * x * y)
#
#     # Define domain
#     domain = {"x": (0, 2), "y": (0, 2), "t": (0, 2)}
#
#     # Initialize the Manufactured Solution PDE framework
#     pde_solver = ManufacturedSolutionPDE(pde, manufactured_solution, (x, t, u), domain)
#
#     # Compute and print results
#     results = pde_solver.generate_pde_conditions()
#     for key, value in results.items():
#         print(f"\n{key}:")
#         sp.pprint(value)
import sympy as sp
import json


class MMS:
    """Generalized Manufactured Solution Method for PDEs in any dimension and multiple equations."""

    def __init__(self, equations, manufactured_solutions, variables, domain):
        """
        Initialize the MMS solver.

        Parameters:
        - equations: Dictionary where keys are equation names and values are SymPy PDEs.
        - manufactured_solutions: Dictionary where keys are variable names and values are their MMS solutions.
        - variables: Tuple of symbolic variables (e.g., (x, y, z, t, p)).
        - domain: Dictionary defining spatial and temporal domain limits.
        """
        self.equations = equations  # {"eq1": PDE1, "eq2": PDE2, ...}
        self.solutions = manufactured_solutions  # {"u": u_exact, "v": v_exact, ...}
        self.variables = variables  # (x, y, z, t, p, ...)
        self.domain = domain
        self.source_terms = {}  # Stores computed source terms
        self.boundary_conditions = {}  # Stores boundary conditions
        self.initial_conditions = {}  # Stores initial conditions

    def compute_source_terms(self):
        """Computes the extra source terms required for MMS."""
        for eq_name, eq in self.equations.items():
            # Substitute the manufactured solutions into the equation
            residual = eq.subs(self.solutions)
            self.source_terms[eq_name] = -sp.simplify(residual)
        return self.source_terms

    def extract_boundary_conditions(self):
        """Computes boundary conditions from the manufactured solutions."""
        for var in self.variables[:-1]:  # Exclude time variable
            if var in self.domain:
                var_min, var_max = self.domain[var]
                self.boundary_conditions[var] = {
                    f"{var}={var_min}": {func: self.solutions[func].subs(var, var_min) for func in self.solutions},
                    f"{var}={var_max}": {func: self.solutions[func].subs(var, var_max) for func in self.solutions}
                }
        return self.boundary_conditions

    def extract_initial_conditions(self):
        """Computes the initial conditions for all variables at t=0."""
        if "t" in self.domain:
            t0 = self.domain["t"][0]
            self.initial_conditions = {func: self.solutions[func].subs("t", t0) for func in self.solutions}
        return self.initial_conditions

    def generate_mms_conditions(self):
        """Compute source terms, boundary conditions, and initial conditions."""
        self.compute_source_terms()
        self.extract_boundary_conditions()
        self.extract_initial_conditions()
        return {
            "Modified Equations": {eq: sp.latex(self.equations[eq]) + " = " + sp.latex(self.source_terms[eq]) for eq in
                                   self.equations},
            "Boundary Conditions": {key: {k: {var: sp.latex(val[var]) for var in val} for k, val in val_set.items()} for
                                    key, val_set in self.boundary_conditions.items()},
            "Initial Conditions": {var: sp.latex(val) for var, val in self.initial_conditions.items()}
        }


# Example Usage
if __name__ == "__main__":
    # Define symbolic variables for a 3D problem
    x, y, z, t, p = sp.symbols('x y z t p')

    # Define unknown functions
    u = sp.Function('u')(x, y, z, t)
    v = sp.Function('v')(x, y, z, t)
    w = sp.Function('w')(x, y, z, t)

    # Define a 3D PDE system (Navier-Stokes-like)
    nu = sp.Symbol('nu')  # Viscosity term

    equations = {
        "x-momentum": sp.diff(u, t) + u * sp.diff(u, x) + v * sp.diff(u, y) + w * sp.diff(u, z) + sp.diff(p, x) - nu * (
                    sp.diff(u, x, x) + sp.diff(u, y, y) + sp.diff(u, z, z)),
        "y-momentum": sp.diff(v, t) + u * sp.diff(v, x) + v * sp.diff(v, y) + w * sp.diff(v, z) + sp.diff(p, y) - nu * (
                    sp.diff(v, x, x) + sp.diff(v, y, y) + sp.diff(v, z, z)),
        "z-momentum": sp.diff(w, t) + u * sp.diff(w, x) + v * sp.diff(w, y) + w * sp.diff(w, z) + sp.diff(p, z) - nu * (
                    sp.diff(w, x, x) + sp.diff(w, y, y) + sp.diff(w, z, z))
    }

    # Choose manufactured solutions
    manufactured_solutions = {
        "u": sp.exp(-t) * sp.sin(sp.pi * x) * sp.cos(sp.pi * y) * sp.sin(sp.pi * z),
        "v": sp.exp(-t) * sp.cos(sp.pi * x) * sp.sin(sp.pi * y) * sp.sin(sp.pi * z),
        "w": sp.exp(-t) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * sp.cos(sp.pi * z)
    }

    # Define domain
    domain = {"x": (0, 2), "y": (0, 2), "z": (0, 2), "t": (0, 2)}

    # MMS for n-Dimensional Navier-Stokes
    mms = MMS(equations, manufactured_solutions, (x, y, z, t, p), domain)
    results = mms.generate_mms_conditions()

    # Save results to JSON
    with open("mms_general.json", "w") as file:
        json.dump(results, file, indent=4)

    print("✅ JSON file 'mms_general.json' created successfully!")
