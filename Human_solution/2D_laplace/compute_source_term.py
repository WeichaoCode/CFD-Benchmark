import sympy as sp

# Define symbols
x, y, pi = sp.symbols('x y pi')

# Define the manufactured solution (MMS) for the 2D Laplace equation
u = sp.sin(pi * x) * sp.sin(pi * y)

# Compute second derivatives
u_xx = sp.diff(sp.diff(u, x), x)
u_yy = sp.diff(sp.diff(u, y), y)

# Compute source term f(x, y)
f = u_xx + u_yy

# Simplify expression
f_simplified = sp.simplify(f)  # -2*pi**2*sin(pi*x)*sin(pi*y)

# Display the computed source term
print(f_simplified)
