import sympy as sp

# Define symbols
x, y, t, pi, nu = sp.symbols('x y t pi nu')  # Alpha is the diffusion coefficient

# Define the manufactured solution (MMS)
u = sp.exp(-t) * sp.sin(pi * x) * sp.sin(pi * y)

# Compute derivatives
du_dt = sp.diff(u, t)  # ∂u/∂t
du_dxx = sp.diff(u, x, x)  # ∂²u/∂x²
du_dyy = sp.diff(u, y, y)  # ∂²u/∂y²

# Compute source term f(x, y, t)
f = du_dt - nu * (du_dxx + du_dyy)

# Simplify the expression
f_simplified = sp.simplify(f)  # (2*nu*pi**2 - 1)*exp(-t)*sin(pi*x)*sin(pi*y)

# Print the computed source term
print("Computed source term f(x,y,t):")
print(f_simplified)
