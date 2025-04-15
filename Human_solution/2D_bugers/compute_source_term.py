import sympy as sp

# Define symbols
x, y, t, pi, nu = sp.symbols('x y t pi nu')

# Define the manufactured solutions (MMS)
u = sp.exp(-t) * sp.sin(pi * x) * sp.sin(pi * y)
v = sp.exp(-t) * sp.cos(pi * x) * sp.cos(pi * y)

# Compute derivatives for u
du_dt = sp.diff(u, t)
du_dx = sp.diff(u, x)
du_dy = sp.diff(u, y)
du_dxx = sp.diff(du_dx, x)
du_dyy = sp.diff(du_dy, y)

# Compute derivatives for v
dv_dt = sp.diff(v, t)
dv_dx = sp.diff(v, x)
dv_dy = sp.diff(v, y)
dv_dxx = sp.diff(dv_dx, x)
dv_dyy = sp.diff(dv_dy, y)

# Compute source terms f_u and f_v for 2D Burgers' equation
f_u = du_dt + u * du_dx + v * du_dy - nu * (du_dxx + du_dyy)
f_v = dv_dt + u * dv_dx + v * dv_dy - nu * (dv_dxx + dv_dyy)

# Simplify expressions
f_u_simplified = sp.simplify(f_u)  # (pi*cos(pi*x) + (2*nu*pi**2 - 1)*exp(t)*sin(pi*y))*exp(-2*t)*sin(pi*x)
f_v_simplified = sp.simplify(f_v)  # (-pi*sin(pi*y) + (2*nu*pi**2 - 1)*exp(t)*cos(pi*x))*exp(-2*t)*cos(pi*y)

# Display the computed source terms
print(f_u_simplified)
print('\n')
print(f_v_simplified)
