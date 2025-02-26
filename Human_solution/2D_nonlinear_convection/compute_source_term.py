import sympy as sp

# Define symbols
x, y, t, pi = sp.symbols('x y t pi')
cx, cy = sp.symbols('cx cy')  # Convection speeds

# Define MMS solutions
u = sp.exp(-t) * sp.sin(pi * x) * sp.cos(pi * y)
v = - sp.exp(-t) * sp.cos(pi * x) * sp.sin(pi * y)

# Compute derivatives for f_u
du_dt = sp.diff(u, t)
du_dx = sp.diff(u, x)
du_dy = sp.diff(u, y)

# Compute derivatives for f_v
dv_dt = sp.diff(v, t)
dv_dx = sp.diff(v, x)
dv_dy = sp.diff(v, y)

# Compute source terms
f_u = du_dt + u * du_dx + v * du_dy
f_v = dv_dt + u * dv_dx + v * dv_dy

# Simplify expressions
f_u_simplified = sp.simplify(f_u)  # (pi*cos(pi*x) - exp(t)*cos(pi*y))*exp(-2*t)*sin(pi*x)
f_v_simplified = sp.simplify(f_v)  # (pi*cos(pi*y) + exp(t)*cos(pi*x))*exp(-2*t)*sin(pi*y)

print(f_u_simplified)
print("\n")
print(f_v_simplified)
