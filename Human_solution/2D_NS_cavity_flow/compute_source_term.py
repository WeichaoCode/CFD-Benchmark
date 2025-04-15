import sympy as sp

# Define symbols
x, y, t, pi, nu, rho = sp.symbols('x y t pi nu rho')

# Define the manufactured solutions (MMS)
u = sp.exp(-t) * sp.sin(pi * x) * sp.cos(pi * y)
v = -sp.exp(-t) * sp.cos(pi * x) * sp.sin(pi * y)
p = sp.cos(pi * x) * sp.cos(pi * y)

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

# Compute pressure derivatives
dp_dx = sp.diff(p, x)
dp_dy = sp.diff(p, y)
dp_dxx = sp.diff(dp_dx, x)
dp_dyy = sp.diff(dp_dy, y)

# Compute convective terms for u and v
conv_u = u * du_dx + v * du_dy
conv_v = u * dv_dx + v * dv_dy

# Compute source terms f_u and f_v for Navier-Stokes
f_u = du_dt + conv_u + (1/rho) * dp_dx - nu * (du_dxx + du_dyy)
f_v = dv_dt + conv_v + (1/rho) * dp_dy - nu * (dv_dxx + dv_dyy)

# Compute source term f_p for the Pressure Poisson Equation
f_p = dp_dxx + dp_dyy + rho * (du_dx**2 + 2 * du_dy * dv_dx + dv_dy**2)

# Simplify expressions
f_u_simplified = sp.simplify(f_u)  # (pi*rho*cos(pi*x) - pi*exp(t)*cos(pi*y) + rho*(2*nu*pi**2 - 1)*exp(t)*cos(pi*y))*exp(-2*t)*sin(pi*x)/rho
f_v_simplified = sp.simplify(f_v)  # (pi*rho*cos(pi*y) - pi*exp(t)*cos(pi*x) + rho*(-2*nu*pi**2 + 1)*exp(t)*cos(pi*x))*exp(-2*t)*sin(pi*y)/rho
f_p_simplified = sp.simplify(f_p)  # pi**2*(rho*(cos(2*pi*x) + cos(2*pi*y)) - 2*exp(2*t)*cos(pi*x)*cos(pi*y))*exp(-2*t)

# Display the computed source terms
print("f_u:", f_u_simplified)
print("\n")
print("f_v:", f_v_simplified)
print("\n")
print("f_p:", f_p_simplified)

