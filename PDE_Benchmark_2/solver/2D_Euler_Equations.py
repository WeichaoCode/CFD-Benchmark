import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from math import sin,cos,exp,pi

# Define Constants
NX, NY = 256, 256
LX, LY = 2.0, 2.0
DX, DY = LX/(NX-1), LY/(NY-1)
DT = 0.0025
Nt = 100
gama = 1.4

# Utility Functions
def newMatrix():
    return np.zeros((NY, NX))

# Solution Functions
def rho_sol(x, y, t):
    return exp(-t) * sin(pi*x) * sin(pi*y)

def u_sol(x, y, t):
    return exp(-t) * sin(pi*x) * cos(pi*y)

def v_sol(x, y, t):
    return  - exp(-t) * cos(pi*x) * sin(pi*y)

def p_sol(x, y, t):
    return exp(-t) * (1 + sin(pi*x)*sin(pi*y))

# Source Functions
def src_rho(rho, u, v, t):
    return -rho - pi*exp(-t)*(u*cos(pi*x) + v*sin(pi*y))

def src_mom_u(rho, u, v, p, t):
    return -u - (gama-1)*pi*exp(-t)*sin(pi*y)*(v-pi*u) - 2*pi*exp(-t)*p*sin(pi*x)

def src_mom_v(rho, u, v, p, t):
    return -v - (gama-1)*pi*exp(-t)*sin(pi*x)*(u+pi*v) - 2*pi*exp(-t)*p*sin(pi*y)

def src_E(rho, u, v, p, t):
    return -gama*exp(-t)*p +2*(gama-1)*pi**2*exp(-t)*p + \
            (gama-1)*pi*exp(-t)*(u*cos(pi*x)+v*sin(pi*y))*((u**2+v**2)*0.5 + p/rho)

# Primary Functions
def euler_forward(rho, u, v, p, t):
    rho_new = rho + DT * src_rho(rho, u, v, t)
    u_new = u + DT * src_mom_u(rho, u, v, p, t)
    v_new = v + DT * src_mom_v(rho, u, v, p, t)
    p_new = p + DT * src_E(rho, u, v, p, t)
    
    return rho_new, u_new, v_new, p_new

# Main Solver
def solve():
    # Initialize Arrays
    rho = newMatrix()
    u = newMatrix()
    v = newMatrix()
    p = newMatrix()
    
    # Initialize Time
    t = 0
    
    # Iterable Grid Points
    X = np.linspace(0, LX, NX)
    Y = np.linspace(0, LY, NY)
    
    # Main Loop
    for _ in range(Nt):
        for i in range(NX):
            for j in range(NY):
                # Position Update
                x, y = X[i], Y[j]
                
                # Solution Update
                rho[j,i] = rho_sol(x, y, t)
                u[j,i] = u_sol(x, y, t)
                v[j,i] = v_sol(x, y, t)
                p[j,i] = p_sol(x, y, t)
        
        # Time Update
        t += DT
        
        # Update rho, u, v, p
        rho[:,:], u[:,:], v[:,:], p[:,:] = euler_forward(rho, u, v, p, t)
        
    plt.imshow(rho, cmap='hot', interpolation='nearest')
    plt.show()

# Execute Solver
solve()