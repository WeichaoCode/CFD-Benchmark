import numpy as np
import matplotlib.pyplot as plt

def solve_1d_wave_equation(c, x_end, t_end, nx, nt):
    # step 1: define PARAMETERS
    dx = x_end / (nx - 1)
    dt = t_end / (nt - 1)
    
    x = np.linspace(0, x_end, nx)
    t = np.linspace(0, t_end, nt)
    
    # step 2: check CFL CONDITION
    cfl = c * dt / dx
    if cfl > 1:
        raise ValueError('CFL condition not met: dt needs to be < dx / c')
    
    # step 3: compute source term from MMS solution
    ff = lambda x,t: np.exp(-t) * ((c**2) * (np.pi**2) * np.sin(np.pi*x) * np.cos(t) - 2*np.sin(np.pi*x) * np.sin(t))

    # step 4: compute initial and boundary conditions from MMS
    u0_x = lambda x: np.sin(np.pi*x)  # At t=0 
    u_x0 = lambda t: 0  # At x=0
    u_xend = lambda t: 0  # At x=x_end
    
    # step 5: solve the PDE using FINITE DIFFERENCE
    u = np.zeros((nt, nx))
    u[0,:] = u0_x(x)
    u[:,0] = u_x0(t)
    u[:,-1] = u_xend(t)
    
    for n in range(0, nt - 1):
        for j in range(1, nx - 1):
            u[n+1, j] = (2*cfl**2 * u[n, j-1] + 2*(1-cfl**2) * u[n, j] + dt**2 * ff(x[j], t[n]) - u[n-1, j])
    
    # step 6: compute exact solution for comparison
    exact_soln = np.zeros((nt, nx))
    for i in range(nt):
        for j in range(nx):
            exact_soln[i, j] = np.exp(-t[i]) * np.sin(np.pi*x[j]) * np.cos(t[i])
            
    # step 7: error analysis and plot numerical, exact solution and error
    error = np.abs(exact_soln - u)
    print(f'Max error: {np.max(error)}')
    
    plt.plot(x, u[-1,:], label='Numerical')
    plt.plot(x, exact_soln[-1,:], label='Exact')
    plt.legend()
    plt.show()
    
    plt.plot(x, error[-1,:])
    plt.title('Error')
    plt.show()
    
    return u, exact_soln, error

# Example usage
u, exact_soln, error = solve_1d_wave_equation(1, 1, 1, 100, 4000)