#!/usr/bin/env python3
import numpy as np

def main():
    # Parameters
    alpha = 0.01  # thermal diffusivity
    Q0 = 200.0    # source term amplitude (°C/s)
    sigma = 0.1

    # Domain parameters
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    Nx = 101  # number of grid points in x
    Ny = 101  # number of grid points in y

    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]  # Should be same as dx if equally spaced

    X, Y = np.meshgrid(x, y)
    
    # Temporal parameters
    t_final = 3.0
    dt = 0.001  # time step (chosen to satisfy stability criterion dt <= dx^2/(4*alpha))
    nt = int(t_final/dt)
    
    # Initial condition: T(x,y,0) = 1 + 200*exp(-((x^2+y^2)/(2*0.1^2)))
    T = 1.0 + 200.0 * np.exp(- (X**2 + Y**2) / (2 * sigma**2))
    
    # Boundary conditions: T = 1 on all boundaries
    T[0, :] = 1.0
    T[-1, :] = 1.0
    T[:, 0] = 1.0
    T[:, -1] = 1.0

    # Precompute the source term, note it is independent of time in this problem.
    q = Q0 * np.exp(- (X**2 + Y**2) / (2 * sigma**2))

    # Time stepping loop
    for n in range(nt):
        Tn = T.copy()
        # Compute Laplacian using central differences for interior points
        # Note: Using dx for both x and y directions (dx == dy)
        T[1:-1, 1:-1] = Tn[1:-1, 1:-1] + dt * (
            alpha * ((Tn[2:, 1:-1] + Tn[:-2, 1:-1] + Tn[1:-1, 2:] + Tn[1:-1, :-2] - 4.0 * Tn[1:-1, 1:-1]) / dx**2)
            + q[1:-1, 1:-1]
        )
        # Reapply Dirichlet boundary conditions (T=1 on boundaries)
        T[0, :] = 1.0
        T[-1, :] = 1.0
        T[:, 0] = 1.0
        T[:, -1] = 1.0

    # Save final solution in a .npy file (2D array)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/T_2D_Unsteady_Heat_Equation.npy', T)

if __name__ == '__main__':
    main()