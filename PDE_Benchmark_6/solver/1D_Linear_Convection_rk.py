import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# Define function to calculate the second order central difference of u
def second_derivative(N_x, h):
    diagonal = np.ones(N_x-2)*-2
    upper = np.ones(N_x-3)
    lower = np.ones(N_x-3)
    A = sparse.diags([diagonal, upper, lower], [0, 1, -1])
    A /= h*h
    return A

def main(N_x=101, T=10, c=1, epsilon=[0, 5e-4], filename='solution.npy'):
    # Discretize the domain
    x_start, x_end = -5, 5
    x = np.linspace(x_start, x_end, N_x)
    h = x[1] - x[0]
    dt = h / (2 * c)
    N_t = int(T / dt)
    
    # Precompute grid values
    A = second_derivative(N_x, h)
    B = sparse.eye(N_x-2)
    u = np.exp(-x**2)
    
    for eps in epsilon:
        copy_u = u.copy()
        for n in range(N_t):
            # Time integration
            k1 = -c*A.dot(copy_u[1:-1]) + eps*A.dot(copy_u[1:-1])
            k2 = -c*A.dot(copy_u[1:-1] + dt/2*k1) + eps*A.dot(copy_u[1:-1] + dt/2*k1)
            k3 = -c*A.dot(copy_u[1:-1] + dt/2*k2) + eps*A.dot(copy_u[1:-1] + dt/2*k2)
            k4 = -c*A.dot(copy_u[1:-1] + dt*k3) + eps*A.dot(copy_u[1:-1] + dt*k3)
            
            copy_u[1:-1] += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Apply periodic boundary conditions
            copy_u[0] = copy_u[-2]
            copy_u[-1] = copy_u[1]
        # Save solution
        np.save(str(eps) + "_" + filename, copy_u)

        plt.plot(x, copy_u, label = 'Damping Factor = ' + str(eps))

    plt.title('1D linear convection-diffusion equation')
    plt.xlabel('Position, x')
    plt.ylabel('Amplitude, u')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()