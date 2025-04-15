# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameter Definitions
alpha = 0.1 # Thermal diffusivity
Q_0 = 200 # Maximum heating at the center 
sigma = 0.1 # Controls radial decay of heat source

# Define the grid parameters
Nx = 100 # Number of points in x axis
Ny = 100 # Number of points in y axis
T_end = 0.5 # End Time
dt = 0.01 # time step size
Nt = int(T_end/dt) # Number of time steps

# Generate the grid
x = np.linspace(-1, 1, Nx) 
y = np.linspace(-1, 1, Ny) 
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y) # Grid for spatial domain

# Set initial condition
T = np.zeros((Nx,Ny)) 

# Source term
q = Q_0 * np.exp(-(X**2 + Y**2) / (2*sigma**2))

# Function to solve heat equation using Dufort-Frankel method
def dufort_frankel(T,q,alpha,dt,dx,dy,Nt):
    # T : initial temperature field
    # alpha : thermal diffusivity
    # dt : time step
    # dx : spatial step in x direction
    # dy : spatial step in y direction
    # Nt : Total number of time steps
    # Function returns the temperature field evolution as an array
    
    # Computations
    for t in range(Nt):
        T_old = T.copy()
        T[1:-1,1:-1] = ((1 - 2*alpha*dt*(1/dx**2 + 1/dy**2))*T_old[1:-1,1:-1] 
                        + alpha*dt/dx**2 * (T[2:,1:-1] + T[:-2,1:-1]) 
                        + alpha*dt/dy**2 * (T[1:-1,2:] + T[1:-1,:-2]) 
                        + dt*q[1:-1,1:-1])
    return T
  
# Call the function
T_final_Dufort_Frankel = dufort_frankel(T,q,alpha,dt,dx,dy,Nt)

# Function to plot the final result
def plot_field(T, title=''):
    plt.figure(figsize=(5,5))
    plt.imshow(T, interpolation='none', extent=(x[0], x[-1], y[0], y[-1]))
    plt.title(title)
    plt.colorbar()
    plt.show()

# Plot the final temperature field computed by Dufort-Frankel method
plot_field(T_final_Dufort_Frankel, "Temperature distribution by Dufort-Frankel Method")