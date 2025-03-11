import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Constants
ALPHA = 0.01 # Thermal diffusivity
Q0 = 200 # Maximum heating at the center
SIGMA = 0.1 # Controls radial decay of heat source
NX, NY = 100, 100 # Number of grid points in x and y directions
DT = 0.01 # Time step
TFINAL = 1.0

# Functions
def initialize_grid():
    """Initialize grid and boundary conditions."""
    grid = np.zeros((NX+2, NY+2))
    return grid

def calculate_source(x, y):
    """Calculate the heat source term."""
    q = Q0 * np.exp(-(x**2 + y**2) / (2*SIGMA**2))
    return q

def dufort_frankel_method(grid):
    """Perform one time step using Dufort-Frankel method."""
    # Calculate heat source term
    x = np.linspace(-1, 1, NX+2)
    y = np.linspace(-1, 1, NY+2)
    X, Y = np.meshgrid(x, y)
    q = calculate_source(X, Y)
    
    # Apply Dufort-Frankel method
    new_grid = np.copy(grid)
    new_grid[1:-1, 1:-1] = (1 / (2 + 2 * DT * ALPHA * (2 / DX**2 + 2 / DY**2))) * ((2 - 2 * DT * ALPHA * (2 / DX**2 + 2 / DY**2)) * grid[1:-1, 1:-1] + DT * ALPHA * (grid[:-2, 1:-1] + grid[2:, 1:-1] / DX**2 + grid[1:-1, :-2] + grid[1:-1, 2:] / DY**2 + 2 * q[1:-1, 1:-1]))
    return new_grid

def ADI_method(grid):
    """Perform one time step using ADI method."""
    # to be completed...

def plot_temperature(grid, t):
    """Plot the temperature field using a heatmap."""
    plt.figure(figsize=(6,5))
    plt.imshow(grid, cmap='hot', extent = [-1,1,-1,1], origin='lower')
    plt.title('Temperature at time {:.2f}'.format(t))
    plt.colorbar(label='Temperature (°C)')
    plt.show()

def compare_methods():
    """Compare the temperature evolutions obtained using the two methods."""
    # to be completed...

# Main part of the program
DX = 2 / NX # The x grid size
DY = 2 / NY # The y grid size

# Initialize the grid
grid = initialize_grid()

# Time evolution
t = 0.0
while t < TFINAL:
    t += DT
    grid_dufort = dufort_frankel_method(grid)
    #grid_adi = ADI_method(grid)
    if t % 0.1 == 0:
        plot_temperature(grid_dufort, t)
        #plot_temperature(grid_adi, t)
    grid = grid_dufort

# Compare methods
#compare_methods()