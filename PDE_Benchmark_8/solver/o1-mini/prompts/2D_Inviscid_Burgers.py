import numpy as np

# Parameters
Lx, Ly = 2.0, 2.0
T = 0.40
Nx, Ny = 101, 101
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize u and v
u = np.ones((Ny, Nx))
v = np.ones((Ny, Nx))

# Set initial conditions: u = v = 2 for 0.5 <= x <=1 and 0.5 <= y <=1
mask = (X >= 0.5) & (X <=1.0) & (Y >=0.5) & (Y <=1.0)
u[mask] = 2.0
v[mask] = 2.0

u_new = np.empty_like(u)
v_new = np.empty_like(v)

# Time step based on CFL condition
CFL = 0.5
max_u = np.max(np.abs(u))
max_v = np.max(np.abs(v))
dt = CFL * min(dx, dy) / (max_u + max_v)
nt = int(T / dt) + 1
dt = T / nt

for n in range(nt):
    # Compute fluxes using upwind scheme
    u_x = np.zeros_like(u)
    u_y = np.zeros_like(u)
    v_x = np.zeros_like(v)
    v_y = np.zeros_like(v)
    
    # Upwind differences for u
    u_x[1:-1,1:-1] = np.where(u[1:-1,1:-1] > 0,
                              (u[1:-1,1:-1] - u[1:-1,0:-2]) / dx,
                              (u[1:-1,2:] - u[1:-1,1:-1]) / dx)
    u_y[1:-1,1:-1] = np.where(v[1:-1,1:-1] > 0,
                              (u[1:-1,1:-1] - u[0:-2,1:-1]) / dy,
                              (u[2:,1:-1] - u[1:-1,1:-1]) / dy)
    
    # Upwind differences for v
    v_x[1:-1,1:-1] = np.where(u[1:-1,1:-1] > 0,
                              (v[1:-1,1:-1] - v[1:-1,0:-2]) / dx,
                              (v[1:-1,2:] - v[1:-1,1:-1]) / dx)
    v_y[1:-1,1:-1] = np.where(v[1:-1,1:-1] > 0,
                              (v[1:-1,1:-1] - v[0:-2,1:-1]) / dy,
                              (v[2:,1:-1] - v[1:-1,1:-1]) / dy)
    
    # Update u and v
    u_new[1:-1,1:-1] = u[1:-1,1:-1] - dt * (u[1:-1,1:-1] * u_x[1:-1,1:-1] + v[1:-1,1:-1] * u_y[1:-1,1:-1])
    v_new[1:-1,1:-1] = v[1:-1,1:-1] - dt * (u[1:-1,1:-1] * v_x[1:-1,1:-1] + v[1:-1,1:-1] * v_y[1:-1,1:-1])
    
    # Apply boundary conditions: u = v =1 on all boundaries
    u_new[0,:] = 1.0
    u_new[-1,:] = 1.0
    u_new[:,0] = 1.0
    u_new[:,-1] = 1.0
    
    v_new[0,:] = 1.0
    v_new[-1,:] = 1.0
    v_new[:,0] = 1.0
    v_new[:,-1] = 1.0
    
    # Update for next time step
    u, v = u_new.copy(), v_new.copy()

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/u_2D_Inviscid_Burgers.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/v_2D_Inviscid_Burgers.npy', v)