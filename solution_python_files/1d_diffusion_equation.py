import json
import os
import numpy  # loading our favorite library

nx = 41
dx = 2 / (nx - 1)
# nt = 50  # the number of timesteps we want to calculate
nu = 0.3  # the value of viscosity
# sigma = .2  # sigma is a parameter, we'll learn more about it later
# dt = 0.0025  # dt is defined using sigma ... more later!
sigma = 0.2  # CFL-like number for stability
dt = sigma * dx**2 / nu  # time step
nt = 100  # number of time steps

u = numpy.ones(nx)  # a numpy array with nx elements all equal to 1.
u[int(.5 / dx):int(1 / dx + 1)] = 2  # setting u = 2 between 0.5 and 1 as per our I.C.s

un = numpy.ones(nx)  # our placeholder array, un, to advance the solution in time

for n in range(nt):  # iterate through time
    un = u.copy()  ##copy the existing values of u into un
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx ** 2 * (un[i + 1] - 2 * un[i] + un[i - 1])

##############################################
# The following lines are used to print output
##############################################

# Identify the filename of the running script
script_filename = os.path.basename(__file__)

# Define the JSON file
json_filename = "/opt/CFD-Benchmark/data/output_true.json"

# Load existing JSON data if the file exists
if os.path.exists(json_filename):
    with open(json_filename, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = {}  # Handle empty or corrupted file
else:
    data = {}

# Save filename and output array in a structured format
data[script_filename] = {
    "filename": script_filename,
    "u": u.tolist()  # Convert NumPy array to list for JSON serialization
}

# Save the updated JSON data
with open(json_filename, "w") as file:
    json.dump(data, file, indent=4)

print(f"Saved output of {script_filename} to {json_filename}")