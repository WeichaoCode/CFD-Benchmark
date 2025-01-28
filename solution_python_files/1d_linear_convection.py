import numpy as np
import os
import json

nx = 41  # try changing this number from 41 to 81 and Run All ... what happens?
dx = 2 / (nx - 1)
nt = 25  # nt is the number of timesteps we want to calculate
dt = .025  # dt is the amount of time each timestep covers (delta t)
c = 1  # assume wavespeed of c = 1
u = np.ones(nx)  # numpy function ones()
u[int(.5 / dx):int(1 / dx + 1)] = 2  # setting u = 2 between 0.5 and 1 as per our I.C.s
un = np.ones(nx)  # initialize a temporary array
for n in range(nt):  # loop for values of n from 0 to nt, so it will run nt times
    un = u.copy()  ##copy the existing values of u into un
    for i in range(1, nx):  ## you can try commenting this line and...
        # for i in range(nx): ## ... uncommenting this line and see what happens!
        u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1])

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