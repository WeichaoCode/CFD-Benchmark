"""
Solution file for 1d, steady, diffusion solver
"""

import numpy as np
import A01_solver as solver
import matplotlib.pyplot as plt

# # Problem a input parameters
# g_a = 1E-4  # m^2/s diffusive coefficient
# L_a = 0.1  # m total length
# BCa_a = 10  # moles/m^3
# BCb_a = 100  # moles/m^3
# A_a = 1.0  # m^2 area
# qsrc = 0.0  # no source

# # solve a
# x_a, c_a = solver.solve_phi_array(L_a, 5, A_a, g_a, BCa_a, BCb_a, qsrc, qsrc)
#
# # plot approximation with exact solution
# xe_a = np.linspace(0.0, L_a, 100)
# ce_a = BCa_a + (BCb_a - BCa_a) / L_a * xe_a
#
# fig = plt.figure(1, figsize=(6, 4))
# plt.plot(xe_a, ce_a, 'k', label='exact')
# plt.plot(x_a, c_a, '*r', label='approximation')
# plt.xlabel("Length [m]", fontsize=14)
# plt.ylabel('Concentration [moles/m^3]', fontsize=14)
# plt.legend(fontsize=14, loc=4)
# plt.title('Problem a: Species concentration')

##########################################################################

# Problem b geometry, constant parameters, boundary conditions
L = 0.5  # m, length
d = 0.01  # m, diamter of rod
area = np.pi * (d / 2) ** 2  # m*m, cross-sectional area
# # p = np.pi*d # m, perimeter of rod
k = 1000  # W/mK, thermal conductivity
Ta = 100.  # C, constant temp boundary A
Tb = 200.  # C, constant temp boundary B
qu = 2000.0e3  # W/m^3 independent heat source
qj = 0.0  # no dependent heat source

# # Discretize domain
N_b = [5, 10, 25]
# # N_b = [5]
form = ['*r', 'oc', '+b']
label = ['5 CV', '10 CV', '25 CV']

fig = plt.figure(2, figsize=(6, 4))
nx = 1000
x = np.linspace(0, L, nx)
phi_exact = ((Tb - Ta) / L + qu / (2 * k) * (L - x)) * x + Ta
plt.plot(x, phi_exact, 'k', label='exact')
T = None
# solve temperature distribution
for i, N in enumerate(N_b):
    xcv, Tcv = solver.solve_phi_array(L, N, area, k, Ta, Tb, qu, qj)
    if N == 25:
        T = Tcv
    plt.plot(xcv, Tcv, form[i], label=label[i])

# Plot exact solution and approximated solution
plt.xlabel("Length [m]", fontsize=14)
plt.ylabel('Temperature [C]', fontsize=14)
plt.legend(fontsize=14, loc=4)
plt.title('Problem b: Heated rod')


import os

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "results")

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# Get the current Python file name without extension
python_filename = os.path.splitext(os.path.basename(__file__))[0]

# Define the file name dynamically
output_file = os.path.join(OUTPUT_FOLDER, f"T_{python_filename}.npy")

# Save the array u in the results folder
np.save(output_file, T)
# # problem c: rod with convection input parameters
# # h = 62.5 # W/m^2K convective htc
# # peri = np.pi*d
# # T_inf = 20 # C ambient temperature
# # qu = h*peri/area*T_inf
# # qj = -h*peri/area
#
# # # exact solution
# # n = np.sqrt(h*peri/k/area)
# # Bcoef = (Tb-T_inf - (Ta-T_inf)*np.exp(n*L))/(np.exp(-n*L)-np.exp(n*L))
# # Acoef = Ta - Bcoef - T_inf
# # Tc_ex = Acoef*np.exp(n*x) + Bcoef*np.exp(-n*x) + T_inf
# # fig = plt.figure(3, figsize=(6, 4))
# # plt.plot(x,Tc_ex,'k',label='exact')
# # N_c = [5, 10]
#
# # # solve temperature distribution
# # for i,N in enumerate(N_c):
# #     x_c, T_c = solver.solve_phi_array(L,N,area,k,Ta,Tb,qu,qj)
# #     plt.plot(x_c, T_c, form[i], label=label[i])
#
# # # plot temperature distribution part c
# # plt.xlabel("Length [m]",fontsize=14)
# # plt.ylabel('Temperature [C]',fontsize=14)
# # plt.legend(fontsize=14, loc=2)
# # plt.title('Problem c: Circular fin')
