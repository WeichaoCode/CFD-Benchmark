# CFD-Benchmark
## How to use it
* step 1: add problem descriptions to cfd_probelms.json
* step 2: call function /utils/generate_prompt.py to generate prompt automatically get cfd_prompts.json
* step 3: if possible call API, but to save money, do it manually
* step 4: copy generated code to /utils/generate_response.py to save code to folder generated_python_files
* step 5: need to prepare solution (true solution not written by me) in folder solution_python_files
* step 6: call /utils/automat_run.py to run all files and save output (final time solution) to output_true.json and output_generate.json
* step 7: call /utils/compare.py to compute the MSE and plot.
* # structured mesh
    * ## simple incompressible fluid flow
        * 1d burgers' equation
        * 1d diffusion equation
        * 1d linear convection equation
        * 1d nonlinear convection equation
        * 2d burgers' equation
        * 2d convection equation
        * 2d diffusion equation
        * 2d laplace equation
        * 2d linear convection equation
        * 2d poisson equation
        * cavity flow with navier–stokes
        * channel flow with navier–stokes
