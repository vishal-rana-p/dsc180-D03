## Example Prompts

### Understanding papers and concepts
1. "what are hyperparamters and what hyperparamteres did the paper use?" — used to understand the
   Gaussian-process settings in the ClimateBench paper (Watson-Parris et al., 2022), such as the
   number of EOFs and the kernel.
2. "when we compare NorESM2 erf to 2x, 4x, 8x speedy implementation, why are we comparing to the
   myhre 1998" — used to understand why the Myhre et al. (1998) CO₂ forcing formula is the
   benchmark for both models.
3. "why are all the points clustered? how does Gregory regression even work?" — used to understand
   the method from Gregory et al. (2004) and why a fast-responding slab ocean leaves few useful
   points.
4. "what does this variability from the paper explain about question 3 asking about ensemble
   variability" — used to interpret the "Variability" row of the ClimateBench leaderboard as the
   lowest error any emulator could physically achieve.

### Writing code
1. "explain what I need to do, and then lets work on implementing it" — used to plan and then
   write a cleaned-up pattern-scaling implementation, including both the oracle and the
   input-driven versions.
2. "we need to recreate this using jax" — used to write the JAX re-implementation of pattern
   scaling and compare its scores with the scikit-learn version.
3. "lets check the other assets and start running the regression after confirming what the data
   looks like within the .nc files" — used to inspect the simulation summary files and then write
   the Gregory regression code for them.
4. "format it into a table like with pandas data frame values" — used to turn printed results
   into pandas tables for the report.

### Explaining starter notebooks and existing code
1. "let me paste the code, chunk by chunk and we can walk through what is going on and what to do
   for task 1" — used to go through the provided notebook one block at a time:
   data loading, the regression, prediction, and plotting.
2. "what does this chunk of code do? [pasted code] ... what is got?" — used to understand how the
   notebook computes the test-time global-mean temperature in two ways: read from the test
   outputs (the "oracle") or predicted from CO₂.
3. "how to open slabocean" — used to locate the `SlabOceanModel` source file in the JEM package,
   then read its slab-ocean equation and Q-flux options.
4. "[pasted run_abrupt_co2.py] ... let me know what I need from that task to work on this one" —
   used to understand a teammate's script for branching the abrupt-CO₂ runs from the control, and
   why it swaps CO₂ into the restored model state instead of using a command-line restart.


## How We Tested the Code

We ran every script ourselves and checked results against an independent reference before
trusting them. For the emulators, our pattern-scaling scores matched the published ClimateBench
leaderboard (total temperature NRMSE 0.320), and the scikit-learn, NumPy, and JAX versions gave
identical scores. For the slab ocean, we checked the Q-flux's signs and global mean against
physical expectations, ran a 2-day test before the long run, and let the model's own loader
reject a file with flipped latitudes. For the analysis, we confirmed the control had no drift
(−0.02 K per decade), checked that the CSV and NetCDF summaries matched exactly, verified map
orientation against the model's coastlines, and removed a spin-up year that had inflated the
week 4 ERF uncertainty. Checks like these caught several AI errors, including an incorrect NRMSE
normalization and a misread energy-budget result.
