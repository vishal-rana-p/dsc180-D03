"""Abrupt CO2 run branched from the end of the Q-flux slab control.

Why this is a script and not `python -m jem.main ... coupled_run.initial_condition=`:
the atmosphere's carry holds its boundary forcing, CO2 included, and JEM
restores EVERY leaf of a checkpoint. Branching from the control checkpoint on
the command line therefore silently puts CO2 back to the control's 420 ppmv,
whatever forcing file you pass (checked: mod_radcon.ablco2 came out 6.0, the
control value, instead of 12.0 for 2xCO2). Here the control state is loaded,
only its `co2_vmr` is replaced with the experiment's, and the run continues.

Everything else is the control's configuration: earth-slab + slab_qflux with
the same Q-flux file. The forcing file is the packaged T30 forcing.nc plus a
scalar `co2` variable (made by this script if missing).

Usage (jcm env, from anywhere):
    python run_abrupt_co2.py 2 --years 40      # -> /home/colecam/runs/abrupt-2xco2
    python run_abrupt_co2.py 4 --years 60      # -> /home/colecam/runs/abrupt-4xco2
Rerunning the same command resumes from the run's own checkpoint.
"""
import argparse
import logging
from pathlib import Path

import jax.numpy as jnp
import xarray as xr

import jem
from jem.configurations import load

RUNS = Path("/home/colecam/runs")
CONTROL_CHECKPOINT = RUNS / "earth-qflux-v2" / "checkpoint"
QFLUX = RUNS / "qflux_t30.nc"
BASE_FORCING = Path("/home/colecam/jax-gcm/jcm/data/bc/t30/clim/forcing.nc")
CONTROL_CO2_PPMV = 420.0      # jcm's DEFAULT_CO2_VMR_PPMV, what the control ran with

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("factor", type=float, help="CO2 multiple of the control (2 or 4)")
parser.add_argument("--years", type=int, default=40, help="length in 360-day years (12 x 30-day chunks)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
logging.getLogger("jem").setLevel(logging.INFO)

tag = f"{args.factor:g}xco2"
forcing_file = RUNS / f"forcing_{tag}.nc"
if not forcing_file.exists():
    ds = xr.open_dataset(BASE_FORCING).load()
    ds["co2"] = xr.DataArray(CONTROL_CO2_PPMV * args.factor, attrs={"units": "ppmv"})
    ds.to_netcdf(forcing_file)

exp = load(
    "earth-slab",
    ocean="slab_qflux",
    coupled_run="long_run",
    **{
        "ocean.q_flux_file": str(QFLUX),
        "atmosphere.forcing.file": str(forcing_file),
        # A user forcing file is not in jcm's manifest, so `auto` cannot resolve it;
        # it is a 12-month climatology, as in week 4.
        "atmosphere.forcing.align": "wrap_year",
        "coupled_run.total_time": f"{args.years * 360} days",
        "coupled_run.output_dir": str(RUNS / f"abrupt-{tag}"),
    },
)

# The experiment's own CO2, as the freshly built atmosphere holds it.
co2 = exp.coupler.initialize().components["atm"]["forcing"].co2_vmr
assert jnp.allclose(co2, CONTROL_CO2_PPMV * args.factor), f"forcing file gave CO2 = {co2}"

# Control end state, clock restarted at the experiment's start date, CO2 swapped in.
control = exp.coupler.restart_clock(exp.coupler.load_carry(CONTROL_CHECKPOINT, check_clock=False))
atm = dict(control.components["atm"])
atm["forcing"] = atm["forcing"].copy(co2_vmr=co2)
start = control.replace(components={**control.components, "atm": atm})

# If this run already has a checkpoint, run_chunked resumes from it and ignores `start`.
result = jem.run_chunked(exp.coupler, initial_carry=start, **exp.run_kwargs)
print(f"{tag}: {result.steps_completed} coupled steps, completed={result.completed}")
