import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # avoid GPU out-of-memory

import json, subprocess, time
import jax
jax.config.update("jax_debug_infs", False)
jax.config.update("jax_debug_nans", False)

from importlib import resources
from jcm.model import Model
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.speedy_coords import get_speedy_coords

OUT = "runs/control"
N_YEARS = 31            # 1 spin-up + 30
SAVE_INTERVAL = 5       # days; 73 saves per 365-day year
KEEP = [
    "shortwave_rad.fsol", "shortwave_rad.ftop", "longwave_rad.ftop",
    "surface_flux.t0", "surface_exchange.precipitation",
    "condensation.precls", "convection.precnv", "shortwave_rad.cloudc",
    "temperature", "u_wind", "v_wind", "normalized_surface_pressure",
    "mod_radcon.ablco2",
]
os.makedirs(OUT, exist_ok=True)

data_dir = resources.files("jcm.data.bc.t30.clim")
coords = get_speedy_coords()
terrain = TerrainData.from_file(data_dir / "terrain.nc", coords=coords)
forcing = ForcingData.from_file(data_dir / "forcing.nc", coords=coords, align_mode="wrap_year")
model = Model(coords=coords, terrain=terrain)

# Record the exact configuration
config = {
    "machine": "laptop (WSL2)",
    "devices": str(jax.devices()),
    "jax_version": jax.__version__,
    "jcm_commit": subprocess.run(["git", "-C", os.path.expanduser("~/jax-gcm"),
                                  "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "boundary_conditions": "jcm.data.bc.t30.clim (terrain.nc, forcing.nc, align_mode=wrap_year)",
    "co2_vmr_ppmv": float(forcing.co2_vmr),
    "n_years": N_YEARS, "spinup_years": 1,
    "save_interval_days": SAVE_INTERVAL, "total_time_per_chunk_days": 365,
    "output_averages": True,
    "saved_variables": KEEP,
    "saved_dtype": "float32, zlib complevel 4",
}
with open(f"{OUT}/config.json", "w") as f:
    json.dump(config, f, indent=2)
print(config, flush=True)

for year in range(N_YEARS):
    t0 = time.perf_counter()
    kw = dict(save_interval=SAVE_INTERVAL, total_time=365,
              output_averages=True, forcing=forcing)
    out = model.run(**kw) if year == 0 else model.resume(**kw)
    ds = out.to_xarray()[KEEP].astype("float32")
    ds.to_netcdf(f"{OUT}/year{year:03d}.nc",
                 encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print(f"year {year} done in {time.perf_counter() - t0:.1f} s", flush=True)