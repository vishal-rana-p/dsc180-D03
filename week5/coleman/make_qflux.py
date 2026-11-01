"""Derive a monthly Q-flux climatology from the relaxed slab-ocean spin-up.

JEM's relaxed slab (SlabOceanModel.step, forcing_method="relaxation") relaxes
the SST *anomaly* and then re-adds the climatology at the end of the step:
    T_new = T_clim(t+dt) + (T - T_clim(t) - dt*F_up/C) / (1 + dt/tau)
so the heat the relaxation supplies, beyond the surface flux, is
    Q = C * (dT_clim/dt - (T - T_clim)/tau),    C = rho*cp*h,
i.e. the seasonal warming/cooling of the climatology itself PLUS the restoring
term. Dropping the dT_clim/dt term leaves Q hundreds of W m-2 wrong in spring
and autumn over shallow mid-latitude seas (it only cancels in the annual mean).

JEM interpolates monthly records linearly with record k anchored at year
fraction k/12 (jem.utils.cycles.evaluate_cyclic_linear), i.e. at the START of
month k, for both the SST climatology and the Q-flux. T_clim(t) here uses the
same interpolation, and each 30-day record is binned to its nearest anchor.

Run (jcm env), after runs/earth-relax has finished:
    python make_qflux.py
"""
import glob

import numpy as np
import xarray as xr

RUN = "/home/colecam/runs/earth-relax"
OUT = "/home/colecam/runs/qflux_t30.nc"
BC = "/home/colecam/jax-gcm/jcm/data/bc/t30/clim/"

RHO, CP = 1025.0, 3985.0      # jem.constants ocean_density, ocean_specific_heat_capacity
TAU = 30 * 86400.0            # earth-slab's relaxation_time (s)
T_FREEZE = 271.35             # JEM floors the SST climatology here
SPINUP_RECORDS = 24           # drop the first ~2 years (30-day records)
N = 12

ocn = xr.open_mfdataset(sorted(glob.glob(f"{RUN}/ocn-*.nc"))).isel(time=slice(SPINUP_RECORDS, None)).load()
print(f"using {ocn.sizes['time']} records: {str(ocn.time.values[0])[:10]} .. {str(ocn.time.values[-1])[:10]}")

clim = xr.open_dataset(BC + "forcing.nc")["sst"].clip(min=T_FREEZE)   # (lon, lat, 12)
ocean = xr.open_dataset(BC + "terrain.nc")["lsm"].values < 0.5         # (lon, lat)
clim_values = clim.values

sst = ocn.sea_surface_temperature.transpose("time", "lon", "lat").values
heat_capacity = RHO * CP * ocn.mixed_layer_depth.isel(time=0).transpose("lon", "lat").values

# Year fraction of each record's midpoint (record time labels are interval midpoints).
time = ocn.time.dt
days_in_year = np.where(time.is_leap_year.values, 366.0, 365.0)
year_fraction = (time.dayofyear.values - 1 + time.hour.values / 24.0) / days_in_year

# T_clim and dT_clim/dt exactly as evaluate_cyclic_linear sees them.
pos = year_fraction * N
idx0 = np.floor(pos).astype(int) % N
idx1 = (idx0 + 1) % N
alpha = (pos - np.floor(pos))[:, None, None]
clim_at_t = (1 - alpha) * clim_values[..., idx0].transpose(2, 0, 1) + alpha * clim_values[..., idx1].transpose(2, 0, 1)
seconds_per_record = days_in_year[:, None, None] * 86400.0 / N
clim_tendency = (clim_values[..., idx1] - clim_values[..., idx0]).transpose(2, 0, 1) / seconds_per_record

heating = heat_capacity * (clim_tendency - (sst - clim_at_t) / TAU)    # (time, lon, lat), W m-2

# Bin to the nearest month-start anchor, so record k describes the Q-flux at year fraction k/12.
anchor = np.rint(pos).astype(int) % N
qflux = np.stack([heating[anchor == k].mean(axis=0) for k in range(N)], axis=-1)   # (lon, lat, 12)
qflux = np.where(ocean[..., None], qflux, 0.0)   # JEM ignores land cells anyway

# Same lon/lat/time coordinates as forcing.nc, which JEM's loader checks against its grid.
out = clim.copy(data=qflux).rename("qflux")
out.attrs = {"long_name": "Q-flux (ocean heat convergence)", "units": "W m-2", "positive": "warms mixed layer"}
out.to_dataset().to_netcdf(OUT)

# Sanity checks: the area-weighted ocean mean should be small next to the regional values.
w = np.cos(np.deg2rad(clim.lat.values))[None, :] * ocean
annual = qflux.mean(axis=-1)
print(f"records per anchor: {np.bincount(anchor, minlength=N).tolist()}")
print(f"global ocean mean Q: {(annual * w).sum() / w.sum():+.2f} W m-2")
print(f"annual-mean Q range: {annual[ocean].min():.0f} .. {annual[ocean].max():.0f} W m-2")
print(f"wrote {OUT}")
