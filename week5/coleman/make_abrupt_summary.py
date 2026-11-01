"""Condense the abrupt 2x/4xCO2 runs (2.4 + 3.6 GB of raw output) into what the
Gregory / ECS / warming-pattern steps need: data/abrupt_summary.nc and
data/abrupt_annual.csv.

Years are consecutive blocks of 12 records (12 x 30 days) counted from the CO2
jump, not calendar years: the output is 30-day means, so a calendar year holds
12 or 13 records and the last calendar year of a run is partial. The control is
treated the same way over its equilibrated 2010-2029 window.

Besides TOA net, the surface net heat flux is kept: in this model the TOA
anomaly does not decay as the run equilibrates (see README), so the surface
flux -- what actually heats the slab -- is the energy-uptake measure to check.

Run (jcm env):  python make_abrupt_summary.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RUNS = Path("/home/colecam/runs")
OUT = Path(__file__).parent / "data"
EQ_YEARS = 10          # equilibrium = mean of each run's last 10 years

ATM = ["surface_flux.t0", "shortwave_rad.ftop", "longwave_rad.ftop",
       "surface_exchange.net_heat_flux", "condensation.precls", "convection.precnv"]


def load(run_dir, window=None):
    """30-day-mean fields of one run, trimmed to whole 12-record years."""
    atm = xr.open_mfdataset(sorted(glob.glob(f"{run_dir}/atm-*.nc")),
                            preprocess=lambda d: d[ATM])
    ocn = xr.open_mfdataset(sorted(glob.glob(f"{run_dir}/ocn-*.nc")),
                            preprocess=lambda d: d[["sea_surface_temperature"]])
    if window:
        atm, ocn = atm.sel(time=window), ocn.sel(time=window)
    n = atm.sizes["time"] // 12 * 12
    ds = xr.Dataset({
        "tas": atm["surface_flux.t0"],
        "toa_net": atm["shortwave_rad.ftop"] - atm["longwave_rad.ftop"],
        # The atmosphere's view: positive down, into the surface (per its attrs).
        "sfc_net": atm["surface_exchange.net_heat_flux"],
        # g m-2 s-1 -> mm day-1
        "pr": (atm["condensation.precls"] + atm["convection.precnv"]) * 86.4,
        "sst": ocn["sea_surface_temperature"].assign_coords(lon=atm.lon, lat=atm.lat),
    }).isel(time=slice(0, n)).load()
    year = np.repeat(np.arange(1, n // 12 + 1), 12)
    month = np.tile(np.arange(1, 13), n // 12)
    return ds.assign_coords(year=("time", year), month=("time", month))


def global_mean(da):
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(["lon", "lat"])


ATTRS = {
    "tas": {"units": "K", "long_name": "near-surface air temperature (surface_flux.t0)"},
    "toa_net": {"units": "W m-2", "long_name": "TOA net flux, positive down"},
    "sfc_net": {"units": "W m-2", "long_name": "surface net heat flux, positive into the surface"},
    "pr": {"units": "mm day-1", "long_name": "total precipitation"},
    "sst": {"units": "K", "long_name": "slab SST (land cells hold a fill value)"},
}

runs = {"control": load(RUNS / "earth-qflux-v2", slice("2010", "2029"))}
for f in (2, 4):
    runs[f"{f}xCO2"] = load(RUNS / f"abrupt-{f}xco2")

out, rows = {}, []
for name, ds in runs.items():
    annual = global_mean(ds[["tas", "toa_net", "sfc_net", "pr"]]).groupby("year").mean()
    last = ds.where(ds.year > ds.year.max() - EQ_YEARS, drop=True)
    clim = last.groupby("month").mean()
    for v in ["tas", "toa_net", "sfc_net", "pr"]:
        out[f"{v}_global_annual_{name}"] = annual[v].rename(year=f"year_{name}").assign_attrs(ATTRS[v])
    for v in ["tas", "pr", "toa_net", "sst"]:
        out[f"{v}_eqclim_{name}"] = clim[v].transpose("month", "lon", "lat").assign_attrs(ATTRS[v])
    rows.append(annual.to_dataframe()[["tas", "toa_net", "sfc_net", "pr"]].assign(run=name))

summary = xr.Dataset(out)
summary.attrs = {
    "source": "JEM earth-slab + slab_qflux (SPEEDY T31 atmosphere, T30 boundary files). "
              "control: CO2 420 ppmv, years 2010-2029 of the equilibrated control. "
              "2xCO2 / 4xCO2: 840 / 1680 ppmv, branched from the control's end state.",
    "years": "year_<run> counts consecutive 12 x 30-day blocks (year 1 = first year after the CO2 jump).",
    "eqclim": f"*_eqclim_<run>: monthly climatology of each run's last {EQ_YEARS} years "
              "(for the control, its last 10 of 2010-2029). Equilibrium dT/dP maps = "
              "annual mean of <run> minus control.",
}
summary.to_netcdf(OUT / "abrupt_summary.nc")

table = pd.concat(rows).reset_index().set_index(["run", "year"]).round(4)
table.to_csv(OUT / "abrupt_annual.csv")

ctl = table.loc["control"].mean()
print(f"control (2010-2029): {ctl.round(2).to_dict()}")
for name in ("2xCO2", "4xCO2"):
    d = table.loc[name] - ctl
    print(f"{name}: anomalies from control, first 3 and last {EQ_YEARS} years")
    print(pd.concat([d.head(3).T, d.tail(EQ_YEARS).mean().rename("last10")], axis=1).round(2))
print(f"wrote {OUT / 'abrupt_summary.nc'} and {OUT / 'abrupt_annual.csv'}")
