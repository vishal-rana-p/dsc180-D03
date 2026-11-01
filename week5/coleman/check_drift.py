"""Drift of the 30-year Q-flux slab control against the week-4 SST climatology.

Week 4 prescribed the `sst` of jcm's packaged T30 forcing.nc; the Q-flux control
starts from that climatology and runs freely, so its departure from it, and any
trend in that departure, is the drift.

Run (jcm env):
    python check_drift.py
Writes drift_yearly.csv (annual means: SST bias in K, TOA net in W m-2, positive
down), drift_timeseries.png and drift_map.png next to this script.
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RUN = "/home/colecam/runs/earth-qflux-v2"
BC = "/home/colecam/jax-gcm/jcm/data/bc/t30/clim/"
HERE = Path(__file__).parent
T_FREEZE = 271.35
TREND_YEARS = 20          # trend fitted over the last 20 complete years

clim = xr.open_dataset(BC + "forcing.nc")["sst"].clip(min=T_FREEZE).values    # (lon, lat, 12)
ocean = xr.open_dataset(BC + "terrain.nc")["lsm"].values < 0.5                  # (lon, lat)

ocn = xr.open_mfdataset(sorted(glob.glob(f"{RUN}/ocn-*.nc"))).load()
atm = xr.open_mfdataset(sorted(glob.glob(f"{RUN}/atm-*.nc")))[
    ["shortwave_rad.ftop", "longwave_rad.ftop"]].load()

lat = ocn.lat.values
area = np.cos(np.deg2rad(lat))[None, :] * np.ones_like(ocean, dtype=float)
w_ocean = area * ocean
w_icefree = w_ocean * (np.abs(lat) < 60)[None, :]


def area_mean(field, weights):
    """Weighted mean over (lon, lat) of a (time, lon, lat) array."""
    return (field * weights).sum(axis=(-2, -1)) / weights.sum()


# Month-matched SST bias: each 30-day record against its calendar month of the climatology.
sst = ocn.sea_surface_temperature.transpose("time", "lon", "lat").values
month = ocn.time.dt.month.values - 1
bias = sst - clim[..., month].transpose(2, 0, 1)
toa_net = (atm["shortwave_rad.ftop"] - atm["longwave_rad.ftop"]).transpose("time", "lon", "lat").values

series = xr.Dataset(
    {
        "bias_icefree": ("time", area_mean(bias, w_icefree)),
        "bias_ocean": ("time", area_mean(bias, w_ocean)),
        "rms_icefree": ("time", np.sqrt(area_mean(bias**2, w_icefree))),
        "toa_net": ("time", area_mean(toa_net, area)),
    },
    coords={"time": ocn.time.values},
)
yearly = series.groupby("time.year").mean()
counts = series.time.groupby("time.year").count()
yearly = yearly.sel(year=counts.year[counts >= 12])     # complete years only

yearly.to_dataframe().round(4).to_csv(HERE / "drift_yearly.csv")

print(" year  SST bias 60S-60N  SST bias all ocean  RMS bias 60S-60N  TOA net (W m-2)")
for y in yearly.year.values:
    r = yearly.sel(year=y)
    print(f" {y}   {float(r.bias_icefree):+7.3f} K        {float(r.bias_ocean):+7.3f} K"
          f"          {float(r.rms_icefree):5.2f} K        {float(r.toa_net):+6.2f}")


def trend_per_decade(values):
    """OLS slope per decade and its standard error, for annual values."""
    years = np.arange(len(values))
    slope, intercept = np.polyfit(years, values, 1)
    resid = values - (slope * years + intercept)
    se = np.sqrt((resid**2).sum() / (len(values) - 2) / ((years - years.mean())**2).sum())
    return 10 * slope, 10 * se


last = yearly.isel(year=slice(-TREND_YEARS, None))
y0, y1 = int(last.year[0]), int(last.year[-1])
print(f"\nlast {TREND_YEARS} years ({y0}-{y1}):")
for name, units in [("bias_icefree", "K"), ("bias_ocean", "K"), ("toa_net", "W m-2")]:
    slope, se = trend_per_decade(last[name].values)
    print(f"  {name:13s} mean {float(last[name].mean()):+7.3f} {units:5s}  "
          f"trend {slope:+.3f} +/- {se:.3f} {units}/decade")

# Figures: annual series, and the bias map over the last 10 years.
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(yearly.year, yearly.bias_icefree, marker="o", label="60S-60N ocean")
axes[0].plot(yearly.year, yearly.bias_ocean, marker="s", label="all ocean")
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_ylabel("SST - week-4 clim (K)")
axes[0].legend()
axes[1].plot(yearly.year, yearly.toa_net, marker="o", color="C3")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_ylabel("TOA net down (W m$^{-2}$)")
axes[1].set_xlabel("model year")
fig.suptitle("Q-flux slab control: annual-mean drift")
fig.tight_layout()
fig.savefig(HERE / "drift_timeseries.png", dpi=150)

last10 = ocn.time.dt.year.values >= int(yearly.year[-1]) - 9
mean_bias = np.where(ocean, bias[last10].mean(axis=0), np.nan)
fig, ax = plt.subplots(figsize=(9, 4.5))
mesh = ax.pcolormesh(ocn.lon.values, lat, mean_bias.T, cmap="RdBu_r", vmin=-3, vmax=3, shading="nearest")
fig.colorbar(mesh, ax=ax, label="K")
ax.set_title(f"SST bias vs week-4 climatology, mean of last 10 years")
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
fig.tight_layout()
fig.savefig(HERE / "drift_map.png", dpi=150)
print(f"\nwrote {HERE / 'drift_timeseries.png'} and {HERE / 'drift_map.png'}")
