import sys, glob
import xarray as xr

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/control"
files = sorted(glob.glob(f"{RUN}/year*.nc"))

ds = xr.open_mfdataset(files, combine="by_coords")
monthly = ds.resample(time="MS").mean()
monthly["n_saves"] = ds.time.resample(time="MS").count()   # 5-day saves per month

monthly = monthly.load()
print("months:", monthly.sizes["time"], "| unique:", monthly.indexes["time"].is_unique)
print("saves per month (min/max):", int(monthly.n_saves.min()), int(monthly.n_saves.max()))

monthly.to_netcdf(f"{RUN}/monthly_means.nc",
                  encoding={v: {"zlib": True, "complevel": 4} for v in monthly.data_vars})