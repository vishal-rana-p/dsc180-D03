import os, subprocess
import xarray as xr

BASE = os.path.expanduser("~/jcm/jcm_runs/reference")
OUT, TMP = f"{BASE}/era5_years", f"{BASE}/tmp"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)
YEARS = range(2005, 2015)  #same years as JCM forcing
#YEARS = [2005]  #test run
LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 50]   # hPa
COARSEN = 8                                   # 0.25° → 2° (we regrid to T31 ≈ 3.75° later)

# Paste the 1995 link for each variable (keep the quotes)
TEMPLATES = {
    "tas":        "https://osdf-director.osg-htc.org/ncar/gdex/d633001/e5.moda.an.sfc/2005/e5.moda.an.sfc.128_167_2t.ll025sc.2005010100_2005120100.nc",
    "pr":         "https://osdf-director.osg-htc.org/ncar/gdex/d633001/e5.moda.fc.sfc.meanflux/2005/e5.moda.fc.sfc.meanflux.235_055_mtpr.ll025sc.2005010100_2005120100.nc",
    "toa_net_sw": "https://osdf-director.osg-htc.org/ncar/gdex/d633001/e5.moda.fc.sfc.meanflux/2005/e5.moda.fc.sfc.meanflux.235_039_mtnswrf.ll025sc.2005010100_2005120100.nc",
    "toa_net_lw": "https://osdf-director.osg-htc.org/ncar/gdex/d633001/e5.moda.fc.sfc.meanflux/2005/e5.moda.fc.sfc.meanflux.235_040_mtnlwrf.ll025sc.2005010100_2005120100.nc",
    "toa_in_sw":  "https://osdf-director.osg-htc.org/ncar/gdex/d633001/e5.moda.fc.sfc.accumu/2005/e5.moda.fc.sfc.accumu.128_212_tisr.ll025sc.2005010100_2005120100.nc",
    "ta":         "https://osdf-director.osg-htc.org/ncar/gdex/d633001/e5.moda.an.pl/2005/e5.moda.an.pl.128_130_t.ll025sc.2005010100_2005120100.nc",
    "ua":         "https://osdf-director.osg-htc.org/ncar/gdex/d633001/e5.moda.an.pl/2005/e5.moda.an.pl.128_131_u.ll025uv.2005010100_2005120100.nc",
    "va":         "https://osdf-director.osg-htc.org/ncar/gdex/d633001/e5.moda.an.pl/2005/e5.moda.an.pl.128_132_v.ll025uv.2005010100_2005120100.nc",
}
PRESSURE = {"ta", "ua", "va"}

def main_var(ds):
    """Each ERA5 file holds one variable; pick it regardless of its exact name."""
    cands = [v for v in ds.data_vars if {"latitude", "longitude"} <= set(ds[v].dims)]
    assert len(cands) == 1, f"expected one data variable, found {cands}"
    return ds[cands[0]]

for name, tmpl in TEMPLATES.items():
    for y in YEARS:
        out = f"{OUT}/{name}_{y}.nc"
        if os.path.exists(out):               # already done → skip (safe to rerun)
            continue
        url = tmpl.replace("2005", str(y))
        raw = f"{TMP}/{os.path.basename(url)}"
        subprocess.run(["wget", "-q", "-c", "-O", raw, url], check=True)

        ds = xr.open_dataset(raw)
        da = main_var(ds)
        if name in PRESSURE:
            da = da.sel(level=LEVELS)          # keep 12 of 37 levels
        da = da.coarsen(latitude=COARSEN, longitude=COARSEN, boundary="trim").mean()
        da.load().to_netcdf(out)
        ds.close()
        os.remove(raw)                         # delete the big original
        print(f"{name} {y} done", flush=True)