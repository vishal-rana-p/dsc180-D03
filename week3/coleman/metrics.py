import numpy as np


def global_mean(da):
    """cos(latitude)-weighted mean over lat/lon."""
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(["lat", "lon"])

def nrmse_scores(truth, pred, start=2080, end=2100):
    """ClimateBench NRMSE. truth = ssp245 ensemble mean; same units/coords as pred."""
    t = truth.sel(time=slice(start, end))
    p = pred.sel(time=slice(start, end))
    assert t.sizes["time"] == p.sizes["time"] == end - start + 1, "time window mismatch"
    assert np.allclose(t.lat, p.lat) and np.allclose(t.lon, p.lon), "lat/lon grids differ"
    norm = abs(float(global_mean(t.mean("time"))))   # |global mean of truth| over the window
    spatial = float(np.sqrt(global_mean((t.mean("time") - p.mean("time")) ** 2))) / norm
    glob    = float(np.sqrt(((global_mean(t) - global_mean(p)) ** 2).mean("time"))) / norm
    return {"spatial": spatial, "global": glob, "total": spatial + 5 * glob}
