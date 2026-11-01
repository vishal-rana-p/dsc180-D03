"""Steps 1-3 for the abrupt 2x/4xCO2 slab runs: time series, Gregory regression, ECS.

Anomalies are taken against the EQUILIBRATED control: the mean of control years
2010-2029 (data/control_summary.nc). The abrupt runs branch from the control's
end state but restart their clock at 2000, so subtracting control year 2000
(still spinning up) would be wrong. The control's TOA net is +6.2 W m-2 at
equilibrium -- the atmosphere's own imbalance, carried away by the Q-flux -- so
dN is measured from that, not from zero.

Run (jcm env), after run_abrupt_co2.py has written one or both runs:
    python analyze_abrupt_co2.py
Writes abrupt_timeseries.png, gregory.png and abrupt_results.csv next to this
script. Whichever of the two runs exists is analysed.
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

RUNS = Path("/home/colecam/runs")
HERE = Path(__file__).parent
CONTROL = xr.open_dataset(HERE / "data" / "control_summary.nc")
BASE_YEARS = slice(2010, 2029)      # equilibrated control
EQ_YEARS = 10                       # "equilibrium" = mean of each run's last 10 years

# Week-4 fixed-SST ERF (W m-2), for comparison with the Gregory intercept.
# 2x: Coleman's ~/DSC180A/runs/co2x2 minus runs/control, 16 years year-matched
#     (7.09 +/- 0.04; same packaged T30 boundary files as this control).
#     Vishal's speedy-t31 runs (week4/vishalrana/w4_task6-2.ipynb, cell 13) give 7.10 +/- 0.05.
# 4x: Vishal's run only, 10.39 +/- 0.06.
WEEK4_ERF = {2: 7.09, 4: 10.39}

NORESM2_ECS = 2.5                   # Seland et al. (2020)
AR6 = {"best": 3.0, "likely": (2.5, 4.0), "very likely": (2.0, 5.0)}

VARS = ["surface_flux.t0", "shortwave_rad.ftop", "longwave_rad.ftop"]


def annual_global(run_dir):
    """Annual global means of near-surface T (K) and TOA net (W m-2, down)."""
    files = sorted(glob.glob(f"{run_dir}/atm-*.nc"))
    atm = xr.open_mfdataset(files, preprocess=lambda d: d[VARS]).load()
    w = np.cos(np.deg2rad(atm.lat))
    ds = xr.Dataset({
        "tas": atm["surface_flux.t0"],
        "toa_net": atm["shortwave_rad.ftop"] - atm["longwave_rad.ftop"],
    }).weighted(w).mean(["lon", "lat"])
    yearly = ds.groupby("time.year").mean()
    counts = ds.time.groupby("time.year").count()
    return yearly.sel(year=counts.year[counts >= 12])      # complete years only


base_tas = float(CONTROL.tas_global_annual.sel(year=BASE_YEARS).mean())
base_toa = float(CONTROL.toa_net_global_annual.sel(year=BASE_YEARS).mean())
print(f"control baseline ({BASE_YEARS.start}-{BASE_YEARS.stop}): "
      f"T = {base_tas:.2f} K, N = {base_toa:+.2f} W m-2")

runs = {}
for factor in (2, 4):
    run_dir = RUNS / f"abrupt-{factor}xco2"
    if not list(run_dir.glob("atm-*.nc")):
        print(f"{factor}xCO2: no output in {run_dir}, skipped")
        continue
    yearly = annual_global(run_dir)
    runs[factor] = pd.DataFrame({
        "dT": yearly.tas.values - base_tas,
        "dN": yearly.toa_net.values - base_toa,
    }, index=np.arange(1, yearly.sizes["year"] + 1))      # years since the CO2 jump

if not runs:
    raise SystemExit("no abrupt runs to analyse yet")

# --- Step 2: Gregory regression, and Step 3: ECS --------------------------------
rows = []
for factor, df in runs.items():
    slope, intercept = np.polyfit(df.dT, df.dN, 1)
    last = df.tail(EQ_YEARS)
    trend = np.polyfit(last.index, last.dT, 1)[0] * 10
    rows.append({
        "run": f"{factor}xCO2",
        "years": len(df),
        "ERF_intercept_Wm2": intercept,
        "lambda_Wm2K": slope,
        "dT_eq_regression_K": -intercept / slope,
        f"dT_last{EQ_YEARS}yr_K": last.dT.mean(),
        f"dT_trend_last{EQ_YEARS}yr_K_per_decade": trend,
        f"dN_last{EQ_YEARS}yr_Wm2": last.dN.mean(),
        "week4_fixedSST_ERF_Wm2": WEEK4_ERF[factor],
    })
results = pd.DataFrame(rows).set_index("run")
results.round(4).to_csv(HERE / "abrupt_results.csv")

print("\nGregory regression (annual means, dN on dT):")
for run, r in results.iterrows():
    print(f"  {run}: ERF = {r.ERF_intercept_Wm2:.2f} W m-2, lambda = {r.lambda_Wm2K:.2f} W m-2 K-1, "
          f"regression dT_eq = {r.dT_eq_regression_K:.2f} K")
    print(f"         last {EQ_YEARS} yr: dT = {r[f'dT_last{EQ_YEARS}yr_K']:.2f} K "
          f"(trend {r[f'dT_trend_last{EQ_YEARS}yr_K_per_decade']:+.3f} K/decade), "
          f"dN = {r[f'dN_last{EQ_YEARS}yr_Wm2']:+.2f} W m-2")
    if r.week4_fixedSST_ERF_Wm2 is not None and not pd.isna(r.week4_fixedSST_ERF_Wm2):
        print(f"         week-4 fixed-SST ERF = {r.week4_fixedSST_ERF_Wm2:.2f} W m-2 "
              f"(intercept - fixed-SST = {r.ERF_intercept_Wm2 - r.week4_fixedSST_ERF_Wm2:+.2f})")

print("\nClimate sensitivity:")
if 2 in runs:
    r2 = results.loc["2xCO2"]
    print(f"  ECS, equilibrium dT of 2xCO2 (last {EQ_YEARS} yr): {r2[f'dT_last{EQ_YEARS}yr_K']:.2f} K")
    print(f"  ECS, -F2x/lambda from 2xCO2 regression:      {r2.dT_eq_regression_K:.2f} K")
    if abs(r2[f"dT_trend_last{EQ_YEARS}yr_K_per_decade"]) > 0.05:
        print("  ! 2xCO2 is still warming: extend it (same command, larger --years) before quoting ECS")
if 4 in runs:
    r4 = results.loc["4xCO2"]
    print(f"  F2x from 4xCO2 intercept / 2: {r4.ERF_intercept_Wm2 / 2:.2f} W m-2;  "
          f"4xCO2 regression dT_eq / 2: {r4.dT_eq_regression_K / 2:.2f} K")
print(f"  NorESM2-LM: ~{NORESM2_ECS} K.  IPCC AR6: best {AR6['best']} K, "
      f"likely {AR6['likely'][0]}-{AR6['likely'][1]} K, very likely {AR6['very likely'][0]}-{AR6['very likely'][1]} K")

# --- Figures ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for i, (factor, df) in enumerate(runs.items()):
    axes[0].plot(df.index, df.dT, marker="o", ms=3, color=f"C{i}", label=f"{factor}xCO$_2$")
    axes[1].plot(df.index, df.dN, marker="o", ms=3, color=f"C{i}")
axes[0].set_ylabel("$\\Delta T$ near-surface (K)")
axes[0].legend()
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_ylabel("$\\Delta N$ TOA net (W m$^{-2}$)")
axes[1].set_xlabel("years since CO$_2$ increase")
fig.suptitle("Abrupt CO$_2$ slab runs: anomalies from the Q-flux control")
fig.tight_layout()
fig.savefig(HERE / "abrupt_timeseries.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 5))
for i, (factor, df) in enumerate(runs.items()):
    r = results.loc[f"{factor}xCO2"]
    ax.scatter(df.dT, df.dN, s=12, color=f"C{i}", label=f"{factor}xCO$_2$")
    # Extend the fit to its dN = 0 crossing only when that is a plausible
    # temperature; a near-zero or positive slope puts it at +/- thousands of K.
    x_end = df.dT.max()
    if 0 < r.dT_eq_regression_K < 3 * df.dT.max():
        x_end = r.dT_eq_regression_K
    x = np.array([0, x_end])
    ax.plot(x, r.ERF_intercept_Wm2 + r.lambda_Wm2K * x, color=f"C{i}",
            label=f"F = {r.ERF_intercept_Wm2:.2f}, $\\lambda$ = {r.lambda_Wm2K:.2f}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("$\\Delta T$ (K)")
ax.set_ylabel("$\\Delta N$ (W m$^{-2}$)")
ax.set_title("Gregory regression (annual means)")
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "gregory.png", dpi=150)
print(f"\nwrote abrupt_timeseries.png, gregory.png, abrupt_results.csv in {HERE}")
