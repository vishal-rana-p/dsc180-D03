import glob, re
import numpy as np
import xarray as xr

# ---- settings ----
FILES = sorted(glob.glob("runs/control/year*.nc"))
SW_IN = "shortwave_rad.fsol"     # incoming SW at TOA
SW_NET = "shortwave_rad.ftop"    # net downward SW at TOA
OLR = "longwave_rad.ftop"        # outgoing LW at TOA, positive upward
SAVES_PER_YEAR = 73              # 365 days / 5-day saves
SPINUP_YEARS = 1
TARGET = 0.1                     # W/m2, target standard error for ERF
N_CO2_PLANNED = 40               # years per CO2 run after spin-up (run_co2.py)
# ------------------

# Seconds per simulated year, read from control.log (skips year 0, which includes compilation)
times = [float(t) for t in re.findall(r"year (\d+) done in ([\d.]+) s", open("control.log").read())
         for t in [t[1]]][1:]
SEC_PER_SIM_YEAR = float(np.median(times)) if times else 30.0
print(f"files: {len(FILES)} | seconds per simulated year: {SEC_PER_SIM_YEAR:.1f}")

ds = xr.open_mfdataset(FILES, combine="by_coords")
w = np.cos(np.deg2rad(ds.lat))

def gm(da):
    return da.weighted(w).mean(("lat", "lon"))

def yearly(da):
    y = gm(da).coarsen(time=SAVES_PER_YEAR, boundary="trim").mean()
    return y.isel(time=slice(SPINUP_YEARS, None)).compute().values

sw_in = yearly(ds[SW_IN])
sw_net = yearly(ds[SW_NET])
olr = yearly(ds[OLR])
net = sw_net - olr               # positive downward
m = len(net)

# TOA budget (sanity check, also useful for step 3)
print("\nGlobal-mean TOA budget (after spin-up):")
print(f"  incoming SW : {sw_in.mean():7.2f} W/m2")
print(f"  reflected SW: {(sw_in - sw_net).mean():7.2f} W/m2  (albedo {(1 - sw_net / sw_in).mean():.3f})")
print(f"  OLR         : {olr.mean():7.2f} W/m2")
print(f"  net (down)  : {net.mean():7.2f} W/m2")

# Variability
sigma = net.std(ddof=1)
r1 = np.corrcoef(net[:-1], net[1:])[0, 1] if m > 2 else np.nan
print(f"\ncontrol years used (m): {m}")
print(f"sigma (interannual SD of net): {sigma:.3f} W/m2")
print(f"lag-1 autocorrelation: {r1:.2f}")
print(f"SE of control mean: {sigma / np.sqrt(m):.3f} W/m2")

# Does the current plan meet the target?
se_plan = sigma * np.sqrt(1 / N_CO2_PLANNED + 1 / m)
print(f"\nPlanned: {N_CO2_PLANNED}-yr CO2 runs vs {m}-yr control -> SE(ERF) ≈ {se_plan:.3f} W/m2 "
      f"({'meets' if se_plan < TARGET else 'does NOT meet'} the {TARGET} target)")

# Run-length options (with a 1.3x safety margin on sigma)
sigma_safe = 1.3 * sigma
print("\ncontrol m | CO2 years n needed | extra GPU time (4 runs + control extension)")
for m_try in sorted({m, 20, 30, 40, 50, 80}):
    room = TARGET**2 / sigma_safe**2 - 1 / m_try
    if room <= 0:
        print(f"{m_try:9d} | impossible (control too short)")
        continue
    n = int(np.ceil(1 / room))
    extra = max(0, m_try - m)
    hours = (4 * (n + SPINUP_YEARS) + extra) * SEC_PER_SIM_YEAR / 3600
    print(f"{m_try:9d} | {n:18d} | {hours:.1f} h")

# Save yearly values for later steps
np.savetxt("runs/control/yearly_toa_net.txt", net, fmt="%.4f",
           header="global-mean TOA net (W/m2, positive down), one value per year after spin-up")