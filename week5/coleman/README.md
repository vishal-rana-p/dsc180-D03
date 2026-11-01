# Week 5: Q-flux slab-ocean control (Coleman)

A 30-year coupled control in JEM (`climate-analytics-lab/jax-esm`): the same SPEEDY T31 atmosphere and T30 boundary files as week 4, but with a slab ocean held near the week-4 SST by a Q-flux. It is the baseline for the abrupt 2×/4×CO₂ runs.

## Result
- **Equilibrated.** After about 4 years of spin-up, the SST settles about 0.2 K below the week-4 climatology. Over 2010–2029 the trend is −0.006 ± 0.012 K/decade (no drift).
- **TOA net is +6.2 W/m², steady.** Your week-4 fixed-SST control has the same value; it is the atmosphere model's own imbalance, which the Q-flux removes. So measure ΔN against the control's 6.2, not against 0.
- **Regional SST bias** is mostly within ±1 K. It is largest (1–2 K) in western boundary currents and upwelling zones (`drift_map.png`).

## Data (`data/`)
| File | What it is |
|---|---|
| `control_summary.nc` | **Baseline for Δ calculations.** Global annual means of `tas` (K), `toa_net` (W/m², positive down) and `pr` (mm/day) for 2000–2029; monthly maps of `tas`, `toa_net`, `pr` and `sst` averaged over the last 10 years (2020–2029). |
| `control_checkpoint/` | The control's final model state. Start the abrupt-CO₂ runs from this. |
| `qflux_t30.nc` | The monthly Q-flux used by the control. The CO₂ runs must use the same file. |

| `abrupt_summary.nc` | **Steps 4–6.** For `control` (years 2010–2029), `2xCO2` (40 yr, 840 ppmv) and `4xCO2` (60 yr, 1680 ppmv): global annual means of `tas`, `toa_net`, `sfc_net` (surface net heat flux, positive into the surface) and `pr`, indexed by `year_<run>` = years since the CO₂ jump; and monthly maps (`*_eqclim_<run>`) of `tas`, `pr`, `toa_net` and `sst` averaged over each run's last 10 years. Equilibrium ΔT/ΔP maps = annual mean of the run's map minus the control's. |
| `abrupt_annual.csv` | The same global annual means as a table (`run, year, tas, toa_net, sfc_net, pr`). |

Raw abrupt-run output (2.4 and 3.6 GB) is at `~/runs/abrupt-2xco2/` and `~/runs/abrupt-4xco2/` on Coleman's machine. Raw control output (1.7 GB, 30-day means of every variable) is on Coleman's machine at `~/runs/earth-qflux-v2/`; ask if you need something that isn't in the summary.

## Scripts
- `make_qflux.py`: derives the Q-flux from a 10-year relaxed-slab spin-up.
- `check_drift.py`: drift of the control against the week-4 SST. Writes `drift_yearly.csv`, `drift_timeseries.png` and `drift_map.png`.
- `run_abrupt_co2.py`: abrupt N×CO₂ run branched from the control (for example `python run_abrupt_co2.py 2 --years 40`). It was tested for 30 days: CO₂ was doubled, and the run started from the control state.
- `analyze_abrupt_co2.py`: Steps 1–3 once the abrupt runs exist. It plots ΔT and ΔN against time, runs the Gregory regression (ERF, λ) and computes ECS both ways. Anomalies are taken against the control's equilibrated 2010–2029 mean, not its year 2000, because the abrupt runs restart their clock at 2000. Writes `abrupt_timeseries.png`, `gregory.png` and `abrupt_results.csv`. `WEEK4_ERF` at the top holds the week-4 fixed-SST ERF (2×: 7.09 W/m², from Coleman's week-4 runs; 4×: 10.39 W/m², from Vishal's `w4_task6-2.ipynb`).
- `make_abrupt_summary.py`: condenses the raw abrupt-run output into `data/abrupt_summary.nc` and `data/abrupt_annual.csv`.

## Abrupt CO₂ runs (step 3): results and a caveat you need for steps 4–6
Both runs finished: 2×CO₂ for 40 years and 4×CO₂ for 60 years, each branched from the control's end state. Anomalies are against the control's 2010–2029 mean.

| | ΔT, years 1 / 3 / last 10 | ΔN TOA, last 10 yr | ΔT land / ocean, last 5 yr |
|---|---|---|---|
| 2×CO₂ | 0.69 / 1.73 / **2.46 K** | +6.5 W/m² | 2.16 / 2.58 K |
| 4×CO₂ | 1.03 / 2.64 / **3.73 K** | +9.8 W/m² | 3.46 / 3.86 K |

ΔT levels off after about 6 years and then stays flat, with no trend. **ΔN does not decay, though.** It stays at roughly the forcing (+6 to +6.5 W/m² at 2×) from year 1 to year 40. So the Gregory regression is degenerate (λ ≈ 0, and −F/λ is meaningless).

**Why:** `earth-slab` relaxes the land temperature toward its present-day climatology with `tdland = 1 day`. This is a stopgap noted in `jem/config/configuration/earth-slab.yaml` (jax-esm#1). The land therefore acts as an infinite heat sink. It cannot warm, so at "equilibrium" it keeps absorbing about 14 W/m² more than in the control. The surface net heat flux (`sfc_net`, positive into the surface) carries the TOA anomaly straight down: its anomaly is about +6 W/m², nearly equal to ΔN. Where it goes:
- **Land:** about 4 W/m² global-equivalent, from 14 W/m² over roughly 30% of the area.
- **Ocean:** about 2 W/m². SST is flat, so this is a smaller ocean or sea-ice sink that we have not tracked down.

What this means for the next steps:
- **Gregory / λ:** regressing TOA ΔN on ΔT does not give a feedback parameter here, because energy leaks into the land. Report that, and show the time series. One option is to regress over just the first few years, but the slope will still be wrong in sign or size.
- **ECS:** the equilibrium ΔT (2.46 K at 2×) is a true steady state of this setup. It is an *underestimate* of SPEEDY's own ECS, because the land isn't allowed to warm. (It happens to land close to NorESM2-LM's ~2.5 K.)
- **Warming patterns:** land warms *less* than ocean here. In NorESM2 and in observations, land warms about 1.5× more. Expect disagreement over continents: that mismatch is an artifact of this setup, not physics.
- **The real fix:** rerun with a longer `tdland`, e.g. SPEEDY's 40 days, the jem default. The config comment warns that the land then runs too hot and too cold at T30, so the control would also have to be redone. We did not do this.

The paths at the top of each script point to `/home/colecam/...`; change them for your machine. You need JEM installed (`pip install -e jax-esm`) next to your jax-gcm checkout.

## Pitfalls we hit (please read)
1. **Don't branch with `coupled_run.initial_condition=` on the command line.** JEM restores every value in the checkpoint, including the atmosphere's CO₂, so a 2×CO₂ forcing file is silently ignored and the run uses 420 ppmv. `run_abrupt_co2.py` loads the checkpoint and swaps in the new CO₂ value itself. To confirm which CO₂ a run used, check `mod_radcon.ablco2` in the `atm` output: 6.0 at 420 ppmv, 12.0 at 2×, 24.0 at 4×.
2. **The Q-flux is also stored in checkpoints.** Rerunning into an existing output directory resumes from its checkpoint, so a changed setting can be silently ignored. Use a fresh output directory for each experiment.
3. **The Q-flux from a relaxed slab must include ρc_p·h·dT_clim/dt**, because JEM's relaxation follows the climatology's seasonal cycle. Without that term, shallow seas drifted 20 K too cold (`make_qflux.py` includes it).
4. **The control's CO₂ is 420 ppmv**, jcm's default, so "2×" means 840 ppmv and "4×" means 1680 ppmv.
5. **Model output uses 30-day windows, not calendar months.** A run of `--years N` covers N × 360 days, so model "years" drift slightly against the calendar. Average by `time.year` for annual means, as `check_drift.py` does.
