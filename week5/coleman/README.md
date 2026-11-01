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

Raw control output (1.7 GB, 30-day means of every variable) is on Coleman's machine at `~/runs/earth-qflux-v2/`; ask if you need something that isn't in the summary.

## Scripts
- `make_qflux.py`: derives the Q-flux from a 10-year relaxed-slab spin-up.
- `check_drift.py`: drift of the control against the week-4 SST. Writes `drift_yearly.csv`, `drift_timeseries.png` and `drift_map.png`.
- `run_abrupt_co2.py`: abrupt N×CO₂ run branched from the control (for example `python run_abrupt_co2.py 2 --years 40`). It was tested for 30 days: CO₂ was doubled, and the run started from the control state.

The paths at the top of each script point to `/home/colecam/...`; change them for your machine. You need JEM installed (`pip install -e jax-esm`) next to your jax-gcm checkout.

## Pitfalls we hit (please read)
1. **Don't branch with `coupled_run.initial_condition=` on the command line.** JEM restores every value in the checkpoint, including the atmosphere's CO₂, so a 2×CO₂ forcing file is silently ignored and the run uses 420 ppmv. `run_abrupt_co2.py` loads the checkpoint and swaps in the new CO₂ value itself. To confirm which CO₂ a run used, check `mod_radcon.ablco2` in the `atm` output: 6.0 at 420 ppmv, 12.0 at 2×, 24.0 at 4×.
2. **The Q-flux is also stored in checkpoints.** Rerunning into an existing output directory resumes from its checkpoint, so a changed setting can be silently ignored. Use a fresh output directory for each experiment.
3. **The Q-flux from a relaxed slab must include ρc_p·h·dT_clim/dt**, because JEM's relaxation follows the climatology's seasonal cycle. Without that term, shallow seas drifted 20 K too cold (`make_qflux.py` includes it).
4. **The control's CO₂ is 420 ppmv**, jcm's default, so "2×" means 840 ppmv and "4×" means 1680 ppmv.
5. **Model output uses 30-day windows, not calendar months.** A run of `--years N` covers N × 360 days, so model "years" drift slightly against the calendar. Average by `time.year` for annual means, as `check_drift.py` does.
