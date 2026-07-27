# Predicting lifetime 100 m personal bests from junior competition trajectories (CAA cohort)

<!-- After creating the Zenodo release, replace the line below with the DOI badge Zenodo gives you. -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Feature-engineering and modelling code for a study of whether junior sprint
performance trajectories predict lifetime 100 m personal bests, using
competition records published by the Chinese Athletics Association (CAA).

This repository contains the complete analysis pipeline, the configuration that
fixes every random seed and hyperparameter, the aggregate result tables reported
in the paper, and a de-identified sample dataset that lets anyone execute the
modelling code without access to personal data.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements-lock.txt, see Environment

# Run the modelling code on the public de-identified sample
python scripts/demo_reproduce.py
```

Expected output (Group A, cutoff age 18, `traj_wind` feature set), next to the
values reported in the paper, which were computed on the full cohort:

| Sex | Model | Sample (this repo) | Published (full cohort) |
|---|---|---|---|
| Male | Ridge | R² = 0.793 (n = 475) | R² = 0.803 (n = 2,662) |
| Male | RandomForest | R² = 0.780 | R² = 0.796 |
| Male | GradientBoosting | R² = 0.785 | R² = 0.799 |
| Female | RandomForest | R² = 0.629 (n = 369) | R² = 0.624 (n = 522) |
| Female | GradientBoosting | R² = 0.616 | R² = 0.644 |

The sample reproduces the substantive result but not the exact point estimates:
it contains a fraction of the cohort, so estimates are noisier. Ridge is
unstable for females in both the sample and the full cohort.

## What is in this repository

| Path | Contents |
|---|---|
| `src/s01`–`s15` | Full pipeline: parsing, cleaning, province mapping, wind correction, feature engineering, modelling, robustness, figures |
| `s16_sensitivity_dk.py`, `redraw_figures_*.py` | Revision-stage sensitivity analyses and figure scripts, run from the project root |
| `src/utils/` | `features.py` (feature definitions), `validation.py` (CV strategies), `io.py` |
| `config/config.yaml` | Every tunable parameter, including `random_seed: 42`, cutoff ages, model hyperparameters, era splits |
| `config/feature_registry.yaml` | Feature groups and dataset variants |
| `data/sample/` | De-identified sample dataset (see below) |
| `results/tables/` | Aggregate result tables underlying the reported numbers |
| `scripts/demo_reproduce.py` | Runs the modelling code on the sample |
| `scripts/make_public_sample.py` | Builds `data/sample/` from the internal tables |

## Data

**Raw data.** Competition records are published by the Chinese Athletics
Association at <https://www.athletics.org.cn>. The raw scrape contains athlete
names and full dates of birth and is therefore **not** included here.

**Sample data.** `data/sample/` holds a de-identified extract:

- `athlete_id` is replaced by a salted SHA-256 digest; the salt is secret and
  not distributed, so the identifiers cannot be reversed.
- Names, dates of birth and team strings are absent. Only birth cohort
  (5-year band) and competition year are retained.
- `province` is replaced by an arbitrary group code. The province-disjoint
  cross-validation needs a grouping variable, not the province name.
- Ages are given in whole years, so a date of birth cannot be reconstructed
  from an age together with a competition date. All other floating-point
  columns are rounded to four decimals.
- The extract is a **purposive demo sample, not a random sample**: Group A
  athletes are over-sampled so the primary analysis can be run end to end. It
  must not be used for inference about the CAA population.
- `data/sample/MANIFEST.json` records the sampling parameters and the
  de-identification rules.
- **Condition of use:** no attempt may be made to re-identify any individual in
  this extract, or to link it to any other dataset for that purpose.

**Full processed dataset.** Contains personal information and is available
from the corresponding author on reasonable request under a data-use agreement.

## Reproducing the full analysis

With access to the raw records placed in `data/raw/`:

```bash
python src/run_pipeline.py          # steps 1–9, end to end
python src/run_pipeline.py --step 6 # a single step
python src/run_pipeline.py --quick  # 20 CV repeats instead of 500, for smoke tests
```

Steps 10–15 (group split, controlled Group A analysis, anthropometrics,
attrition, composition effects, figures) and the revision-stage sensitivity
analyses are run directly from the project root:

```bash
python src/s10_group_split.py           # Group A/B assignment at each cutoff
python src/s11_controlled_analysis.py   # primary Group A models
python s16_sensitivity_dk.py            # membership / follow-up / R² definition sensitivity
python redraw_figures_2_3.py            # figures 2–3
python redraw_figures_4_5.py            # figures 4–5
```

The two figure scripts resolve their paths relative to the script file and
write to `results/figures/`. `redraw_figures_2_3.py` runs against
`results/tables/composition_decomposition.csv`, which is included here.
`redraw_figures_4_5.py` reads `results/raw_results/controlled_analysis/`,
which is not distributed (see `.gitignore`); it is included as provenance for
those two figures and exits with an explanatory message until the pipeline has
been run. `src/s14_composition_effect.py` likewise reads
`data/external/wa_results_summary.csv`, the World Athletics reference series
from a separate analysis by the same authors, which is not distributed here;
the values it contributes are the `wa_*` columns of
`results/tables/composition_decomposition.csv`, which is included.

All randomness is controlled by `modeling.cv.random_seed` in
`config/config.yaml` (default 42) and by the `random_state` of each estimator.

## Analysis notes

**Cross-validation is athlete-disjoint.** The modelling matrix has one row per
athlete: features are aggregated from that athlete's pre-cutoff records and the
target is the lifetime personal best. A random K-fold split over rows is
therefore a split over athletes, and no athlete contributes to both a training
and a test fold.

**Group A versus all athletes.** Group A comprises athletes who had not yet
achieved their lifetime personal best at the cutoff age. Group B athletes
already had, which makes their target near-trivially predictable from the
pre-cutoff features and inflates R². The paper's primary analysis is restricted
to Group A; `scripts/demo_reproduce.py --group all` shows the inflated estimate
for comparison.

**No leakage across the cutoff.** At cutoff age X only records with
`age_at_comp <= X` enter feature computation, while the target is the lifetime
best over all records.

## Environment

Python 3.10 or newer is required. Two dependency files are provided:

- `requirements.txt` — lower bounds only (scikit-learn ≥ 1.2, pandas ≥ 1.5,
  SciPy ≥ 1.9). Use this for ordinary use; it will not go stale.
- `requirements-lock.txt` — a version-pinned reference environment
  (Python 3.13.3, pandas 2.3.3, NumPy 2.2.6, SciPy 1.16.3,
  scikit-learn 1.8.0).

**Results are stable across dependency versions.** The demo was run twice on the
same machine: once in the version-pinned reference environment above, and once in
a clean virtual environment created from `requirements.txt`, which resolved to
pandas 3.0.5, NumPy 2.5.1, SciPy 1.18.0 and scikit-learn 1.9.0, crossing a major
version boundary in pandas. Every R² and every RMSE was identical to three
decimal places:

| Sex | Model | Reference env | Clean env (latest) |
|---|---|---|---|
| Male | Ridge | 0.793 | 0.793 |
| Male | RandomForest | 0.780 | 0.780 |
| Male | GradientBoosting | 0.785 | 0.785 |
| Female | RandomForest | 0.629 | 0.629 |
| Female | GradientBoosting | 0.616 | 0.616 |

The only difference anywhere in the two outputs was the upper bound of one
percentile interval across cross-validation folds (female RandomForest: 0.763 in
the reference environment, 0.762 in the clean environment). That reflects
floating-point summation order across library versions and is more than two
orders of magnitude smaller than the width of the interval itself.

## Citation

If you use this code, please cite the paper and the archived release. Once the
Zenodo DOI is issued, replace the placeholder in the badge above.

## License

MIT — see [LICENSE](LICENSE).
