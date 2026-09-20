# Predicting lifetime 100 m personal bests from junior competition trajectories (CAA cohort)

Feature-engineering and modelling code for a study of whether junior sprint
performance trajectories predict lifetime 100 m personal bests, using
competition records published by the Chinese Athletics Association (CAA).

This repository contains the complete analysis pipeline, the configuration that
fixes every random seed and hyperparameter, the aggregate result tables reported
in the paper, and a de-identified sample dataset that lets anyone execute the
modelling code without access to personal data.

## Citation

Huang W, Zhou H. Sprint performance prediction in a national-level database of 58,000
athletes: developmental patterns, population composition effects, and the role of
anthropometric features. BMC Sports Sci Med Rehabil. 2026.
https://doi.org/10.1186/s13102-026-02099-5

This repository contains the analysis code and a de-identified demonstration extract
for that article. Released under the MIT License.

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
| Male | Ridge | R² = 0.790 (n = 463) | R² = 0.803 (n = 2,662) |
| Male | RandomForest | R² = 0.777 | R² = 0.796 |
| Male | GradientBoosting | R² = 0.772 | R² = 0.799 |
| Female | Ridge | R² = 0.760 (n = 356) | R² = 0.780 (n = 445) |
| Female | RandomForest | R² = 0.743 | R² = 0.754 |
| Female | GradientBoosting | R² = 0.727 | R² = 0.738 |

The sample reproduces the substantive result but not the exact point estimates:
it contains a fraction of the cohort, so estimates are noisier.

## What is in this repository

| Path | Contents |
|---|---|
| `src/s01`–`s15` | Full pipeline: parsing, cleaning, province mapping, wind correction, feature engineering, modelling, robustness, figures |
| `s16_sensitivity_dk.py`, `redraw_figures_*.py` | Revision-stage sensitivity analyses and figure scripts, run from the project root |
| `r4_reanalysis/` | Analyses added at peer review, plus `gate_reproduce.py`, which checks that an installation reproduces the published estimates (see Environment) |
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

The analyses added at peer review live in `r4_reanalysis/` and are also run from
the project root. They need the full processed dataset and exit with an
explanatory message without it, as the two scripts described above do. Run the
gate first: it refits one published cell and compares it with
`results/tables/table2_group_a_performance.csv`, so a failure means the
installation does not reproduce the paper and nothing downstream would be
comparable with the published tables.

```bash
python r4_reanalysis/gate_reproduce.py --full        # must print GATE PASSED
python r4_reanalysis/r4_02_carry_forward.py          # carry-forward baseline
python r4_reanalysis/r4_03_overlap_split.py          # predictor-outcome identity removed
python r4_reanalysis/r4_05_attrition_censoring.py    # attrition under follow-up requirements
python r4_reanalysis/r4_06_paired_delta_r2.py        # paired CIs for R² differences
python r4_reanalysis/r4_78_sensitivity.py --mode hurdles
python r4_reanalysis/r4_78_sensitivity.py --mode wind
```

Each of these recomputes the published baseline in the same run and checks it
against the published table before reporting any new estimate.

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

**Group A versus all athletes.** Group A comprises athletes with at least one
100 m record *after* the cutoff age, that is those whose competitive career
continued past it (`src/s10_group_split.py`). Group B athletes have no record
after the cutoff, so their lifetime best is already fixed inside the pre-cutoff
window and is near-trivially predictable from the pre-cutoff features, which
inflates R². Athletes with fewer than two pre-cutoff records are excluded from
both. The paper's primary analysis is restricted to Group A;
`scripts/demo_reproduce.py --group all` shows the inflated estimate for
comparison.

Career continuation does not by itself guarantee that the lifetime best is set
after the cutoff. For 22.7% to 46.2% of Group A, depending on sex and cutoff
age, the lifetime best is one of the pre-cutoff races, so the target equals the
feature `best_time_raw` exactly and the model is evaluating an identity rather
than predicting. `r4_reanalysis/r4_03_overlap_split.py` quantifies that share and
refits the published specification with those athletes removed.

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

**Verifying an installation against the published estimates.** The table above is
computed on the demo sample, so it demonstrates version stability rather than
agreement with the paper. `r4_reanalysis/gate_reproduce.py` makes the stronger
check available to anyone with access to the full processed dataset: it rebuilds
the Group A feature matrix for cutoff age 16 (male, `traj_wind`) from scratch and
compares the resulting R², both confidence limits and RMSE with the values stored
in `results/tables/table2_group_a_performance.csv`.

In a third environment, Python 3.11.15 with pandas 3.0.2, NumPy 2.4.4,
SciPy 1.17.1, scikit-learn 1.8.0, statsmodels 0.14.6 and joblib 1.5.3, the gate
reproduced Ridge to within 6e-14 and GradientBoosting to within 1.2e-08, with the
confidence limits identical. Both are floating-point noise.

The only difference anywhere in the two outputs was the upper bound of one
percentile interval across cross-validation folds (female RandomForest: 0.763 in
the reference environment, 0.762 in the clean environment). That reflects
floating-point summation order across library versions and is more than two
orders of magnitude smaller than the width of the interval itself.

## License

MIT — see [LICENSE](LICENSE).
