"""
Step 14: Assemble population composition decomposition table.

Merges diagnostic results (diag1–3) with WA reference into a single
unified table for Fig 2 and the composition analysis narrative.

Input:  results/diagnostic/diag1_career_overlap.csv
        results/diagnostic/diag2_pb_variance.csv
        results/diagnostic/diag3_stratified_r2.csv
        data/external/wa_results_summary.csv
          (World Athletics reference results; excluded from the public
           repository by .gitignore, see README)
Output: results/tables/composition_decomposition.csv
"""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_config, ensure_dirs

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    diag_dir = Path(config["paths"]["results"]) / "diagnostic"
    tables_dir = Path(config["paths"]["tables"])
    external_dir = Path(config["paths"]["external"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load sources
    # ------------------------------------------------------------------
    diag1 = pd.read_csv(diag_dir / "diag1_career_overlap.csv")
    diag2 = pd.read_csv(diag_dir / "diag2_pb_variance.csv")
    diag3 = pd.read_csv(diag_dir / "diag3_stratified_r2.csv")

    wa = pd.read_csv(external_dir / "wa_results_summary.csv")
    # The World Athletics reference for this table is the cutoff-age sweep
    # (500 bootstrap repeats, GradientBoosting, random 5-fold CV), which is the
    # only WA analysis that spans every cutoff age from 16 to 26. The
    # "Random5Fold" rows in the same file are the WA main-table results and
    # exist only at cutoff ages 18, 20 and 22; selecting them here would leave
    # most of this table's WA column empty.
    WA_STRATEGY = "CutoffCurve500Repeat"
    wa_rand = wa[wa["cv_strategy"] == WA_STRATEGY][["sex", "cutoff_age", "R2", "RMSE", "n"]]
    if wa_rand.empty:
        raise SystemExit(
            f"no rows with cv_strategy == {WA_STRATEGY!r} in "
            f"{external_dir / 'wa_results_summary.csv'}"
        )
    wa_rand = wa_rand.rename(columns={
        "R2": "wa_r2", "RMSE": "wa_rmse", "n": "wa_n",
    })

    logger.info(f"diag1: {len(diag1)} rows  diag2: {len(diag2)} rows  diag3: {len(diag3)} rows")
    logger.info(f"WA reference: {len(wa_rand)} rows ({WA_STRATEGY})")

    # ------------------------------------------------------------------
    # Build unified table from diag3 (already has per-sex, per-cutoff)
    # ------------------------------------------------------------------
    df = diag3.rename(columns={"cutoff": "cutoff_age"}).copy()

    # Merge diag1 (career overlap — combined sexes, add as context columns)
    diag1_cols = diag1.rename(columns={
        "pct_pb_already_achieved": "pct_pb_achieved_combined",
        "pct_career_ended": "pct_career_ended_combined",
        "n_athletes": "n_eligible_combined",
    })
    df = df.merge(
        diag1_cols[["cutoff_age", "pct_pb_achieved_combined",
                     "pct_career_ended_combined", "n_eligible_combined"]],
        on="cutoff_age",
        how="left",
    )

    # Merge WA reference (sex-specific)
    df = df.merge(
        wa_rand[["sex", "cutoff_age", "wa_r2", "wa_rmse", "wa_n"]],
        on=["sex", "cutoff_age"],
        how="left",
    )
    # diag3 already has wa_r2 column from the diagnostic script; prefer the
    # external file values and drop the diagnostic copy
    if "wa_r2_x" in df.columns:
        df = df.drop(columns=["wa_r2_x"]).rename(columns={"wa_r2_y": "wa_r2"})

    # Merge diag2 (PB variance — per-sex, overall not per-cutoff)
    diag2_map = diag2.set_index("sex")[["sd", "wa_sd", "sd_ratio_vs_wa"]]
    diag2_map.index = diag2_map.index.map({"Male": "Male", "Female": "Female"})
    df["pb_sd_population"] = df["sex"].map(diag2_map["sd"])
    df["wa_pb_sd"] = df["sex"].map(diag2_map["wa_sd"])
    df["sd_ratio_vs_wa"] = df["sex"].map(diag2_map["sd_ratio_vs_wa"])

    # ------------------------------------------------------------------
    # Derived columns
    # ------------------------------------------------------------------
    df["r2_inflation"] = df["r2_all"] - df["r2_A_true_pred"]
    df["r2_A_minus_wa"] = df["r2_A_true_pred"] - df["wa_r2"]
    df["pct_B"] = 100.0 - df["pct_A"]

    # ------------------------------------------------------------------
    # Select and order columns for output
    # ------------------------------------------------------------------
    output_cols = [
        "cutoff_age", "sex",
        # Sample sizes
        "n_all", "n_A", "n_B", "pct_A", "pct_B",
        # R² values
        "r2_all", "r2_A_true_pred", "r2_A_ci_lo", "r2_A_ci_hi",
        "r2_B_trivial", "wa_r2",
        # Decomposition
        "r2_inflation", "r2_A_minus_wa",
        # PB variance
        "pb_sd_all", "pb_sd_A", "pb_sd_B",
        "pb_sd_population", "wa_pb_sd", "sd_ratio_vs_wa",
        # Career overlap
        "pct_pb_pre_in_A",
        "pct_pb_achieved_combined", "pct_career_ended_combined",
        # WA context
        "wa_rmse", "wa_n",
    ]
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[output_cols].sort_values(["sex", "cutoff_age"])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df_out.to_csv(tables_dir / "composition_decomposition.csv", index=False)
    logger.info(f"Saved composition_decomposition.csv: {len(df_out)} rows × {len(output_cols)} cols")

    # ------------------------------------------------------------------
    # Log summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 90)
    logger.info("COMPOSITION DECOMPOSITION SUMMARY")
    logger.info("=" * 90)

    for sex in ["Male", "Female"]:
        sub = df_out[df_out["sex"] == sex]
        logger.info(f"\n  {sex}:")
        logger.info(f"  {'cut':>3}  {'n_all':>6}  {'n_A':>5}  {'%A':>5}  "
                     f"{'R²_full':>7}  {'R²_A':>6}  {'R²_B':>5}  "
                     f"{'infl':>6}  {'WA':>5}  {'A-WA':>6}  {'SD_A':>5}")
        for _, r in sub.iterrows():
            wa_val = f"{r['wa_r2']:.3f}" if pd.notna(r.get("wa_r2")) else "  n/a"
            a_wa = f"{r['r2_A_minus_wa']:+.3f}" if pd.notna(r.get("r2_A_minus_wa")) else "  n/a"
            r2_a = f"{r['r2_A_true_pred']:.3f}" if pd.notna(r.get("r2_A_true_pred")) else "  n/a"
            logger.info(
                f"  {int(r['cutoff_age']):>3}  {int(r['n_all']):>6,}  "
                f"{int(r['n_A']):>5,}  {r['pct_A']:>4.1f}%  "
                f"{r['r2_all']:>7.4f}  {r2_a:>6}  {r['r2_B_trivial']:>5.3f}  "
                f"{r['r2_inflation']:>+6.3f}  {wa_val:>5}  {a_wa:>6}  "
                f"{r['pb_sd_A']:>5.3f}"
            )


if __name__ == "__main__":
    main()
