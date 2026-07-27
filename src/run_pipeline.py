"""
Master pipeline orchestrator for CAA Sprint Project.
Run: python src/run_pipeline.py [--step N] [--quick]
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

STEPS = [
    ("s01_parse_raw", "Parse raw CSV files → unified tables"),
    ("s02_clean", "Data cleaning, validation, deduplication"),
    ("s03_province_map", "Map team names → province-level units"),
    ("s04_wind_correction", "Extract wind speeds, apply correction"),
    ("s05_feature_engineer", "Build feature matrices at each cutoff age"),
    ("s06_model_train", "Train models with 3 CV strategies"),
    ("s07_anthro_analysis", "Anthropometric subsample analysis"),
    ("s08_robustness", "Sensitivity analyses and robustness checks"),
    ("s09_comparison", "Compare with WA results, generate overlay plots"),
]


def run_step(step_name: str, description: str, quick: bool = False) -> bool:
    """Run a single pipeline step."""
    logger.info(f"{'='*60}")
    logger.info(f"STARTING: {step_name} — {description}")
    logger.info(f"{'='*60}")

    cmd = [sys.executable, f"src/{step_name}.py"]
    if quick:
        cmd.append("--quick")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error(f"FAILED: {step_name} (exit code {result.returncode})")
        return False

    logger.info(f"COMPLETED: {step_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="CAA Sprint Pipeline")
    parser.add_argument(
        "--step", type=int, default=None,
        help="Run only step N (1-9). Default: run all.",
    )
    parser.add_argument(
        "--from-step", type=int, default=1,
        help="Start from step N (1-9). Default: 1.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer CV repeats (20 instead of 500).",
    )
    args = parser.parse_args()

    if args.step is not None:
        idx = args.step - 1
        if 0 <= idx < len(STEPS):
            run_step(STEPS[idx][0], STEPS[idx][1], args.quick)
        else:
            logger.error(f"Invalid step {args.step}. Must be 1-{len(STEPS)}.")
        return

    for i, (name, desc) in enumerate(STEPS):
        if i + 1 < args.from_step:
            continue
        success = run_step(name, desc, args.quick)
        if not success:
            logger.error(f"Pipeline halted at step {i+1}.")
            sys.exit(1)

    logger.info("="*60)
    logger.info("ALL STEPS COMPLETED SUCCESSFULLY")
    logger.info("="*60)


if __name__ == "__main__":
    main()
