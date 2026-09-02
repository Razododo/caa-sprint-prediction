"""
Assemble the public release directory.

Copies only the files cleared for publication into publish/caa-sprint-prediction/,
so the public git repository never contains the manuscript, the raw scrape or any
intermediate table with personal information. Re-run this after changing any
published file, then commit inside the publish directory.

A final leak scan runs over everything that was copied; the build fails if any
name-like string or full date is found.

Usage:
    python scripts/build_public_repo.py
"""
import csv
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = PROJECT_ROOT / "publish" / "caa-sprint-prediction"

# (source, destination) relative to PROJECT_ROOT / TARGET
DIRS = [
    ("src", "src"),
    ("config", "config"),
    ("data/sample", "data/sample"),
    ("results/tables", "results/tables"),
]
FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "requirements-lock.txt",
    ".gitignore",
    "scripts/demo_reproduce.py",
    "scripts/make_public_sample.py",
    "scripts/build_public_repo.py",
    "scripts/run_permutation_and_parsimonious_group_a.py",
    "scripts/build_supplementary_tables_docx.py",
    "scripts/scrape_track_events.py",
    # Run from the project root; they produce published tables and figures.
    "s16_sensitivity_dk.py",
    "redraw_figures_2_3.py",
    "redraw_figures_4_5.py",
    # Recomputed for the third revision: Supplementary Tables S4 and S13,
    # Table 1, the Methods wind statistics, and the decomposition of the
    # female Group A loss reported in the response letter.
    "recompute_s4_s13.py",
    "recompute_table1.py",
    "recompute_wind_stats.py",
    "decompose_female_loss.py",
    # Peer-review reanalysis scripts, run from the project root. Listed one by
    # one rather than as a directory so that the run logs and RUN_ON_MAC.md
    # sitting beside them stay out of the public repository.
    "r4_reanalysis/gate_reproduce.py",
    "r4_reanalysis/r4_02_carry_forward.py",
    "r4_reanalysis/r4_03_overlap_split.py",
    "r4_reanalysis/r4_05_attrition_censoring.py",
    "r4_reanalysis/r4_06_paired_delta_r2.py",
    "r4_reanalysis/r4_124_remaining.py",
    "r4_reanalysis/r4_78_sensitivity.py",
]

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "*.log")

# Full dates and the fallback id scheme that embeds names are both disqualifying.
LEAK_PATTERNS = [
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "full date"),
    (re.compile(r"_no_reg_"), "name-bearing fallback athlete_id"),
]
# data/sample/ must stay de-identified. Two checks that the pattern list above
# cannot express: Chinese text (province names are a strong quasi-identifier),
# and any float that carries an age at day precision, since age_in_years times
# 365.25 landing on a whole number reconstructs a date of birth from a
# competition date, which is public.
CJK_RE = re.compile(r"[一-鿿]")
DAYS_PER_YEAR = 365.25
# Files where these patterns are legitimate: source code that defines or parses them.
SCAN_EXTS = {".csv", ".json", ".txt", ".md", ".yaml", ".yml"}
SCAN_SKIP = {"README.md"}  # documents the id scheme in prose


def released_files(root: Path):
    """Every file that will be published, excluding git's own bookkeeping."""
    for fp in sorted(root.rglob("*")):
        if fp.is_file() and ".git" not in fp.relative_to(root).parts:
            yield fp


def leak_scan(root: Path) -> list[str]:
    problems = []
    for fp in released_files(root):
        if fp.suffix not in SCAN_EXTS or fp.name in SCAN_SKIP:
            continue
        text = fp.read_text(encoding="utf-8", errors="ignore")
        for pattern, label in LEAK_PATTERNS:
            m = pattern.search(text)
            if m:
                problems.append(f"{fp.relative_to(root)}: {label} -> {m.group(0)!r}")
    return problems


def deid_scan(root: Path) -> list[str]:
    problems = []
    for fp in sorted((root / "data" / "sample").glob("*.csv")):
        with open(fp, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        for col in rows[0]:
            vals = [r[col] for r in rows if r.get(col)]
            if any(CJK_RE.search(v) for v in vals):
                problems.append(f"{fp.name}:{col}: Chinese text in the released sample")
            try:
                nums = [float(v) for v in vals]
            except ValueError:
                continue
            nums = [v for v in nums if abs(v - round(v)) > 1e-6]
            if len(nums) < 20:
                continue
            days = [v * DAYS_PER_YEAR for v in nums]
            frac = sum(abs(d - round(d)) < 1e-6 for d in days) / len(days)
            if frac > 0.5:
                problems.append(
                    f"{fp.name}:{col}: {frac:.0%} of values are a whole number of "
                    "days, so this column carries an age at day precision"
                )
    return problems


def clear_target() -> None:
    """Empty the release directory, but keep .git so the published history and
    remote survive a rebuild."""
    if not TARGET.exists():
        TARGET.mkdir(parents=True)
        return
    for entry in TARGET.iterdir():
        if entry.name == ".git":
            continue
        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()


def main() -> None:
    clear_target()

    for src_rel, dst_rel in DIRS:
        src = PROJECT_ROOT / src_rel
        if not src.exists():
            sys.exit(f"Missing {src}. Run scripts/make_public_sample.py first?")
        shutil.copytree(src, TARGET / dst_rel, ignore=IGNORE)

    for rel in FILES:
        src = PROJECT_ROOT / rel
        if not src.exists():
            sys.exit(f"Missing {src}")
        dst = TARGET / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    problems = leak_scan(TARGET) + deid_scan(TARGET)
    if problems:
        print("LEAK SCAN FAILED — the release was not built cleanly:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    files = list(released_files(TARGET))
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"Leak scan passed. {len(files)} files, {size_mb:.1f} MB in {TARGET}")
    print("\nNext: cd publish/caa-sprint-prediction && git add -A && git commit && git push")


if __name__ == "__main__":
    main()
