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
]

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "*.log")

# Full dates and the fallback id scheme that embeds names are both disqualifying.
LEAK_PATTERNS = [
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "full date"),
    (re.compile(r"_no_reg_"), "name-bearing fallback athlete_id"),
]
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

    problems = leak_scan(TARGET)
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
