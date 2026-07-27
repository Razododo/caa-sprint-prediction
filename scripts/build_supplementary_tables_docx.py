"""
Build Supplementary Tables S1 and S2 into a single Word document.
Requires: pip install python-docx
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH


def fmt_num(x, decimals=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if isinstance(x, float):
        if decimals == 0:
            return f"{int(round(x))}"
        return f"{x:.{decimals}f}"
    return str(x)


def main():
    tables_dir = ROOT / "results" / "tables"
    out_path = ROOT / "results" / "Supplementary_Tables_S1_S2.docx"

    doc = Document()
    doc.add_heading("Supplementary Tables", level=0)

    # ========== S1: Provincial heterogeneity ==========
    doc.add_heading("Supplementary Table S1.", level=1)
    doc.add_paragraph(
        "Provincial heterogeneity in attrition metrics among provinces with ≥ 200 registered "
        "100-m male athletes."
    )
    df1 = pd.read_csv(tables_dir / "province_heterogeneity.csv")
    df1 = df1[df1["n_athletes"] >= 200].copy()
    df1["dropout_rate_age_18"] = (df1["dropout_rate_age_18"] * 100).round(1)
    df1["mean_lifetime_pb"] = df1["mean_lifetime_pb"].round(3)
    df1["median_career_span_years"] = df1["median_career_span_years"].round(2)
    df1 = df1.rename(columns={
        "province": "Province",
        "n_athletes": "N athletes",
        "median_career_span_years": "Median career (yr)",
        "mean_lifetime_pb": "Mean PB (s)",
        "active_at_age_18": "Active @18",
        "dropout_at_age_18": "Dropout @18",
        "dropout_rate_age_18": "Dropout rate @18 (%)",
        "n_national_level": "N national",
        "pct_male": "% male",
    })
    cols_s1 = ["Province", "N athletes", "Median career (yr)", "Mean PB (s)", "Active @18", "Dropout @18", "Dropout rate @18 (%)", "N national", "% male"]
    df1 = df1[[c for c in cols_s1 if c in df1.columns]]

    t1 = doc.add_table(rows=1 + len(df1), cols=len(df1.columns))
    t1.style = "Table Grid"
    for j, c in enumerate(df1.columns):
        t1.rows[0].cells[j].text = c
    for i, row in df1.iterrows():
        for j, c in enumerate(df1.columns):
            val = row[c]
            if "rate" in c or "male" in c:
                t1.rows[i + 1].cells[j].text = fmt_num(val, 1)
            elif "PB" in c or "career" in c:
                t1.rows[i + 1].cells[j].text = fmt_num(val, 3)
            else:
                t1.rows[i + 1].cells[j].text = fmt_num(val, 0)
    doc.add_paragraph()

    # ========== S2: Composition decomposition ==========
    doc.add_heading("Supplementary Table S2.", level=1)
    doc.add_paragraph(
        "Full population composition decomposition across cutoff ages 16–26 for male and female athletes, "
        "including sample sizes, Group A proportions, R²full, R²A (with 95% CI), R²B, World Athletics reference values, "
        "composition inflation, and PB standard deviations."
    )
    df2 = pd.read_csv(tables_dir / "composition_decomposition.csv")
    df2 = df2.sort_values(["cutoff_age", "sex"])
    df2["pct_A"] = df2["pct_A"].round(1)
    df2["r2_all"] = df2["r2_all"].round(4)
    df2["r2_A_true_pred"] = df2["r2_A_true_pred"].round(4)
    df2["r2_A_ci_lo"] = df2["r2_A_ci_lo"].round(4)
    df2["r2_A_ci_hi"] = df2["r2_A_ci_hi"].round(4)
    df2["r2_B_trivial"] = df2["r2_B_trivial"].round(4)
    df2["r2_inflation"] = df2["r2_inflation"].round(4)
    df2["r2_A_minus_wa"] = df2["r2_A_minus_wa"].round(4)
    df2["pb_sd_A"] = df2["pb_sd_A"].round(3)

    cols_s2 = [
        "cutoff_age", "sex", "n_all", "n_A", "n_B", "pct_A",
        "r2_all", "r2_A_true_pred", "r2_A_ci_lo", "r2_A_ci_hi", "r2_B_trivial",
        "wa_r2", "r2_inflation", "r2_A_minus_wa", "pb_sd_A"
    ]
    headers_s2 = [
        "Cutoff", "Sex", "N all", "N A", "N B", "% A",
        "R²full", "R²A", "R²A CI lo", "R²A CI hi", "R²B",
        "WA R²", "R² inflation", "R²A − WA", "PB SD (A)"
    ]
    df2 = df2[[c for c in cols_s2 if c in df2.columns]]
    df2.columns = headers_s2

    t2 = doc.add_table(rows=1 + len(df2), cols=len(df2.columns))
    t2.style = "Table Grid"
    for j, c in enumerate(headers_s2):
        t2.rows[0].cells[j].text = c
    for i, idx in enumerate(df2.index):
        row = df2.loc[idx]
        for j, c in enumerate(headers_s2):
            val = row[c]
            if c in ("Cutoff", "Sex"):
                t2.rows[i + 1].cells[j].text = str(int(val)) if c == "Cutoff" and pd.notna(val) else str(val)
            elif c in ("N all", "N A", "N B"):
                t2.rows[i + 1].cells[j].text = str(int(val)) if pd.notna(val) else "—"
            elif pd.isna(val) or (isinstance(val, float) and pd.isna(val)):
                t2.rows[i + 1].cells[j].text = "—"
            elif c == "WA R²":
                t2.rows[i + 1].cells[j].text = fmt_num(val, 3)
            elif c == "PB SD (A)":
                t2.rows[i + 1].cells[j].text = fmt_num(val, 3)
            else:
                t2.rows[i + 1].cells[j].text = fmt_num(val, 4)
    doc.add_paragraph()

    doc.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
