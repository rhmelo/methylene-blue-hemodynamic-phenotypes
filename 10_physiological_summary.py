"""Source: original notebook cell 18. Run through run.py."""

from common import phase_dir

# ============================================================
# HiRID Methylene Blue Hemodynamic Phenotypes
# STEP 09 — Final physiological validation documentation
#
# PURPOSE
#
# Generate the final documentation for the HiRID development
# phase, restricted to physiological construct validation.
#
# The script:
# 1. Creates a binary high-norepinephrine-burden endpoint
#    based on AUC above the HiRID cohort median.
# 2. Documents all time windows and endpoint definitions.
# 3. Produces final DOCX tables for the supplement.
# 4. Produces a DOCX Methods and Results draft.
# 5. Freezes the HiRID phenotype definition before MIMIC-IV.
#
# Mortality and other prognostic outcomes are not included.
# ============================================================

import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. DOCX DEPENDENCY
# ------------------------------------------------------------

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import (
    WD_TABLE_ALIGNMENT,
    WD_CELL_VERTICAL_ALIGNMENT,
)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# ------------------------------------------------------------
# 2. PATHS
# ------------------------------------------------------------

PROJECT_DIR = phase_dir("hirid")

OUTPUT_DIR = PROJECT_DIR / "outputs"

TABLE_DIR = OUTPUT_DIR / "tables"

FIGURE_DIR = OUTPUT_DIR / "figures"

LOG_DIR = OUTPUT_DIR / "logs"

DERIVED_DIR = PROJECT_DIR / "derived_data"

FINAL_DIR = PROJECT_DIR / "HiRID_Final_Physiological_Package"

DOCX_DIR = FINAL_DIR / "DOCX"

DATA_DIR = FINAL_DIR / "Final_Data_Tables"

PLAN_DIR = FINAL_DIR / "Frozen_Definitions"

for folder in [
    FINAL_DIR,
    DOCX_DIR,
    DATA_DIR,
    PLAN_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

ANALYSIS_PATH = (
    DERIVED_DIR / "step08_HiRID_final_dataset_with_clinical_outcomes.parquet"
)

if not ANALYSIS_PATH.exists():

    ANALYSIS_PATH = (
        DERIVED_DIR
        / "step07_analysis_dataset_with_prespecified_physiological_phenotypes.parquet"
    )

if not ANALYSIS_PATH.exists():

    raise FileNotFoundError(f"Analysis dataset not found:\n{ANALYSIS_PATH}")

# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

analysis = pd.read_parquet(ANALYSIS_PATH)

analysis["patientid"] = pd.to_numeric(
    analysis["patientid"],
    errors="coerce",
).astype(int)

print("=" * 80)
print("INPUT DATA")
print("=" * 80)

print("Input file:", ANALYSIS_PATH.name)
print("Patients:", len(analysis))

# ------------------------------------------------------------
# 4. STANDARDIZE PHENOTYPE LABELS
# ------------------------------------------------------------

PHENOTYPE_LABELS = {
    1: "Vasoplegia-compatible",
    2: "Mixed/low-pulsatility",
    3: "Preserved diastolic pressure",
}

analysis["physiological_phenotype_label_final"] = analysis[
    "physiological_phenotype"
].map(PHENOTYPE_LABELS)

# ------------------------------------------------------------
# 5. EXACT VALIDATION-ENDPOINT DEFINITIONS
# ------------------------------------------------------------

# Validation period:
# Hourly bins from >1 through 6 hours after time zero.
#
# The hourly dataset created previously contains:
# hour_bin 1 = >1 to <2 h
# hour_bin 2 = >=2 to <3 h
# hour_bin 3 = >=3 to <4 h
# hour_bin 4 = >=4 to <5 h
# hour_bin 5 = >=5 to <=6 h
#
# Persistent hypotension was defined by an abnormal value in
# at least 60% of available hourly bins.
#
# With all 5 bins available, this is equivalent to at least
# 3 of the 5 hourly bins.

if "persistent_dbp_lt_50_h1_h6" not in analysis.columns:

    analysis["persistent_dbp_lt_50_h1_h6"] = np.where(
        analysis["proportion_hours_dbp_lt_50_h1_h6_validation"].notna(),
        analysis["proportion_hours_dbp_lt_50_h1_h6_validation"].ge(0.60).astype(float),
        np.nan,
    )

if "persistent_map_lt_65_h1_h6" not in analysis.columns:

    analysis["persistent_map_lt_65_h1_h6"] = np.where(
        analysis["proportion_hours_map_lt_65_h1_h6_validation"].notna(),
        analysis["proportion_hours_map_lt_65_h1_h6_validation"].ge(0.60).astype(float),
        np.nan,
    )

# ------------------------------------------------------------
# 6. HIGH SUBSEQUENT NOREPINEPHRINE BURDEN
# ------------------------------------------------------------

NE_AUC_COLUMN = "norepi_ug_kg_min_auc_h1_h6"

if NE_AUC_COLUMN not in analysis.columns:

    raise RuntimeError(f"Required column not found: {NE_AUC_COLUMN}")

valid_ne_auc = analysis[NE_AUC_COLUMN].dropna()

if valid_ne_auc.empty:

    raise RuntimeError("No valid norepinephrine AUC values were found.")

NE_AUC_MEDIAN = float(valid_ne_auc.median())

analysis["high_norepinephrine_burden_h1_h6"] = np.where(
    analysis[NE_AUC_COLUMN].notna(),
    analysis[NE_AUC_COLUMN].gt(NE_AUC_MEDIAN).astype(float),
    np.nan,
)

print(
    "\nMedian norepinephrine AUC (>1–6 h):",
    round(
        NE_AUC_MEDIAN,
        6,
    ),
)

print(
    "Patients with valid norepinephrine AUC:",
    int(analysis[NE_AUC_COLUMN].notna().sum()),
)

print(
    "Patients above median norepinephrine AUC:",
    int(analysis["high_norepinephrine_burden_h1_h6"].eq(1).sum()),
)

# ------------------------------------------------------------
# 7. OPTIONAL PHYSIOLOGICAL BURDEN COUNT
# ------------------------------------------------------------

# This is descriptive only, not a primary composite endpoint.
# It counts the number of adverse physiological domains:
#
# 1. Persistent DBP <50
# 2. Persistent MAP <65
# 3. NE AUC above the cohort median

burden_components = [
    "persistent_dbp_lt_50_h1_h6",
    "persistent_map_lt_65_h1_h6",
    "high_norepinephrine_burden_h1_h6",
]

analysis["n_adverse_physiological_domains_h1_h6"] = analysis[burden_components].sum(
    axis=1,
    min_count=3,
)

# ------------------------------------------------------------
# 8. SUMMARY HELPERS
# ------------------------------------------------------------


def median_iqr(
    series,
    digits=2,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return "NA"

    median = values.median()
    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)

    return f"{median:.{digits}f} " f"({q25:.{digits}f}–{q75:.{digits}f})"


def n_percent(
    series,
    digits=1,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return "NA"

    events = int(values.eq(1).sum())

    percent = 100 * events / len(values)

    return f"{events}/{len(values)} " f"({percent:.{digits}f}%)"


# ------------------------------------------------------------
# 9. PHENOTYPE COUNTS
# ------------------------------------------------------------

phenotype_counts = (
    analysis.dropna(subset=["physiological_phenotype"])
    .groupby(
        [
            "physiological_phenotype",
            "physiological_phenotype_label_final",
        ],
        as_index=False,
    )
    .agg(
        n=(
            "patientid",
            "size",
        )
    )
)

phenotype_counts["percent"] = 100 * phenotype_counts["n"] / phenotype_counts["n"].sum()

# ------------------------------------------------------------
# 10. FINAL PHYSIOLOGICAL TABLE
# ------------------------------------------------------------

TABLE_VARIABLES = [
    {
        "label": "Patients, n (%)",
        "type": "count",
    },
    {
        "label": "Baseline DBP, mmHg",
        "type": "continuous",
        "column": "dbp_baseline",
        "digits": 1,
    },
    {
        "label": "Baseline pulse pressure, mmHg",
        "type": "continuous",
        "column": "pulse_pressure_baseline",
        "digits": 1,
    },
    {
        "label": "Baseline heart rate, beats/min",
        "type": "continuous",
        "column": "heart_rate_baseline",
        "digits": 1,
    },
    {
        "label": "Baseline MAP, mmHg",
        "type": "continuous",
        "column": "map_baseline",
        "digits": 1,
    },
    {
        "label": "Baseline norepinephrine, µg/kg/min",
        "type": "continuous",
        "column": "norepi_ug_kg_min_baseline_winsor",
        "digits": 3,
    },
    {
        "label": "Baseline lactate, mmol/L",
        "type": "continuous",
        "column": "lactate_baseline",
        "digits": 2,
    },
    {
        "label": "Follow-up DBP, >1–6 h, mmHg",
        "type": "continuous",
        "column": "dbp_median_h1_h6_weighted_validation",
        "digits": 1,
    },
    {
        "label": "Follow-up pulse pressure, >1–6 h, mmHg",
        "type": "continuous",
        "column": "pulse_pressure_median_h1_h6_validation",
        "digits": 1,
    },
    {
        "label": "Persistent DBP <50 mmHg, >1–6 h",
        "type": "binary",
        "column": "persistent_dbp_lt_50_h1_h6",
    },
    {
        "label": "Persistent MAP <65 mmHg, >1–6 h",
        "type": "binary",
        "column": "persistent_map_lt_65_h1_h6",
    },
    {
        "label": "Median norepinephrine, >1–6 h, µg/kg/min",
        "type": "continuous",
        "column": "norepi_ug_kg_min_median_h1_h6",
        "digits": 3,
    },
    {
        "label": "Norepinephrine AUC, >1–6 h",
        "type": "continuous",
        "column": "norepi_ug_kg_min_auc_h1_h6",
        "digits": 3,
    },
    {
        "label": "Norepinephrine AUC above cohort median",
        "type": "binary",
        "column": "high_norepinephrine_burden_h1_h6",
    },
    {
        "label": "Diastolic shock index",
        "type": "continuous",
        "column": "diastolic_shock_index",
        "digits": 2,
    },
]

table_rows = []

for specification in TABLE_VARIABLES:

    row = {"Variable": specification["label"]}

    for phenotype in [
        1,
        2,
        3,
    ]:

        group = analysis.loc[analysis["physiological_phenotype"].eq(phenotype)]

        column_label = PHENOTYPE_LABELS[phenotype]

        if specification["type"] == "count":

            n = len(group)

            percent = 100 * n / len(analysis)

            row[column_label] = f"{n} " f"({percent:.1f}%)"

        elif specification["type"] == "continuous":

            column = specification["column"]

            if column in group.columns:

                row[column_label] = median_iqr(
                    group[column],
                    digits=specification["digits"],
                )

            else:

                row[column_label] = "NA"

        elif specification["type"] == "binary":

            column = specification["column"]

            if column in group.columns:

                row[column_label] = n_percent(group[column])

            else:

                row[column_label] = "NA"

    table_rows.append(row)

final_physiological_table = pd.DataFrame(table_rows)

# ------------------------------------------------------------
# 11. EXACT ANALYSIS DEFINITIONS TABLE
# ------------------------------------------------------------

definitions_table = pd.DataFrame(
    [
        {
            "Element": "Time zero",
            "Operational definition": (
                "Start of the first continuous norepinephrine "
                "episode lasting at least 60 minutes"
            ),
        },
        {
            "Element": "Baseline phenotype window",
            "Operational definition": (
                "Median invasive measurements from 0 through " "1 hour after time zero"
            ),
        },
        {
            "Element": "Physiological validation window",
            "Operational definition": (
                "Five hourly bins from >1 through 6 hours " "after time zero"
            ),
        },
        {
            "Element": "Hourly summary",
            "Operational definition": ("Median value within each hourly bin"),
        },
        {
            "Element": "Persistent DBP <50 mmHg",
            "Operational definition": (
                "DBP <50 mmHg in at least 60% of available "
                "hourly bins during >1–6 hours; with all five "
                "bins available, equivalent to at least 3 of 5 bins"
            ),
        },
        {
            "Element": "Persistent MAP <65 mmHg",
            "Operational definition": (
                "MAP <65 mmHg in at least 60% of available "
                "hourly bins during >1–6 hours; with all five "
                "bins available, equivalent to at least 3 of 5 bins"
            ),
        },
        {
            "Element": "Norepinephrine AUC",
            "Operational definition": (
                "Trapezoidal area under the curve of hourly "
                "median weight-normalized norepinephrine doses "
                "during >1–6 hours"
            ),
        },
        {
            "Element": "High norepinephrine burden",
            "Operational definition": (
                "Norepinephrine AUC above the median AUC among "
                "HiRID patients with valid AUC data "
                f"(median={NE_AUC_MEDIAN:.6f})"
            ),
        },
        {
            "Element": "Vasoplegia-compatible phenotype",
            "Operational definition": (
                "Baseline DBP <50 mmHg and pulse pressure " "≥40 mmHg"
            ),
        },
        {
            "Element": "Mixed/low-pulsatility phenotype",
            "Operational definition": (
                "Baseline DBP <50 mmHg and pulse pressure " "<40 mmHg"
            ),
        },
        {
            "Element": "Preserved-diastolic-pressure phenotype",
            "Operational definition": (
                "Baseline DBP ≥50 mmHg, irrespective of " "pulse pressure"
            ),
        },
    ]
)

# ------------------------------------------------------------
# 12. METHODS/RESULTS NUMERIC VALUES
# ------------------------------------------------------------


def phenotype_n(
    phenotype,
):

    return int(analysis["physiological_phenotype"].eq(phenotype).sum())


def phenotype_percent(
    phenotype,
):

    return float(100 * analysis["physiological_phenotype"].eq(phenotype).mean())


def phenotype_continuous(
    phenotype,
    column,
    digits=1,
):

    return median_iqr(
        analysis.loc[
            analysis["physiological_phenotype"].eq(phenotype),
            column,
        ],
        digits=digits,
    )


def phenotype_binary(
    phenotype,
    column,
):

    return n_percent(
        analysis.loc[
            analysis["physiological_phenotype"].eq(phenotype),
            column,
        ]
    )


# ------------------------------------------------------------
# 13. DOCX HELPERS
# ------------------------------------------------------------


def configure_document(
    document,
):

    section = document.sections[0]

    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    styles = document.styles

    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10)

    for style_name in [
        "Title",
        "Heading 1",
        "Heading 2",
        "Heading 3",
    ]:

        styles[style_name].font.name = "Arial"


def set_cell_shading(
    cell,
    fill="D9EAF7",
):

    tc_pr = cell._tc.get_or_add_tcPr()

    shading = OxmlElement("w:shd")

    shading.set(
        qn("w:fill"),
        fill,
    )

    tc_pr.append(shading)


def add_dataframe_table(
    document,
    dataframe,
    font_size=8,
):

    table = document.add_table(
        rows=1,
        cols=len(dataframe.columns),
    )

    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for index, column in enumerate(dataframe.columns):

        cell = table.rows[0].cells[index]

        cell.text = str(column)

        set_cell_shading(cell)

        for run in cell.paragraphs[0].runs:

            run.bold = True
            run.font.size = Pt(font_size)

    for _, row in dataframe.iterrows():

        cells = table.add_row().cells

        for index, value in enumerate(row):

            cells[index].text = str(value)

            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

            for paragraph in cells[index].paragraphs:

                paragraph.paragraph_format.space_after = Pt(0)

                for run in paragraph.runs:

                    run.font.size = Pt(font_size)

    return table


# ------------------------------------------------------------
# 14. METHODS AND RESULTS DOCX
# ------------------------------------------------------------

document = Document()

configure_document(document)

title = document.add_paragraph()

title.alignment = WD_ALIGN_PARAGRAPH.CENTER

title_run = title.add_run("HiRID Physiological Development Phase: Methods and Results")

title_run.bold = True
title_run.font.size = Pt(16)

document.add_heading(
    "Purpose of the HiRID phase",
    level=1,
)

document.add_paragraph(
    (
        "The HiRID analysis was designed exclusively as a "
        "macrohemodynamic phenotype-development and physiological "
        "construct-validation phase. It was not intended to evaluate "
        "methylene blue exposure, estimate a causal treatment effect, "
        "or develop a general prognostic model. The prespecified "
        "phenotypes were subsequently frozen for application in the "
        "MIMIC-IV methylene blue cohort."
    )
)

document.add_heading(
    "Methods",
    level=1,
)

document.add_heading(
    "Cohort and time zero",
    level=2,
)

document.add_paragraph(
    (
        "Patients with vasopressor-dependent probable septic shock "
        "were identified using continuous norepinephrine exposure "
        "lasting at least 60 minutes, systemic intravenous antibiotic "
        "administration temporally associated with vasopressor "
        "initiation, an antibiotic course spanning more than 24 hours, "
        "and availability of invasive arterial pressure measurements. "
        "Time zero was defined as the beginning of the first qualifying "
        "continuous norepinephrine episode."
    )
)

document.add_heading(
    "Baseline macrohemodynamic phenotype window",
    level=2,
)

document.add_paragraph(
    (
        "Baseline systolic arterial pressure, diastolic arterial "
        "pressure, mean arterial pressure, pulse pressure, and heart "
        "rate were summarized using the median of invasive measurements "
        "recorded from time zero through 1 hour. Pulse pressure was "
        "calculated as systolic minus diastolic arterial pressure."
    )
)

document.add_heading(
    "Prespecified phenotype definitions",
    level=2,
)

document.add_paragraph(
    (
        "The vasoplegia-compatible phenotype was defined as baseline "
        "diastolic arterial pressure <50 mmHg with pulse pressure "
        "≥40 mmHg. The mixed/low-pulsatility phenotype was defined as "
        "diastolic arterial pressure <50 mmHg with pulse pressure "
        "<40 mmHg. The preserved-diastolic-pressure phenotype was "
        "defined as diastolic arterial pressure ≥50 mmHg, irrespective "
        "of pulse pressure. Heart rate, norepinephrine dose, lactate, "
        "shock indices, and subsequent hemodynamic measurements were "
        "not used to assign phenotypes."
    )
)

document.add_heading(
    "Physiological validation window",
    level=2,
)

document.add_paragraph(
    (
        "Physiological validation was performed after completion of "
        "the phenotype-assignment window. Measurements from more than "
        "1 hour through 6 hours after time zero were divided into five "
        "hourly intervals (>1–2, >2–3, >3–4, >4–5, and >5–6 hours). "
        "The median value within each hourly interval was used for "
        "longitudinal summaries."
    )
)

document.add_heading(
    "Persistent arterial hypotension",
    level=2,
)

document.add_paragraph(
    (
        "Persistent diastolic hypotension was defined as diastolic "
        "arterial pressure <50 mmHg in at least 60% of the available "
        "hourly intervals during the >1–6-hour validation window. "
        "Persistent mean arterial hypotension was similarly defined "
        "as mean arterial pressure <65 mmHg in at least 60% of the "
        "available intervals. Among patients with all five intervals "
        "available, the 60% criterion corresponded to at least three "
        "of five hourly intervals."
    )
)

document.add_heading(
    "Subsequent norepinephrine burden",
    level=2,
)

document.add_paragraph(
    (
        "Norepinephrine administration was reconstructed from serial "
        "infusion records and normalized to body weight. Subsequent "
        "vasopressor burden was summarized using the median hourly "
        "norepinephrine dose and the trapezoidal area under the "
        "norepinephrine dose-time curve during the >1–6-hour "
        "validation period. A high subsequent norepinephrine-burden "
        "endpoint was defined as an area under the curve above the "
        f"median of the HiRID development cohort ({NE_AUC_MEDIAN:.6f})."
    )
)

document.add_heading(
    "Comparator indices",
    level=2,
)

document.add_paragraph(
    (
        "The prespecified phenotypes were compared with diastolic "
        "arterial pressure alone, baseline norepinephrine dose, the "
        "and the diastolic shock index. These variables were used for construct "
        "validation and did not contribute to phenotype assignment."
    )
)

document.add_heading(
    "Statistical analysis",
    level=2,
)

document.add_paragraph(
    (
        "Continuous variables were summarized as medians and "
        "interquartile ranges, and binary endpoints as counts and "
        "percentages. Cross-validated logistic models were used to "
        "compare discrimination for persistent diastolic and mean "
        "arterial hypotension. Cross-validated ridge-regression models "
        "were used to assess prediction of log-transformed subsequent "
        "norepinephrine area under the curve. These analyses evaluated "
        "physiological construct validity rather than causality."
    )
)

document.add_heading(
    "Results",
    level=1,
)

document.add_heading(
    "Phenotype distribution",
    level=2,
)

document.add_paragraph(
    (
        f"Among {len(analysis):,} patients, "
        f"{phenotype_n(1):,} ({phenotype_percent(1):.1f}%) were "
        f"classified as vasoplegia-compatible, "
        f"{phenotype_n(2):,} ({phenotype_percent(2):.1f}%) as "
        f"mixed/low-pulsatility, and "
        f"{phenotype_n(3):,} ({phenotype_percent(3):.1f}%) as having "
        f"preserved diastolic pressure."
    )
)

document.add_heading(
    "Baseline physiological distinction",
    level=2,
)

document.add_paragraph(
    (
        "Baseline diastolic arterial pressure was "
        f"{phenotype_continuous(1, 'dbp_baseline', 1)} mmHg in the "
        "vasoplegia-compatible phenotype and "
        f"{phenotype_continuous(2, 'dbp_baseline', 1)} mmHg in the "
        "mixed/low-pulsatility phenotype. Despite similarly reduced "
        "diastolic pressure, baseline pulse pressure was "
        f"{phenotype_continuous(1, 'pulse_pressure_baseline', 1)} "
        "mmHg in the vasoplegia-compatible phenotype and "
        f"{phenotype_continuous(2, 'pulse_pressure_baseline', 1)} "
        "mmHg in the mixed/low-pulsatility phenotype. The latter also "
        "had a higher heart rate and lactate concentration."
    )
)

document.add_heading(
    "Longitudinal physiological validation",
    level=2,
)

document.add_paragraph(
    (
        "The macrohemodynamic differences persisted during the "
        ">1–6-hour validation period. Follow-up pulse pressure was "
        f"{phenotype_continuous(1, 'pulse_pressure_median_h1_h6_validation', 1)} "
        "mmHg in the vasoplegia-compatible phenotype and "
        f"{phenotype_continuous(2, 'pulse_pressure_median_h1_h6_validation', 1)} "
        "mmHg in the mixed/low-pulsatility phenotype. Persistent "
        "diastolic arterial pressure <50 mmHg occurred in "
        f"{phenotype_binary(1, 'persistent_dbp_lt_50_h1_h6')} of "
        "vasoplegia-compatible patients, "
        f"{phenotype_binary(2, 'persistent_dbp_lt_50_h1_h6')} of "
        "mixed/low-pulsatility patients, and "
        f"{phenotype_binary(3, 'persistent_dbp_lt_50_h1_h6')} of "
        "patients with preserved diastolic pressure."
    )
)

document.add_paragraph(
    (
        "Persistent mean arterial pressure <65 mmHg occurred in "
        f"{phenotype_binary(1, 'persistent_map_lt_65_h1_h6')} of "
        "vasoplegia-compatible patients, "
        f"{phenotype_binary(2, 'persistent_map_lt_65_h1_h6')} of "
        "mixed/low-pulsatility patients, and "
        f"{phenotype_binary(3, 'persistent_map_lt_65_h1_h6')} of "
        "patients with preserved diastolic pressure."
    )
)

document.add_heading(
    "Subsequent vasopressor burden",
    level=2,
)

document.add_paragraph(
    (
        "The mixed/low-pulsatility phenotype demonstrated the highest "
        "subsequent norepinephrine exposure. Median norepinephrine AUC "
        "during >1–6 hours was "
        f"{phenotype_continuous(1, 'norepi_ug_kg_min_auc_h1_h6', 3)} "
        "in the vasoplegia-compatible phenotype, "
        f"{phenotype_continuous(2, 'norepi_ug_kg_min_auc_h1_h6', 3)} "
        "in the mixed/low-pulsatility phenotype, and "
        f"{phenotype_continuous(3, 'norepi_ug_kg_min_auc_h1_h6', 3)} "
        "in the preserved-diastolic-pressure phenotype. An AUC above "
        "the cohort median occurred in "
        f"{phenotype_binary(1, 'high_norepinephrine_burden_h1_h6')}, "
        f"{phenotype_binary(2, 'high_norepinephrine_burden_h1_h6')}, "
        f"and {phenotype_binary(3, 'high_norepinephrine_burden_h1_h6')} "
        "of the respective phenotypes."
    )
)

document.add_heading(
    "Incremental physiological information",
    level=2,
)

document.add_paragraph(
    (
        "Baseline norepinephrine dose alone showed limited "
        "discrimination for persistent diastolic hypotension "
        "(cross-validated AUC 0.56) and persistent mean arterial "
        "hypotension (AUC 0.60). Adding the prespecified phenotype "
        "increased the corresponding AUCs to 0.83 and 0.80. The "
        "phenotype produced only a modest incremental improvement in "
        "prediction of subsequent norepinephrine AUC beyond baseline "
        "norepinephrine dose, consistent with its intended role as a "
        "descriptor of persistent macrohemodynamic state rather than "
        "a general predictor of treatment intensity."
    )
)

document.add_heading(
    "Interpretation",
    level=2,
)

document.add_paragraph(
    (
        "These findings demonstrate that patients with similarly low "
        "diastolic arterial pressure may exhibit distinct persistent "
        "macrohemodynamic states. Low diastolic pressure with preserved "
        "pulsatility remained distinguishable from a lower-pulsatility, "
        "more tachycardic state with greater subsequent vasopressor "
        "burden. The HiRID findings provide physiological construct "
        "validation for transport of the frozen phenotype definition "
        "to the MIMIC-IV methylene blue cohort but do not establish "
        "vasoplegia or treatment responsiveness."
    )
)

document.add_page_break()

document.add_heading(
    "Exact operational definitions",
    level=1,
)

add_dataframe_table(
    document,
    definitions_table,
    font_size=8,
)

document.add_heading(
    "Final physiological results table",
    level=1,
)

add_dataframe_table(
    document,
    final_physiological_table,
    font_size=7,
)

methods_results_path = DOCX_DIR / "HiRID_Final_Physiological_Methods_and_Results.docx"

document.save(methods_results_path)

# ------------------------------------------------------------
# 15. SUPPLEMENTARY TABLES DOCX
# ------------------------------------------------------------

supplement = Document()

configure_document(supplement)

supplement.add_heading(
    "Supplementary Table S1. Exact HiRID operational definitions",
    level=1,
)

add_dataframe_table(
    supplement,
    definitions_table,
    font_size=8,
)

supplement.add_paragraph(
    (
        "DBP indicates diastolic blood pressure; MAP, mean arterial "
        "pressure; and AUC, area under the curve."
    )
)

supplement.add_page_break()

supplement.add_heading(
    (
        "Supplementary Table S2. Baseline characteristics "
        "and longitudinal physiological validation"
    ),
    level=1,
)

add_dataframe_table(
    supplement,
    final_physiological_table,
    font_size=7,
)

supplement.add_paragraph(
    (
        "Values are median (interquartile range) or n/N (%). "
        "Phenotypes were defined solely using baseline invasive DBP "
        "and pulse pressure. Follow-up values were calculated during "
        "the >1–6-hour validation window."
    )
)

supplement_path = DOCX_DIR / "HiRID_Final_Supplementary_Tables.docx"

supplement.save(supplement_path)

# ------------------------------------------------------------
# 16. MACHINE-READABLE FROZEN DEFINITIONS
# ------------------------------------------------------------

frozen_definitions = {
    "HiRID_role": (
        "Macrohemodynamic phenotype development and "
        "physiological construct validation"
    ),
    "time_zero": (
        "Start of first continuous norepinephrine episode "
        "lasting at least 60 minutes"
    ),
    "baseline_window": ("Median invasive measurements from 0 through 1 hour"),
    "validation_window": {
        "start": ">1 hour after time zero",
        "end": "6 hours after time zero",
        "hourly_bins": 5,
        "bins": [
            ">1–2 h",
            ">2–3 h",
            ">3–4 h",
            ">4–5 h",
            ">5–6 h",
        ],
    },
    "persistent_hypotension_definition": {
        "threshold": "Abnormal in >=60% of available hourly bins",
        "complete_data_equivalent": "At least 3 of 5 hourly bins",
        "DBP": "<50 mmHg",
        "MAP": "<65 mmHg",
    },
    "norepinephrine_AUC": {
        "method": "Trapezoidal AUC of hourly median weight-normalized dose",
        "window": ">1–6 hours",
        "cohort_median": NE_AUC_MEDIAN,
        "high_burden_definition": "AUC above HiRID cohort median",
    },
    "primary_phenotypes": {
        "vasoplegia_compatible": "DBP <50 mmHg and PP >=40 mmHg",
        "mixed_low_pulsatility": "DBP <50 mmHg and PP <40 mmHg",
        "preserved_diastolic_pressure": "DBP >=50 mmHg",
    },
    "variables_not_used_for_assignment": [
        "Heart rate",
        "MAP",
        "Norepinephrine dose",
        "Lactate",
        "DSI",
        "Longitudinal measurements",
        "Clinical outcomes",
    ],
    "mortality_analysis_in_HiRID": False,
    "phenotype_definition_frozen_before_MIMIC": True,
}

frozen_json_path = PLAN_DIR / "HiRID_Final_Physiological_Definitions.json"

with open(
    frozen_json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        frozen_definitions,
        f,
        indent=2,
        ensure_ascii=False,
    )

# ------------------------------------------------------------
# 17. SAVE FINAL DATA TABLES
# ------------------------------------------------------------

analysis.to_parquet(
    DERIVED_DIR / "step09_HiRID_final_physiological_dataset.parquet",
    index=False,
)

analysis.to_csv(
    DATA_DIR / "HiRID_final_physiological_dataset.csv",
    index=False,
)

phenotype_counts.to_csv(
    DATA_DIR / "HiRID_final_phenotype_counts.csv",
    index=False,
)

definitions_table.to_csv(
    DATA_DIR / "HiRID_exact_operational_definitions.csv",
    index=False,
)

final_physiological_table.to_csv(
    DATA_DIR / "HiRID_final_physiological_table.csv",
    index=False,
)

# ------------------------------------------------------------
# 18. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 09 COMPLETE — HiRID PHYSIOLOGICAL PHASE CLOSED")
print("=" * 80)

print(
    "\nMethods and Results DOCX:\n",
    methods_results_path,
)

print(
    "\nSupplementary tables DOCX:\n",
    supplement_path,
)

print(
    "\nFrozen definitions:\n",
    frozen_json_path,
)

print(
    "\nFinal physiological dataset:\n",
    DERIVED_DIR / "step09_HiRID_final_physiological_dataset.parquet",
)

print("\nThe HiRID phase now includes:")

print("- Exact phenotype definitions")

print("- Baseline window: 0–1 h")

print("- Validation window: >1–6 h")

print("- Persistent hypotension: >=60% of hourly bins")

print("- High NE burden: AUC above cohort median")

print("- No mortality or causal treatment analysis")

print("- Frozen classification for MIMIC-IV")
