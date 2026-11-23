"""Source: original notebook cell 12. Run through run.py."""

from common import display, hirid_source, phase_dir

# ============================================================
# HiRID Methylene Blue Hemodynamic Phenotypes
# STEP 04 — Weight-normalized norepinephrine dose
#
# Objectives:
# - Identify body-weight variables in the HiRID dictionary
# - Extract the weight closest to time zero
# - Prefer measurements from before time zero
# - Calculate norepinephrine dose in µg/kg/min
# - Audit plausibility and completeness
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import tarfile
import tempfile
import io
import json
import gc
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

HIRID_DIR = hirid_source()

PROJECT_DIR = phase_dir("hirid")
TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
LOG_DIR = PROJECT_DIR / "outputs" / "logs"
DERIVED_DIR = PROJECT_DIR / "derived_data"

REFERENCE_ARCHIVE = HIRID_DIR / "reference_data.tar.gz"
OBS_ARCHIVE = HIRID_DIR / "observation_tables_parquet.tar.gz"

ANALYSIS_PATH = (
    DERIVED_DIR / "step03b_analysis_ready_dataset_clean_norepinephrine.parquet"
)

for path in [
    REFERENCE_ARCHIVE,
    OBS_ARCHIVE,
    ANALYSIS_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

# ------------------------------------------------------------
# 2. LOAD CURRENT ANALYSIS DATASET
# ------------------------------------------------------------

analysis_dataset = pd.read_parquet(ANALYSIS_PATH)

analysis_dataset["patientid"] = pd.to_numeric(
    analysis_dataset["patientid"],
    errors="coerce",
).astype(int)

analysis_dataset["time_zero"] = pd.to_datetime(
    analysis_dataset["time_zero"],
    errors="coerce",
)

patient_ids = set(analysis_dataset["patientid"].tolist())

print("Analysis cohort:", len(analysis_dataset))

# ------------------------------------------------------------
# 3. LOAD VARIABLE REFERENCE
# ------------------------------------------------------------

with tempfile.TemporaryDirectory() as temp_dir:

    temp_path = Path(temp_dir)

    with tarfile.open(
        REFERENCE_ARCHIVE,
        "r:*",
    ) as tar:
        tar.extractall(temp_path, filter="data")

    ref_candidates = list(temp_path.rglob("hirid_variable_reference.csv"))

    if not ref_candidates:
        raise FileNotFoundError("hirid_variable_reference.csv not found.")

    reference = pd.read_csv(
        ref_candidates[0],
        low_memory=False,
    )

reference.columns = [str(col).strip() for col in reference.columns]

reference["ID_numeric"] = pd.to_numeric(
    reference["ID"],
    errors="coerce",
)

reference["Variable_Name_clean"] = reference["Variable Name"].astype(str).str.strip()

# ------------------------------------------------------------
# 4. IDENTIFY BODY-WEIGHT VARIABLES
# ------------------------------------------------------------

weight_candidates = reference.loc[
    reference["Source Table"]
    .astype(str)
    .str.contains(
        "Observation",
        case=False,
        na=False,
    )
    & reference["Variable_Name_clean"].str.contains(
        r"\bweight\b|body weight|patient weight|körpergewicht|gewicht",
        case=False,
        na=False,
        regex=True,
    )
].copy()

print("\n" + "=" * 80)
print("BODY-WEIGHT VARIABLE CANDIDATES")
print("=" * 80)

display(
    weight_candidates[
        [
            "Source Table",
            "ID",
            "Variable Name",
            "Unit",
            "Additional information",
        ]
    ]
)

weight_candidates.to_csv(
    TABLE_DIR / "step04_body_weight_variable_candidates.csv",
    index=False,
)

WEIGHT_IDS = set(weight_candidates["ID_numeric"].dropna().astype(int).tolist())

if not WEIGHT_IDS:
    raise RuntimeError(
        "No body-weight variable was identified. "
        "Review the candidate table before proceeding."
    )

print(
    "\nWeight variable IDs:",
    sorted(WEIGHT_IDS),
)

# ------------------------------------------------------------
# 5. HELPER FUNCTIONS
# ------------------------------------------------------------


def read_parquet_member(
    tar_object,
    member,
    columns=None,
):
    extracted = tar_object.extractfile(member)

    if extracted is None:
        return None

    return pd.read_parquet(
        io.BytesIO(extracted.read()),
        columns=columns,
    )


def detect_column(
    columns,
    candidates,
):
    lookup = {str(col).lower(): col for col in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# ------------------------------------------------------------
# 6. DETECT OBSERVATION SCHEMA
# ------------------------------------------------------------

with tarfile.open(
    OBS_ARCHIVE,
    "r:*",
) as tar:

    obs_members = [
        member
        for member in tar.getmembers()
        if (member.isfile() and member.name.lower().endswith(".parquet"))
    ]

    first_part = read_parquet_member(
        tar,
        obs_members[0],
    )

OBS_PATIENT_COL = detect_column(
    first_part.columns,
    ["patientid", "patient_id"],
)

OBS_VARIABLE_COL = detect_column(
    first_part.columns,
    ["variableid", "variable_id"],
)

OBS_TIME_COL = detect_column(
    first_part.columns,
    [
        "datetime",
        "observationtime",
        "observation_time",
        "recordedat",
        "recorded_at",
    ],
)

OBS_VALUE_COL = detect_column(
    first_part.columns,
    [
        "value",
        "value_numeric",
        "numericvalue",
        "numeric_value",
    ],
)

detected_schema = {
    "patient": OBS_PATIENT_COL,
    "variable": OBS_VARIABLE_COL,
    "time": OBS_TIME_COL,
    "value": OBS_VALUE_COL,
}

print("\nDetected observation schema:")
print(json.dumps(detected_schema, indent=2))

if any(value is None for value in detected_schema.values()):
    raise RuntimeError("Could not identify observation-table schema.")

# ------------------------------------------------------------
# 7. EXTRACT BODY-WEIGHT OBSERVATIONS
# ------------------------------------------------------------

weight_parts = []

with tarfile.open(
    OBS_ARCHIVE,
    "r:*",
) as tar:

    parquet_members = [
        member
        for member in tar.getmembers()
        if (member.isfile() and member.name.lower().endswith(".parquet"))
    ]

    print(
        "\nReading weight observations from",
        len(parquet_members),
        "partitions...",
    )

    for idx, member in enumerate(
        parquet_members,
        start=1,
    ):

        part = read_parquet_member(
            tar,
            member,
            columns=[
                OBS_PATIENT_COL,
                OBS_VARIABLE_COL,
                OBS_TIME_COL,
                OBS_VALUE_COL,
            ],
        )

        if part is None or part.empty:
            continue

        part = part.rename(
            columns={
                OBS_PATIENT_COL: "patientid",
                OBS_VARIABLE_COL: "variableid",
                OBS_TIME_COL: "datetime",
                OBS_VALUE_COL: "weight_raw",
            }
        )

        part["patientid"] = pd.to_numeric(
            part["patientid"],
            errors="coerce",
        )

        part["variableid"] = pd.to_numeric(
            part["variableid"],
            errors="coerce",
        )

        part = part.loc[
            part["patientid"].isin(patient_ids) & part["variableid"].isin(WEIGHT_IDS)
        ].copy()

        if not part.empty:
            weight_parts.append(part)

        if idx % 25 == 0 or idx == len(parquet_members):
            print(f"Weight extraction: processed " f"{idx}/{len(parquet_members)}")

if not weight_parts:
    raise RuntimeError(
        "No body-weight observations were found " "for the analysis cohort."
    )

weights = pd.concat(
    weight_parts,
    ignore_index=True,
)

del weight_parts
gc.collect()

weights["patientid"] = pd.to_numeric(
    weights["patientid"],
    errors="coerce",
)

weights["variableid"] = pd.to_numeric(
    weights["variableid"],
    errors="coerce",
)

weights["datetime"] = pd.to_datetime(
    weights["datetime"],
    errors="coerce",
)

weights["weight_raw"] = pd.to_numeric(
    weights["weight_raw"],
    errors="coerce",
)

weights = weights.dropna(
    subset=[
        "patientid",
        "variableid",
        "datetime",
        "weight_raw",
    ]
)

weights["patientid"] = weights["patientid"].astype(int)

# ------------------------------------------------------------
# 8. LINK WEIGHT TO TIME ZERO
# ------------------------------------------------------------

weights = weights.merge(
    analysis_dataset[
        [
            "patientid",
            "time_zero",
        ]
    ],
    on="patientid",
    how="inner",
)

weights["hours_from_t0"] = (
    weights["datetime"] - weights["time_zero"]
).dt.total_seconds() / 3600

# Prefer measurements from 168 h before to 24 h after t0.
weights = weights.loc[
    weights["hours_from_t0"].between(
        -168,
        24,
        inclusive="both",
    )
].copy()

# ------------------------------------------------------------
# 9. STANDARDIZE WEIGHT UNIT
# ------------------------------------------------------------

weight_reference = weight_candidates[
    [
        "ID_numeric",
        "Unit",
        "Variable Name",
    ]
].copy()

weight_reference = weight_reference.rename(
    columns={
        "ID_numeric": "variableid",
        "Unit": "weight_unit",
        "Variable Name": "weight_variable_name",
    }
)

weights = weights.merge(
    weight_reference,
    on="variableid",
    how="left",
)

weights["weight_kg"] = weights["weight_raw"].copy()

# Convert grams to kilograms if needed.
gram_mask = (
    weights["weight_unit"]
    .astype(str)
    .str.lower()
    .isin(
        [
            "g",
            "gram",
            "grams",
        ]
    )
)

weights.loc[
    gram_mask,
    "weight_kg",
] = (
    weights.loc[
        gram_mask,
        "weight_raw",
    ]
    / 1000
)

# Conservative adult plausibility range.
weights.loc[
    ~weights["weight_kg"].between(
        25,
        300,
        inclusive="both",
    ),
    "weight_kg",
] = np.nan

weights = weights.dropna(subset=["weight_kg"])

# ------------------------------------------------------------
# 10. SELECT WEIGHT CLOSEST TO TIME ZERO
# ------------------------------------------------------------

# Preference order:
# 1. Measurement at or before t0
# 2. Closest absolute time
# 3. Post-t0 measurement when no pre-t0 value exists

weights["is_post_t0"] = (weights["hours_from_t0"] > 0).astype(int)

weights["absolute_hours_from_t0"] = weights["hours_from_t0"].abs()

weight_selected = (
    weights.sort_values(
        [
            "patientid",
            "is_post_t0",
            "absolute_hours_from_t0",
        ]
    )
    .groupby(
        "patientid",
        as_index=False,
    )
    .first()[
        [
            "patientid",
            "weight_kg",
            "datetime",
            "hours_from_t0",
            "variableid",
            "weight_variable_name",
            "weight_unit",
        ]
    ]
    .rename(
        columns={
            "datetime": "weight_datetime",
            "hours_from_t0": "weight_hours_from_t0",
            "variableid": "weight_variableid",
        }
    )
)

# ------------------------------------------------------------
# 11. MERGE AND CALCULATE WEIGHT-NORMALIZED DOSE
# ------------------------------------------------------------

analysis_dataset = analysis_dataset.merge(
    weight_selected,
    on="patientid",
    how="left",
)

analysis_dataset["norepi_ug_kg_min_baseline"] = (
    analysis_dataset["norepi_ug_min_baseline_winsor"] / analysis_dataset["weight_kg"]
)

# Plausibility range; only obvious artifacts are removed.
analysis_dataset.loc[
    ~analysis_dataset["norepi_ug_kg_min_baseline"].between(
        0.001,
        5.0,
        inclusive="both",
    ),
    "norepi_ug_kg_min_baseline",
] = np.nan

positive_weighted_ne = analysis_dataset["norepi_ug_kg_min_baseline"].dropna()

if positive_weighted_ne.empty:
    raise RuntimeError(
        "No valid weight-normalized norepinephrine " "values were obtained."
    )

weighted_upper_p995 = positive_weighted_ne.quantile(0.995)

analysis_dataset["norepi_ug_kg_min_baseline_winsor"] = analysis_dataset[
    "norepi_ug_kg_min_baseline"
].clip(upper=weighted_upper_p995)

analysis_dataset["log_norepi_ug_kg_min_baseline"] = np.log(
    analysis_dataset["norepi_ug_kg_min_baseline_winsor"] + 0.001
)

# ------------------------------------------------------------
# 12. AUDIT WEIGHT AND NORMALIZED DOSE
# ------------------------------------------------------------


def distribution_table(series, variable_name):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return {
        "variable": variable_name,
        "n": int(values.size),
        "percent_available": (100 * values.size / len(analysis_dataset)),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "min": float(values.min()),
        "p01": float(values.quantile(0.01)),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


audit = pd.DataFrame(
    [
        distribution_table(
            analysis_dataset["weight_kg"],
            "weight_kg",
        ),
        distribution_table(
            analysis_dataset["norepi_ug_min_baseline_winsor"],
            "norepi_ug_min_baseline_winsor",
        ),
        distribution_table(
            analysis_dataset["norepi_ug_kg_min_baseline_winsor"],
            "norepi_ug_kg_min_baseline_winsor",
        ),
    ]
)

print("\n" + "=" * 80)
print("WEIGHT AND NOREPINEPHRINE AUDIT")
print("=" * 80)

display(audit)

weight_timing = pd.DataFrame(
    {
        "metric": [
            "Patients with valid weight",
            "Percent with valid weight",
            "Weight measured at or before time zero",
            "Weight measured after time zero",
            "Median weight timing relative to t0, h",
            "IQR weight timing relative to t0, h",
            "Weighted NE 99.5th percentile, µg/kg/min",
        ],
        "value": [
            int(analysis_dataset["weight_kg"].notna().sum()),
            float(100 * analysis_dataset["weight_kg"].notna().mean()),
            int(analysis_dataset["weight_hours_from_t0"].le(0).sum()),
            int(analysis_dataset["weight_hours_from_t0"].gt(0).sum()),
            float(analysis_dataset["weight_hours_from_t0"].median()),
            (
                f"{analysis_dataset['weight_hours_from_t0'].quantile(0.25):.2f}"
                f"–"
                f"{analysis_dataset['weight_hours_from_t0'].quantile(0.75):.2f}"
            ),
            float(weighted_upper_p995),
        ],
    }
)

print("\n" + "=" * 80)
print("WEIGHT TIMING")
print("=" * 80)

display(weight_timing)

# ------------------------------------------------------------
# 13. UPDATE PHENOTYPE COMPLETENESS
# ------------------------------------------------------------

PHENOTYPE_VARIABLES_WEIGHTED = [
    "dbp_baseline",
    "pulse_pressure_baseline",
    "heart_rate_baseline",
    "log_norepi_ug_kg_min_baseline",
    "lactate_baseline",
]

analysis_dataset["n_complete_weighted_phenotype_variables"] = (
    analysis_dataset[PHENOTYPE_VARIABLES_WEIGHTED].notna().sum(axis=1)
)

analysis_dataset["complete_all_weighted_phenotype_variables"] = (
    analysis_dataset[PHENOTYPE_VARIABLES_WEIGHTED].notna().all(axis=1)
)

analysis_dataset["complete_core_weighted_hemodynamics"] = (
    analysis_dataset[
        [
            "dbp_baseline",
            "pulse_pressure_baseline",
            "heart_rate_baseline",
            "log_norepi_ug_kg_min_baseline",
        ]
    ]
    .notna()
    .all(axis=1)
)

counts = pd.DataFrame(
    {
        "criterion": [
            "Broad cohort",
            "Valid body weight",
            "Valid weight-normalized norepinephrine",
            "Complete core weighted phenotype variables",
            "Complete all five weighted phenotype variables",
            "At least 4 of 5 weighted phenotype variables",
        ],
        "n": [
            len(analysis_dataset),
            int(analysis_dataset["weight_kg"].notna().sum()),
            int(analysis_dataset["norepi_ug_kg_min_baseline_winsor"].notna().sum()),
            int(analysis_dataset["complete_core_weighted_hemodynamics"].sum()),
            int(analysis_dataset["complete_all_weighted_phenotype_variables"].sum()),
            int(
                analysis_dataset["n_complete_weighted_phenotype_variables"].ge(4).sum()
            ),
        ],
    }
)

print("\n" + "=" * 80)
print("WEIGHTED PHENOTYPE COUNTS")
print("=" * 80)

display(counts)

# ------------------------------------------------------------
# 14. SAVE
# ------------------------------------------------------------

analysis_dataset.to_parquet(
    DERIVED_DIR / "step04_analysis_dataset_weight_normalized_norepinephrine.parquet",
    index=False,
)

analysis_dataset.to_csv(
    TABLE_DIR / "step04_analysis_dataset_weight_normalized_norepinephrine.csv",
    index=False,
)

weight_selected.to_parquet(
    DERIVED_DIR / "step04_selected_body_weight.parquet",
    index=False,
)

audit.to_csv(
    TABLE_DIR / "step04_weight_and_norepinephrine_audit.csv",
    index=False,
)

weight_timing.to_csv(
    TABLE_DIR / "step04_weight_timing_audit.csv",
    index=False,
)

counts.to_csv(
    TABLE_DIR / "step04_weighted_phenotype_counts.csv",
    index=False,
)

with open(
    LOG_DIR / "step04_weight_normalization_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "weight_ids": sorted(list(WEIGHT_IDS)),
            "valid_weight_n": int(analysis_dataset["weight_kg"].notna().sum()),
            "valid_weight_normalized_norepinephrine_n": int(
                analysis_dataset["norepi_ug_kg_min_baseline_winsor"].notna().sum()
            ),
            "complete_all_weighted_phenotype_variables_n": int(
                analysis_dataset["complete_all_weighted_phenotype_variables"].sum()
            ),
            "weighted_ne_p995": float(weighted_upper_p995),
            "phenotype_variables": PHENOTYPE_VARIABLES_WEIGHTED,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n" + "=" * 80)
print("STEP 04 COMPLETE")
print("=" * 80)

print(
    "\nUpdated dataset saved to:\n"
    f"{DERIVED_DIR / 'step04_analysis_dataset_weight_normalized_norepinephrine.parquet'}"
)
