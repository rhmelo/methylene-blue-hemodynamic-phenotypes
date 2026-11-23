"""Source: original notebook cell 17. Run through run.py."""

from common import display, hirid_source, phase_dir

# ============================================================
# HiRID Methylene Blue Hemodynamic Phenotypes
# STEP 08 — Clinical and incremental construct validation
#
# PURPOSE
#
# 1. Extract HiRID clinical outcomes:
#    - Hospital mortality, if identifiable in general_table
#    - Available length of stay
#    - Baseline and follow-up creatinine
#    - Creatinine-defined AKI
#    - Urine output during the first 24 hours
#
# 2. Compare the macrohemodynamic phenotype with:
#    - Baseline DBP
#    - Baseline norepinephrine dose
#    - Diastolic shock index
#    - Phenotype + baseline norepinephrine
#
# 3. Perform cross-validated model comparisons for:
#    - Subsequent norepinephrine burden
#    - Persistent hypotension
#    - Hospital mortality, if available
#    - Creatinine-defined AKI
#
# IMPORTANT
# - Phenotypes are already frozen.
# - Outcomes are NOT used to redefine the phenotypes.
# - This is construct/prognostic validation, not causal analysis.
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

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from scipy.stats import (
    kruskal,
    chi2_contingency,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

HIRID_DIR = hirid_source()

PROJECT_DIR = phase_dir("hirid")

OUTPUT_DIR = PROJECT_DIR / "outputs"

TABLE_DIR = OUTPUT_DIR / "tables"

LOG_DIR = OUTPUT_DIR / "logs"

DERIVED_DIR = PROJECT_DIR / "derived_data"

for folder in [
    PROJECT_DIR,
    OUTPUT_DIR,
    TABLE_DIR,
    LOG_DIR,
    DERIVED_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

ANALYSIS_PATH = (
    DERIVED_DIR
    / "step07_analysis_dataset_with_prespecified_physiological_phenotypes.parquet"
)

REFERENCE_ARCHIVE = HIRID_DIR / "reference_data.tar.gz"

OBS_ARCHIVE = HIRID_DIR / "observation_tables_parquet.tar.gz"

ID_MAP_PATH = LOG_DIR / "step01c_final_variable_id_map.json"

for required_path in [
    ANALYSIS_PATH,
    REFERENCE_ARCHIVE,
    OBS_ARCHIVE,
    ID_MAP_PATH,
]:

    if not required_path.exists():

        raise FileNotFoundError(f"Required file not found:\n{required_path}")

# ------------------------------------------------------------
# 2. LOAD ANALYSIS DATASET
# ------------------------------------------------------------

analysis = pd.read_parquet(ANALYSIS_PATH)

analysis["patientid"] = pd.to_numeric(
    analysis["patientid"],
    errors="coerce",
).astype(int)

analysis["time_zero"] = pd.to_datetime(
    analysis["time_zero"],
    errors="coerce",
)

COHORT_PATIENTS = set(analysis["patientid"].tolist())

print("=" * 80)
print("INPUT COHORT")
print("=" * 80)

print("Patients:", len(analysis))

print("Phenotype counts:")

display(
    analysis["physiological_phenotype_label"]
    .value_counts(dropna=False)
    .rename_axis("phenotype")
    .reset_index(name="n")
)

# ------------------------------------------------------------
# 3. LOAD VARIABLE MAP
# ------------------------------------------------------------

with open(
    ID_MAP_PATH,
    "r",
    encoding="utf-8",
) as f:

    id_map = json.load(f)

CREATININE_IDS = set(
    map(
        int,
        id_map.get(
            "creatinine_ids",
            [],
        ),
    )
)

URINE_OUTPUT_IDS = set(
    map(
        int,
        id_map.get(
            "urine_output_ids",
            [],
        ),
    )
)

print(
    "\nCreatinine IDs:",
    sorted(CREATININE_IDS),
)

print(
    "Urine-output IDs:",
    sorted(URINE_OUTPUT_IDS),
)

if not CREATININE_IDS:

    print("WARNING: No blood-creatinine IDs were found.")

if not URINE_OUTPUT_IDS:

    print("WARNING: No urine-output IDs were found.")

# ------------------------------------------------------------
# 4. HELPER FUNCTIONS
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

    normalized = {str(column).strip().lower(): column for column in columns}

    for candidate in candidates:

        candidate_lower = candidate.strip().lower()

        if candidate_lower in normalized:

            return normalized[candidate_lower]

    return None


def standardize_binary(
    series,
):
    """
    Converts common binary/status representations to 0/1.
    Unrecognized values become NaN.
    """

    if series is None:
        return pd.Series(dtype=float)

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    unique_numeric = set(numeric.dropna().unique())

    if unique_numeric.issubset({0, 1}):

        return numeric.astype(float)

    text = series.astype(str).str.strip().str.lower()

    death_terms = {
        "dead",
        "deceased",
        "death",
        "died",
        "expired",
        "1",
        "true",
        "yes",
    }

    survival_terms = {
        "alive",
        "survived",
        "survivor",
        "discharged alive",
        "0",
        "false",
        "no",
    }

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    result.loc[text.isin(death_terms)] = 1.0

    result.loc[text.isin(survival_terms)] = 0.0

    return result


# ------------------------------------------------------------
# 5. EXTRACT GENERAL TABLE
# ------------------------------------------------------------

with tempfile.TemporaryDirectory() as temp_dir:

    temp_path = Path(temp_dir)

    with tarfile.open(
        REFERENCE_ARCHIVE,
        "r:*",
    ) as tar:

        tar.extractall(temp_path, filter="data")

    general_candidates = list(temp_path.rglob("general_table.csv"))

    if not general_candidates:

        raise FileNotFoundError(
            "general_table.csv not found " "inside reference_data archive."
        )

    general = pd.read_csv(
        general_candidates[0],
        low_memory=False,
    )

general.columns = [str(column).strip() for column in general.columns]

print("\n" + "=" * 80)
print("GENERAL TABLE SCHEMA")
print("=" * 80)

print(general.columns.tolist())

display(general.head())

PATIENT_COL = detect_column(
    general.columns,
    [
        "patientid",
        "patient_id",
    ],
)

ADMISSION_COL = detect_column(
    general.columns,
    [
        "admissiontime",
        "admission_time",
        "admissiondatetime",
        "admission_datetime",
    ],
)

DISCHARGE_COL = detect_column(
    general.columns,
    [
        "dischargetime",
        "discharge_time",
        "dischargedatetime",
        "discharge_datetime",
    ],
)

MORTALITY_COL = detect_column(
    general.columns,
    [
        "discharge_status",
        "dischargestatus",
        "status",
        "mortality",
        "hospital_mortality",
        "death",
        "died",
    ],
)

AGE_COL = detect_column(
    general.columns,
    [
        "age",
    ],
)

SEX_COL = detect_column(
    general.columns,
    [
        "sex",
        "gender",
    ],
)

detected_general_schema = {
    "patient_column": PATIENT_COL,
    "admission_column": ADMISSION_COL,
    "discharge_column": DISCHARGE_COL,
    "mortality_column": MORTALITY_COL,
    "age_column": AGE_COL,
    "sex_column": SEX_COL,
}

print("\nDetected general-table schema:")

print(
    json.dumps(
        detected_general_schema,
        indent=2,
        ensure_ascii=False,
    )
)

if PATIENT_COL is None:

    raise RuntimeError(
        "Patient identifier could not be " "identified in general_table."
    )

general["patientid"] = pd.to_numeric(
    general[PATIENT_COL],
    errors="coerce",
)

general = general.loc[general["patientid"].isin(COHORT_PATIENTS)].copy()

general = general.dropna(subset=["patientid"])

general["patientid"] = general["patientid"].astype(int)

# ------------------------------------------------------------
# 6. GENERAL OUTCOMES
# ------------------------------------------------------------

general_outcomes = general[["patientid"]].copy()

if AGE_COL is not None:

    general_outcomes["age"] = pd.to_numeric(
        general[AGE_COL],
        errors="coerce",
    )

if SEX_COL is not None:

    general_outcomes["sex"] = general[SEX_COL]

if ADMISSION_COL is not None:

    general_outcomes["admission_time"] = pd.to_datetime(
        general[ADMISSION_COL],
        errors="coerce",
    )

if DISCHARGE_COL is not None:

    general_outcomes["discharge_time"] = pd.to_datetime(
        general[DISCHARGE_COL],
        errors="coerce",
    )

if (
    "admission_time" in general_outcomes.columns
    and "discharge_time" in general_outcomes.columns
):

    general_outcomes["available_length_of_stay_days"] = (
        general_outcomes["discharge_time"] - general_outcomes["admission_time"]
    ).dt.total_seconds() / 86400

    general_outcomes.loc[
        ~general_outcomes["available_length_of_stay_days"].between(
            0,
            365,
            inclusive="both",
        ),
        "available_length_of_stay_days",
    ] = np.nan

if MORTALITY_COL is not None:

    general_outcomes["hospital_mortality"] = standardize_binary(general[MORTALITY_COL])

# ------------------------------------------------------------
# 7. MORTALITY AUDIT
# ------------------------------------------------------------

mortality_audit = pd.DataFrame()

if MORTALITY_COL is not None:

    mortality_audit = (
        general[MORTALITY_COL]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("raw_mortality_value")
        .reset_index(name="n")
    )

    print("\n" + "=" * 80)
    print("RAW MORTALITY VARIABLE AUDIT")
    print("=" * 80)

    display(mortality_audit)

    print(
        "\nMapped hospital mortality availability:",
        int(general_outcomes["hospital_mortality"].notna().sum()),
    )

    print(
        "Mapped deaths:",
        int(general_outcomes["hospital_mortality"].eq(1).sum()),
    )

else:

    print("\nWARNING: A mortality column was not " "automatically identified.")

# ------------------------------------------------------------
# 8. DETECT OBSERVATION SCHEMA
# ------------------------------------------------------------

with tarfile.open(
    OBS_ARCHIVE,
    "r:*",
) as tar:

    observation_members = [
        member
        for member in tar.getmembers()
        if (member.isfile() and member.name.lower().endswith(".parquet"))
    ]

    if not observation_members:

        raise RuntimeError("No observation parquet files found.")

    first_part = read_parquet_member(
        tar,
        observation_members[0],
    )

OBS_PATIENT_COL = detect_column(
    first_part.columns,
    [
        "patientid",
        "patient_id",
    ],
)

OBS_VARIABLE_COL = detect_column(
    first_part.columns,
    [
        "variableid",
        "variable_id",
    ],
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

observation_schema = {
    "patient": OBS_PATIENT_COL,
    "variable": OBS_VARIABLE_COL,
    "time": OBS_TIME_COL,
    "value": OBS_VALUE_COL,
}

print("\nDetected observation schema:")

print(
    json.dumps(
        observation_schema,
        indent=2,
    )
)

if any(value is None for value in observation_schema.values()):

    raise RuntimeError("Observation-table schema could not " "be identified.")

# ------------------------------------------------------------
# 9. EXTRACT CREATININE AND URINE OUTPUT
# ------------------------------------------------------------

wanted_outcome_ids = CREATININE_IDS | URINE_OUTPUT_IDS

outcome_parts = []

if wanted_outcome_ids:

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
            "\nExtracting renal observations from",
            len(parquet_members),
            "partitions...",
        )

        for index, member in enumerate(
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
                    OBS_VALUE_COL: "value",
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
                part["patientid"].isin(COHORT_PATIENTS)
                & part["variableid"].isin(wanted_outcome_ids)
            ].copy()

            if not part.empty:

                outcome_parts.append(
                    part[
                        [
                            "patientid",
                            "variableid",
                            "datetime",
                            "value",
                        ]
                    ]
                )

            if index % 25 == 0 or index == len(parquet_members):

                print(f"Renal extraction: processed " f"{index}/{len(parquet_members)}")

if outcome_parts:

    renal_observations = pd.concat(
        outcome_parts,
        ignore_index=True,
    )

    del outcome_parts
    gc.collect()

else:

    renal_observations = pd.DataFrame(
        columns=[
            "patientid",
            "variableid",
            "datetime",
            "value",
        ]
    )

for column in [
    "patientid",
    "variableid",
    "value",
]:

    renal_observations[column] = pd.to_numeric(
        renal_observations[column],
        errors="coerce",
    )

renal_observations["datetime"] = pd.to_datetime(
    renal_observations["datetime"],
    errors="coerce",
)

renal_observations = renal_observations.dropna(
    subset=[
        "patientid",
        "variableid",
        "datetime",
        "value",
    ]
)

renal_observations["patientid"] = renal_observations["patientid"].astype(int)

renal_observations["variableid"] = renal_observations["variableid"].astype(int)

renal_observations = renal_observations.merge(
    analysis[
        [
            "patientid",
            "time_zero",
            "weight_kg",
        ]
    ],
    on="patientid",
    how="inner",
)

renal_observations["hours_from_t0"] = (
    renal_observations["datetime"] - renal_observations["time_zero"]
).dt.total_seconds() / 3600

# ------------------------------------------------------------
# 10. CREATININE OUTCOMES
# ------------------------------------------------------------

creatinine = renal_observations.loc[
    renal_observations["variableid"].isin(CREATININE_IDS)
].copy()

# HiRID creatinine is expected in µmol/L.
# Conservative plausible range.
creatinine.loc[
    ~creatinine["value"].between(
        15,
        2000,
        inclusive="both",
    ),
    "value",
] = np.nan

creatinine = creatinine.dropna(subset=["value"])

# Baseline:
# closest value from -24 h to +1 h.
creatinine_baseline = creatinine.loc[
    creatinine["hours_from_t0"].between(
        -24,
        1,
        inclusive="both",
    )
].copy()

creatinine_baseline["absolute_hours_from_t0"] = creatinine_baseline[
    "hours_from_t0"
].abs()

creatinine_baseline_selected = (
    creatinine_baseline.sort_values(
        [
            "patientid",
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
            "value",
            "datetime",
            "hours_from_t0",
        ]
    ]
    .rename(
        columns={
            "value": "creatinine_baseline_umol_l",
            "datetime": "creatinine_baseline_time",
            "hours_from_t0": "creatinine_baseline_hours_from_t0",
        }
    )
)

# Follow-up maximum from >1 to 48 h.
creatinine_followup_48h = creatinine.loc[
    creatinine["hours_from_t0"].between(
        1,
        48,
        inclusive="right",
    )
].copy()

creatinine_followup_summary = creatinine_followup_48h.groupby(
    "patientid",
    as_index=False,
).agg(
    creatinine_max_48h_umol_l=(
        "value",
        "max",
    ),
    creatinine_min_48h_umol_l=(
        "value",
        "min",
    ),
    n_creatinine_48h=(
        "value",
        "size",
    ),
)

renal_patient = (
    analysis[
        [
            "patientid",
            "weight_kg",
        ]
    ]
    .merge(
        creatinine_baseline_selected,
        on="patientid",
        how="left",
    )
    .merge(
        creatinine_followup_summary,
        on="patientid",
        how="left",
    )
)

renal_patient["creatinine_delta_48h_umol_l"] = (
    renal_patient["creatinine_max_48h_umol_l"]
    - renal_patient["creatinine_baseline_umol_l"]
)

renal_patient["creatinine_ratio_48h"] = (
    renal_patient["creatinine_max_48h_umol_l"]
    / renal_patient["creatinine_baseline_umol_l"]
)

# Creatinine-only KDIGO-compatible screening:
# - increase >=26.5 µmol/L within 48 h
# OR
# - ratio >=1.5
#
# Since true stable pre-ICU baseline is unavailable,
# this is explicitly labeled creatinine-defined AKI.

renal_patient["creatinine_defined_aki_48h"] = np.where(
    renal_patient["creatinine_baseline_umol_l"].notna()
    & renal_patient["creatinine_max_48h_umol_l"].notna(),
    (
        renal_patient["creatinine_delta_48h_umol_l"].ge(26.5)
        | renal_patient["creatinine_ratio_48h"].ge(1.5)
    ).astype(float),
    np.nan,
)

# Creatinine-stage approximation
renal_patient["creatinine_aki_stage_48h"] = np.nan

valid_creatinine_pair = (
    renal_patient["creatinine_baseline_umol_l"].notna()
    & renal_patient["creatinine_max_48h_umol_l"].notna()
)

renal_patient.loc[
    valid_creatinine_pair,
    "creatinine_aki_stage_48h",
] = 0

stage_1 = valid_creatinine_pair & (
    renal_patient["creatinine_delta_48h_umol_l"].ge(26.5)
    | renal_patient["creatinine_ratio_48h"].between(
        1.5,
        1.999999,
        inclusive="both",
    )
)

stage_2 = valid_creatinine_pair & renal_patient["creatinine_ratio_48h"].between(
    2.0,
    2.999999,
    inclusive="both",
)

stage_3 = valid_creatinine_pair & (
    renal_patient["creatinine_ratio_48h"].ge(3.0)
    | renal_patient["creatinine_max_48h_umol_l"].ge(353.6)
)

renal_patient.loc[
    stage_1,
    "creatinine_aki_stage_48h",
] = 1

renal_patient.loc[
    stage_2,
    "creatinine_aki_stage_48h",
] = 2

renal_patient.loc[
    stage_3,
    "creatinine_aki_stage_48h",
] = 3

# ------------------------------------------------------------
# 11. URINE OUTPUT OUTCOMES
# ------------------------------------------------------------

urine = renal_observations.loc[
    renal_observations["variableid"].isin(URINE_OUTPUT_IDS)
].copy()

# Hourly urine volume is expected in mL/h.
urine.loc[
    ~urine["value"].between(
        0,
        3000,
        inclusive="both",
    ),
    "value",
] = np.nan

urine = urine.dropna(subset=["value"])

urine_0_24h = urine.loc[
    urine["hours_from_t0"].between(
        0,
        24,
        inclusive="both",
    )
].copy()

urine_summary = urine_0_24h.groupby(
    "patientid",
    as_index=False,
).agg(
    urine_output_sum_recorded_0_24h_ml=(
        "value",
        "sum",
    ),
    urine_output_median_ml_h=(
        "value",
        "median",
    ),
    urine_output_mean_ml_h=(
        "value",
        "mean",
    ),
    n_urine_records_0_24h=(
        "value",
        "size",
    ),
)

urine_summary = urine_summary.merge(
    analysis[
        [
            "patientid",
            "weight_kg",
        ]
    ],
    on="patientid",
    how="left",
)

urine_summary["mean_urine_ml_kg_h"] = (
    urine_summary["urine_output_mean_ml_h"] / urine_summary["weight_kg"]
)

# Require at least 6 hourly records for a meaningful
# supportive urine-output summary.
urine_summary["urine_data_sufficient"] = urine_summary["n_urine_records_0_24h"].ge(6)

urine_summary["oliguria_mean_lt_0_5_ml_kg_h"] = np.where(
    urine_summary["urine_data_sufficient"]
    & urine_summary["mean_urine_ml_kg_h"].notna(),
    urine_summary["mean_urine_ml_kg_h"].lt(0.5).astype(float),
    np.nan,
)

renal_patient = renal_patient.merge(
    urine_summary[
        [
            "patientid",
            "urine_output_sum_recorded_0_24h_ml",
            "urine_output_median_ml_h",
            "urine_output_mean_ml_h",
            "n_urine_records_0_24h",
            "mean_urine_ml_kg_h",
            "urine_data_sufficient",
            "oliguria_mean_lt_0_5_ml_kg_h",
        ]
    ],
    on="patientid",
    how="left",
)

# ------------------------------------------------------------
# 12. MERGE ALL OUTCOMES
# ------------------------------------------------------------

analysis_outcomes = analysis.merge(
    general_outcomes,
    on="patientid",
    how="left",
).merge(
    renal_patient.drop(
        columns=[
            "weight_kg",
        ],
        errors="ignore",
    ),
    on="patientid",
    how="left",
)

# ------------------------------------------------------------
# 13. ADD PHYSIOLOGICAL VALIDATION OUTCOMES
# ------------------------------------------------------------

# Persistent hypotension:
# >=60% of available hourly measurements with DBP <50
# during hours >1 to 6.

analysis_outcomes["persistent_dbp_lt_50_h1_h6"] = np.where(
    analysis_outcomes["proportion_hours_dbp_lt_50_h1_h6_validation"].notna(),
    analysis_outcomes["proportion_hours_dbp_lt_50_h1_h6_validation"]
    .ge(0.60)
    .astype(float),
    np.nan,
)

analysis_outcomes["persistent_map_lt_65_h1_h6"] = np.where(
    analysis_outcomes["proportion_hours_map_lt_65_h1_h6_validation"].notna(),
    analysis_outcomes["proportion_hours_map_lt_65_h1_h6_validation"]
    .ge(0.60)
    .astype(float),
    np.nan,
)

analysis_outcomes["log_norepi_auc_h1_h6"] = np.log1p(
    analysis_outcomes["norepi_ug_kg_min_auc_h1_h6"]
)

analysis_outcomes["log_available_length_of_stay_days"] = np.log1p(
    analysis_outcomes.get(
        "available_length_of_stay_days",
        pd.Series(
            np.nan,
            index=analysis_outcomes.index,
        ),
    )
)

# ------------------------------------------------------------
# 14. OUTCOME AVAILABILITY
# ------------------------------------------------------------

OUTCOME_CANDIDATES = [
    "norepi_ug_kg_min_auc_h1_h6",
    "log_norepi_auc_h1_h6",
    "persistent_dbp_lt_50_h1_h6",
    "persistent_map_lt_65_h1_h6",
    "hospital_mortality",
    "available_length_of_stay_days",
    "creatinine_defined_aki_48h",
    "creatinine_aki_stage_48h",
    "creatinine_delta_48h_umol_l",
    "mean_urine_ml_kg_h",
    "oliguria_mean_lt_0_5_ml_kg_h",
]

outcome_availability_rows = []

for outcome in OUTCOME_CANDIDATES:

    if outcome not in analysis_outcomes.columns:
        continue

    available = analysis_outcomes[outcome].notna()

    outcome_availability_rows.append(
        {
            "outcome": outcome,
            "n_available": int(available.sum()),
            "n_total": int(len(analysis_outcomes)),
            "percent_available": float(100 * available.mean()),
            "n_events_or_median": (
                int(
                    analysis_outcomes.loc[
                        available,
                        outcome,
                    ]
                    .eq(1)
                    .sum()
                )
                if outcome
                in [
                    "persistent_dbp_lt_50_h1_h6",
                    "persistent_map_lt_65_h1_h6",
                    "hospital_mortality",
                    "creatinine_defined_aki_48h",
                    "oliguria_mean_lt_0_5_ml_kg_h",
                ]
                else (
                    float(
                        analysis_outcomes.loc[
                            available,
                            outcome,
                        ].median()
                    )
                    if available.any()
                    else np.nan
                )
            ),
        }
    )

outcome_availability = pd.DataFrame(outcome_availability_rows)

print("\n" + "=" * 80)
print("OUTCOME AVAILABILITY")
print("=" * 80)

display(outcome_availability)

# ------------------------------------------------------------
# 15. OUTCOMES BY PHENOTYPE
# ------------------------------------------------------------

CONTINUOUS_OUTCOMES = [
    "norepi_ug_kg_min_auc_h1_h6",
    "norepi_ug_kg_min_median_h1_h6",
    "creatinine_delta_48h_umol_l",
    "creatinine_ratio_48h",
    "mean_urine_ml_kg_h",
    "available_length_of_stay_days",
]

BINARY_OUTCOMES = [
    "persistent_dbp_lt_50_h1_h6",
    "persistent_map_lt_65_h1_h6",
    "hospital_mortality",
    "creatinine_defined_aki_48h",
    "oliguria_mean_lt_0_5_ml_kg_h",
]

continuous_by_phenotype_rows = []

for outcome in CONTINUOUS_OUTCOMES:

    if outcome not in analysis_outcomes.columns:
        continue

    for phenotype, group in analysis_outcomes.dropna(
        subset=["physiological_phenotype"]
    ).groupby("physiological_phenotype"):

        values = pd.to_numeric(
            group[outcome],
            errors="coerce",
        ).dropna()

        continuous_by_phenotype_rows.append(
            {
                "outcome": outcome,
                "phenotype": int(phenotype),
                "phenotype_label": group["physiological_phenotype_label"].iloc[0],
                "n": int(values.size),
                "median": (float(values.median()) if not values.empty else np.nan),
                "q25": (float(values.quantile(0.25)) if not values.empty else np.nan),
                "q75": (float(values.quantile(0.75)) if not values.empty else np.nan),
                "mean": (float(values.mean()) if not values.empty else np.nan),
            }
        )

continuous_by_phenotype = pd.DataFrame(continuous_by_phenotype_rows)

binary_by_phenotype_rows = []

for outcome in BINARY_OUTCOMES:

    if outcome not in analysis_outcomes.columns:
        continue

    for phenotype, group in analysis_outcomes.dropna(
        subset=["physiological_phenotype"]
    ).groupby("physiological_phenotype"):

        values = pd.to_numeric(
            group[outcome],
            errors="coerce",
        ).dropna()

        events = int(values.eq(1).sum())

        binary_by_phenotype_rows.append(
            {
                "outcome": outcome,
                "phenotype": int(phenotype),
                "phenotype_label": group["physiological_phenotype_label"].iloc[0],
                "n": int(values.size),
                "events": events,
                "percent": (
                    float(100 * events / values.size) if values.size > 0 else np.nan
                ),
            }
        )

binary_by_phenotype = pd.DataFrame(binary_by_phenotype_rows)

print("\n" + "=" * 80)
print("CONTINUOUS OUTCOMES BY PHENOTYPE")
print("=" * 80)

display(continuous_by_phenotype)

print("\n" + "=" * 80)
print("BINARY OUTCOMES BY PHENOTYPE")
print("=" * 80)

display(binary_by_phenotype)

# ------------------------------------------------------------
# 16. GLOBAL UNADJUSTED TESTS
# ------------------------------------------------------------

global_outcome_tests = []

for outcome in CONTINUOUS_OUTCOMES:

    if outcome not in analysis_outcomes.columns:
        continue

    groups = []

    for phenotype in [
        1,
        2,
        3,
    ]:

        values = pd.to_numeric(
            analysis_outcomes.loc[
                analysis_outcomes["physiological_phenotype"].eq(phenotype),
                outcome,
            ],
            errors="coerce",
        ).dropna()

        if not values.empty:
            groups.append(values)

    if len(groups) == 3:

        statistic, p_value = kruskal(*groups)

        global_outcome_tests.append(
            {
                "outcome": outcome,
                "test": "Kruskal-Wallis",
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

for outcome in BINARY_OUTCOMES:

    if outcome not in analysis_outcomes.columns:
        continue

    temporary = analysis_outcomes.loc[
        analysis_outcomes["physiological_phenotype"].notna()
        & analysis_outcomes[outcome].notna(),
        [
            "physiological_phenotype",
            outcome,
        ],
    ].copy()

    contingency = pd.crosstab(
        temporary["physiological_phenotype"],
        temporary[outcome],
    )

    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:

        statistic, p_value, _, _ = chi2_contingency(contingency)

        global_outcome_tests.append(
            {
                "outcome": outcome,
                "test": "Chi-square",
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

global_outcome_tests = pd.DataFrame(global_outcome_tests)

# ------------------------------------------------------------
# 17. MODEL SPECIFICATIONS
# ------------------------------------------------------------

MODEL_SPECIFICATIONS = {
    "DBP_only": {
        "numeric": [
            "dbp_baseline",
        ],
        "categorical": [],
    },
    "Norepinephrine_only": {
        "numeric": [
            "norepi_ug_kg_min_baseline_winsor",
        ],
        "categorical": [],
    },
    "DSI_only": {
        "numeric": [
            "diastolic_shock_index",
        ],
        "categorical": [],
    },
    "Phenotype_only": {
        "numeric": [],
        "categorical": [
            "physiological_phenotype",
        ],
    },
    "Phenotype_plus_norepinephrine": {
        "numeric": [
            "norepi_ug_kg_min_baseline_winsor",
        ],
        "categorical": [
            "physiological_phenotype",
        ],
    },
    "DBP_PP_HR": {
        "numeric": [
            "dbp_baseline",
            "pulse_pressure_baseline",
            "heart_rate_baseline",
        ],
        "categorical": [],
    },
    "Phenotype_plus_DBP_plus_norepinephrine": {
        "numeric": [
            "dbp_baseline",
            "norepi_ug_kg_min_baseline_winsor",
        ],
        "categorical": [
            "physiological_phenotype",
        ],
    },
}

# ------------------------------------------------------------
# 18. PIPELINE BUILDERS
# ------------------------------------------------------------


def build_preprocessor(
    numeric_features,
    categorical_features,
):

    transformers = []

    if numeric_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        drop="first",
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def evaluate_binary_model(
    dataframe,
    outcome,
    model_name,
    specification,
):

    numeric_features = [
        feature for feature in specification["numeric"] if feature in dataframe.columns
    ]

    categorical_features = [
        feature
        for feature in specification["categorical"]
        if feature in dataframe.columns
    ]

    predictors = numeric_features + categorical_features

    if not predictors:
        return None

    temporary = dataframe[predictors + [outcome]].copy()

    temporary[outcome] = pd.to_numeric(
        temporary[outcome],
        errors="coerce",
    )

    temporary = temporary.dropna(subset=[outcome])

    if len(temporary) < 100 or temporary[outcome].nunique() < 2:

        return None

    y = temporary[outcome].astype(int)

    event_count = int(y.sum())

    non_event_count = int(len(y) - event_count)

    minimum_class = min(
        event_count,
        non_event_count,
    )

    if minimum_class < 10:
        return None

    n_splits = min(
        5,
        minimum_class,
    )

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=5,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    solver="liblinear",
                    class_weight=None,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # cross_val_predict does not support repeated CV cleanly
    # because each patient appears multiple times.
    # Therefore, evaluate fold-by-fold and aggregate metrics.

    fold_metrics = []

    for train_index, test_index in cv.split(
        temporary[predictors],
        y,
    ):

        X_train = temporary.iloc[train_index][predictors]

        X_test = temporary.iloc[test_index][predictors]

        y_train = y.iloc[train_index]

        y_test = y.iloc[test_index]

        model.fit(
            X_train,
            y_train,
        )

        probability = model.predict_proba(X_test)[:, 1]

        prediction = (probability >= 0.5).astype(int)

        fold_metrics.append(
            {
                "auc": roc_auc_score(
                    y_test,
                    probability,
                ),
                "brier": brier_score_loss(
                    y_test,
                    probability,
                ),
                "accuracy": accuracy_score(
                    y_test,
                    prediction,
                ),
            }
        )

    fold_metrics = pd.DataFrame(fold_metrics)

    return {
        "outcome": outcome,
        "model": model_name,
        "n": int(len(temporary)),
        "events": event_count,
        "event_percent": float(100 * event_count / len(temporary)),
        "auc_mean": float(fold_metrics["auc"].mean()),
        "auc_sd": float(fold_metrics["auc"].std()),
        "auc_p05": float(fold_metrics["auc"].quantile(0.05)),
        "auc_p95": float(fold_metrics["auc"].quantile(0.95)),
        "brier_mean": float(fold_metrics["brier"].mean()),
        "accuracy_mean": float(fold_metrics["accuracy"].mean()),
        "numeric_features": ", ".join(numeric_features),
        "categorical_features": ", ".join(categorical_features),
    }


def evaluate_continuous_model(
    dataframe,
    outcome,
    model_name,
    specification,
):

    numeric_features = [
        feature for feature in specification["numeric"] if feature in dataframe.columns
    ]

    categorical_features = [
        feature
        for feature in specification["categorical"]
        if feature in dataframe.columns
    ]

    predictors = numeric_features + categorical_features

    if not predictors:
        return None

    temporary = dataframe[predictors + [outcome]].copy()

    temporary[outcome] = pd.to_numeric(
        temporary[outcome],
        errors="coerce",
    )

    temporary = temporary.dropna(subset=[outcome])

    if len(temporary) < 100:

        return None

    y = temporary[outcome].astype(float)

    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                Ridge(alpha=1.0),
            ),
        ]
    )

    fold_metrics = []

    for train_index, test_index in cv.split(
        temporary[predictors],
        y,
    ):

        X_train = temporary.iloc[train_index][predictors]

        X_test = temporary.iloc[test_index][predictors]

        y_train = y.iloc[train_index]

        y_test = y.iloc[test_index]

        model.fit(
            X_train,
            y_train,
        )

        prediction = model.predict(X_test)

        fold_metrics.append(
            {
                "r2": r2_score(
                    y_test,
                    prediction,
                ),
                "mae": mean_absolute_error(
                    y_test,
                    prediction,
                ),
                "rmse": np.sqrt(
                    mean_squared_error(
                        y_test,
                        prediction,
                    )
                ),
            }
        )

    fold_metrics = pd.DataFrame(fold_metrics)

    return {
        "outcome": outcome,
        "model": model_name,
        "n": int(len(temporary)),
        "r2_mean": float(fold_metrics["r2"].mean()),
        "r2_sd": float(fold_metrics["r2"].std()),
        "r2_p05": float(fold_metrics["r2"].quantile(0.05)),
        "r2_p95": float(fold_metrics["r2"].quantile(0.95)),
        "mae_mean": float(fold_metrics["mae"].mean()),
        "rmse_mean": float(fold_metrics["rmse"].mean()),
        "numeric_features": ", ".join(numeric_features),
        "categorical_features": ", ".join(categorical_features),
    }


# ------------------------------------------------------------
# 19. BINARY MODEL COMPARISONS
# ------------------------------------------------------------

BINARY_MODEL_OUTCOMES = [
    "persistent_dbp_lt_50_h1_h6",
    "persistent_map_lt_65_h1_h6",
    "hospital_mortality",
    "creatinine_defined_aki_48h",
]

binary_model_rows = []

for outcome in BINARY_MODEL_OUTCOMES:

    if outcome not in analysis_outcomes.columns:
        continue

    for model_name, specification in MODEL_SPECIFICATIONS.items():

        result = evaluate_binary_model(
            dataframe=analysis_outcomes,
            outcome=outcome,
            model_name=model_name,
            specification=specification,
        )

        if result is not None:

            binary_model_rows.append(result)

binary_model_performance = pd.DataFrame(binary_model_rows)

print("\n" + "=" * 80)
print("CROSS-VALIDATED BINARY MODEL PERFORMANCE")
print("=" * 80)

display(
    binary_model_performance.sort_values(
        [
            "outcome",
            "auc_mean",
        ],
        ascending=[
            True,
            False,
        ],
    )
)

# ------------------------------------------------------------
# 20. CONTINUOUS MODEL COMPARISONS
# ------------------------------------------------------------

CONTINUOUS_MODEL_OUTCOMES = [
    "log_norepi_auc_h1_h6",
    "creatinine_delta_48h_umol_l",
    "log_available_length_of_stay_days",
]

continuous_model_rows = []

for outcome in CONTINUOUS_MODEL_OUTCOMES:

    if outcome not in analysis_outcomes.columns:
        continue

    if analysis_outcomes[outcome].notna().sum() < 100:

        continue

    for model_name, specification in MODEL_SPECIFICATIONS.items():

        result = evaluate_continuous_model(
            dataframe=analysis_outcomes,
            outcome=outcome,
            model_name=model_name,
            specification=specification,
        )

        if result is not None:

            continuous_model_rows.append(result)

continuous_model_performance = pd.DataFrame(continuous_model_rows)

print("\n" + "=" * 80)
print("CROSS-VALIDATED CONTINUOUS MODEL PERFORMANCE")
print("=" * 80)

display(
    continuous_model_performance.sort_values(
        [
            "outcome",
            "r2_mean",
        ],
        ascending=[
            True,
            False,
        ],
    )
)

# ------------------------------------------------------------
# 21. INCREMENTAL PHENOTYPE VALUE
# ------------------------------------------------------------

incremental_rows = []

# Compare phenotype+norepi against norepi alone
for outcome in BINARY_MODEL_OUTCOMES:

    if binary_model_performance.empty:
        continue

    outcome_data = binary_model_performance.loc[
        binary_model_performance["outcome"].eq(outcome)
    ]

    base = outcome_data.loc[outcome_data["model"].eq("Norepinephrine_only")]

    expanded = outcome_data.loc[
        outcome_data["model"].eq("Phenotype_plus_norepinephrine")
    ]

    if not base.empty and not expanded.empty:

        incremental_rows.append(
            {
                "outcome": outcome,
                "metric": "AUC",
                "base_model": "Norepinephrine_only",
                "expanded_model": "Phenotype_plus_norepinephrine",
                "base_value": float(base["auc_mean"].iloc[0]),
                "expanded_value": float(expanded["auc_mean"].iloc[0]),
                "absolute_improvement": float(
                    expanded["auc_mean"].iloc[0] - base["auc_mean"].iloc[0]
                ),
            }
        )

for outcome in CONTINUOUS_MODEL_OUTCOMES:

    if continuous_model_performance.empty:
        continue

    outcome_data = continuous_model_performance.loc[
        continuous_model_performance["outcome"].eq(outcome)
    ]

    base = outcome_data.loc[outcome_data["model"].eq("Norepinephrine_only")]

    expanded = outcome_data.loc[
        outcome_data["model"].eq("Phenotype_plus_norepinephrine")
    ]

    if not base.empty and not expanded.empty:

        incremental_rows.append(
            {
                "outcome": outcome,
                "metric": "Cross-validated R2",
                "base_model": "Norepinephrine_only",
                "expanded_model": "Phenotype_plus_norepinephrine",
                "base_value": float(base["r2_mean"].iloc[0]),
                "expanded_value": float(expanded["r2_mean"].iloc[0]),
                "absolute_improvement": float(
                    expanded["r2_mean"].iloc[0] - base["r2_mean"].iloc[0]
                ),
            }
        )

incremental_performance = pd.DataFrame(incremental_rows)

print("\n" + "=" * 80)
print("INCREMENTAL VALUE OF PHENOTYPE BEYOND NOREPINEPHRINE")
print("=" * 80)

display(incremental_performance)

# ------------------------------------------------------------
# 22. SAVE OUTPUTS
# ------------------------------------------------------------

analysis_outcomes.to_parquet(
    DERIVED_DIR / "step08_HiRID_final_dataset_with_clinical_outcomes.parquet",
    index=False,
)

analysis_outcomes.to_csv(
    TABLE_DIR / "step08_HiRID_final_dataset_with_clinical_outcomes.csv",
    index=False,
)

general_outcomes.to_csv(
    TABLE_DIR / "step08_general_table_outcomes.csv",
    index=False,
)

renal_patient.to_csv(
    TABLE_DIR / "step08_renal_outcomes.csv",
    index=False,
)

outcome_availability.to_csv(
    TABLE_DIR / "step08_outcome_availability.csv",
    index=False,
)

continuous_by_phenotype.to_csv(
    TABLE_DIR / "step08_continuous_outcomes_by_phenotype.csv",
    index=False,
)

binary_by_phenotype.to_csv(
    TABLE_DIR / "step08_binary_outcomes_by_phenotype.csv",
    index=False,
)

global_outcome_tests.to_csv(
    TABLE_DIR / "step08_global_outcome_tests.csv",
    index=False,
)

binary_model_performance.to_csv(
    TABLE_DIR / "step08_binary_model_performance.csv",
    index=False,
)

continuous_model_performance.to_csv(
    TABLE_DIR / "step08_continuous_model_performance.csv",
    index=False,
)

incremental_performance.to_csv(
    TABLE_DIR / "step08_incremental_phenotype_performance.csv",
    index=False,
)

if not mortality_audit.empty:

    mortality_audit.to_csv(
        TABLE_DIR / "step08_raw_mortality_audit.csv",
        index=False,
    )

# ------------------------------------------------------------
# 23. SAVE ANALYSIS SUMMARY
# ------------------------------------------------------------

summary = {
    "cohort_n": int(len(analysis_outcomes)),
    "general_table_schema": detected_general_schema,
    "observation_schema": observation_schema,
    "mortality_available_n": (
        int(analysis_outcomes["hospital_mortality"].notna().sum())
        if "hospital_mortality" in analysis_outcomes.columns
        else 0
    ),
    "hospital_deaths_n": (
        int(analysis_outcomes["hospital_mortality"].eq(1).sum())
        if "hospital_mortality" in analysis_outcomes.columns
        else 0
    ),
    "creatinine_pairs_n": int(
        analysis_outcomes["creatinine_defined_aki_48h"].notna().sum()
    ),
    "creatinine_defined_aki_n": int(
        analysis_outcomes["creatinine_defined_aki_48h"].eq(1).sum()
    ),
    "urine_output_sufficient_n": (
        int(analysis_outcomes["urine_data_sufficient"].fillna(False).sum())
        if "urine_data_sufficient" in analysis_outcomes.columns
        else 0
    ),
    "models_compared": list(MODEL_SPECIFICATIONS.keys()),
    "phenotype_redefined": False,
    "purpose": (
        "Construct and prognostic validation " "before external MIMIC-IV application"
    ),
}

with open(
    LOG_DIR / "step08_HiRID_clinical_validation_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )

# ------------------------------------------------------------
# 24. FINAL DISPLAY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 08 COMPLETE")
print("=" * 80)

print(
    "\nFinal HiRID outcome dataset:\n"
    f"{DERIVED_DIR / 'step08_HiRID_final_dataset_with_clinical_outcomes.parquet'}"
)

print("\nOutcome availability:\n" f"{TABLE_DIR / 'step08_outcome_availability.csv'}")

print(
    "\nBinary model performance:\n"
    f"{TABLE_DIR / 'step08_binary_model_performance.csv'}"
)

print(
    "\nContinuous model performance:\n"
    f"{TABLE_DIR / 'step08_continuous_model_performance.csv'}"
)

print(
    "\nIncremental phenotype performance:\n"
    f"{TABLE_DIR / 'step08_incremental_phenotype_performance.csv'}"
)

print(
    "\nAfter reviewing these outputs, the final "
    "HiRID manuscript package can be generated."
)
