"""Source: original notebook cell 9. Run through run.py."""

from common import display, hirid_source, phase_dir

# ============================================================
# HiRID Methylene Blue Hemodynamic Phenotypes
# STEP 02 — Probable septic shock feasibility cohort
#
# Broad probable septic shock:
# - Continuous norepinephrine infusion for >=60 minutes
# - IV antibiotic around shock onset: -24 h to +6 h
# - At least 2 IV antibiotic administrations
# - Antibiotic treatment duration >24 hours
# - Invasive SBP, DBP and MAP near time zero
# - Heart rate near time zero
# - Approximately 6 hours of invasive BP follow-up
#
# Strict probable septic shock:
# - All broad criteria
# - Blood lactate >2 mmol/L within +/-6 hours
#
# Time zero:
# Start of the first qualifying continuous norepinephrine episode
# ============================================================

import pandas as pd
import numpy as np
import tarfile
import json
import gc
import io
import warnings

warnings.filterwarnings("ignore")

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
    folder.mkdir(parents=True, exist_ok=True)

PHARMA_ARCHIVE = HIRID_DIR / "pharma_records_parquet.tar.gz"

OBS_ARCHIVE = HIRID_DIR / "observation_tables_parquet.tar.gz"

ID_MAP_PATH = LOG_DIR / "step01c_final_variable_id_map.json"

for required_path in [
    PHARMA_ARCHIVE,
    OBS_ARCHIVE,
    ID_MAP_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required file not found:\n{required_path}")

# ------------------------------------------------------------
# 2. LOAD FINAL VARIABLE MAP
# ------------------------------------------------------------

with open(
    ID_MAP_PATH,
    "r",
    encoding="utf-8",
) as f:
    id_map = json.load(f)

NOREPI_IDS_ALL = set(
    map(
        int,
        id_map["norepinephrine_ids"],
    )
)

# Bolus norepinephrine will not define a continuous episode.
NOREPI_BOLUS_IDS = {1000658}

NOREPI_INFUSION_IDS = NOREPI_IDS_ALL - NOREPI_BOLUS_IDS

ANTIBIOTIC_IDS = set(
    map(
        int,
        id_map["antibiotic_ids"],
    )
)

LACTATE_IDS = set(
    map(
        int,
        id_map["lactate_ids"],
    )
)

SBP_IDS = set(
    map(
        int,
        id_map["sbp_ids"],
    )
)

DBP_IDS = set(
    map(
        int,
        id_map["dbp_ids"],
    )
)

MAP_IDS = set(
    map(
        int,
        id_map["map_ids"],
    )
)

HR_IDS = set(
    map(
        int,
        id_map["heart_rate_ids"],
    )
)

print("=" * 80)
print("VARIABLE IDs")
print("=" * 80)

print(
    "Continuous norepinephrine IDs:",
    sorted(NOREPI_INFUSION_IDS),
)

print(
    "Excluded norepinephrine bolus IDs:",
    sorted(NOREPI_BOLUS_IDS),
)

print(
    "Number of antibiotic IDs:",
    len(ANTIBIOTIC_IDS),
)

print("Lactate IDs:", sorted(LACTATE_IDS))
print("Invasive SBP IDs:", sorted(SBP_IDS))
print("Invasive DBP IDs:", sorted(DBP_IDS))
print("Invasive MAP IDs:", sorted(MAP_IDS))
print("Heart-rate IDs:", sorted(HR_IDS))

if not NOREPI_INFUSION_IDS:
    raise RuntimeError("No continuous norepinephrine IDs were identified.")

if not ANTIBIOTIC_IDS:
    raise RuntimeError("No antibiotic IDs were identified.")

if not LACTATE_IDS:
    raise RuntimeError("No blood lactate IDs were identified.")

# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------


def read_parquet_member(
    tar_object,
    member,
    columns=None,
):
    """
    Read one parquet file directly from a TAR archive.
    """
    extracted = tar_object.extractfile(member)

    if extracted is None:
        return None

    raw_bytes = extracted.read()

    return pd.read_parquet(
        io.BytesIO(raw_bytes),
        columns=columns,
    )


def detect_column(
    columns,
    candidates,
):
    """
    Identify a column from a list of possible names.
    """
    lookup = {str(col).lower(): col for col in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# ------------------------------------------------------------
# 4. READ RELEVANT PHARMA RECORDS
# ------------------------------------------------------------

wanted_pharma_ids = NOREPI_IDS_ALL | ANTIBIOTIC_IDS

pharma_parts = []

with tarfile.open(
    PHARMA_ARCHIVE,
    "r:*",
) as tar:

    parquet_members = [
        member
        for member in tar.getmembers()
        if (member.isfile() and member.name.lower().endswith(".parquet"))
    ]

    print(
        "\nNumber of pharma partitions:",
        len(parquet_members),
    )

    for idx, member in enumerate(
        parquet_members,
        start=1,
    ):

        part = read_parquet_member(
            tar,
            member,
            columns=[
                "patientid",
                "pharmaid",
                "givenat",
                "givendose",
                "cumulativedose",
                "fluidamount_calc",
                "cumulfluidamount_calc",
                "doseunit",
                "route",
                "infusionid",
                "typeid",
                "subtypeid",
                "recordstatus",
            ],
        )

        if part is None or part.empty:
            continue

        part = part.loc[part["pharmaid"].isin(wanted_pharma_ids)].copy()

        if not part.empty:
            pharma_parts.append(part)

        if idx % 25 == 0 or idx == len(parquet_members):
            print(f"Pharma: processed " f"{idx}/{len(parquet_members)}")

if not pharma_parts:
    raise RuntimeError("No relevant pharma records were found.")

pharma = pd.concat(
    pharma_parts,
    ignore_index=True,
)

del pharma_parts
gc.collect()

pharma["givenat"] = pd.to_datetime(
    pharma["givenat"],
    errors="coerce",
)

pharma["pharmaid"] = pd.to_numeric(
    pharma["pharmaid"],
    errors="coerce",
)

pharma["patientid"] = pd.to_numeric(
    pharma["patientid"],
    errors="coerce",
)

pharma = pharma.dropna(
    subset=[
        "patientid",
        "pharmaid",
        "givenat",
    ]
)

pharma["patientid"] = pharma["patientid"].astype(int)

pharma["pharmaid"] = pharma["pharmaid"].astype(int)

print("\nRelevant pharma records:", len(pharma))

print(
    "Patients with relevant pharma records:",
    pharma["patientid"].nunique(),
)

# ------------------------------------------------------------
# 5. SEPARATE NOREPINEPHRINE AND ANTIBIOTICS
# ------------------------------------------------------------

norepi = pharma.loc[pharma["pharmaid"].isin(NOREPI_INFUSION_IDS)].copy()

# Continuous infusion routes only.
norepi = norepi.loc[
    norepi["route"]
    .astype(str)
    .str.lower()
    .isin(
        [
            "iv-inf",
            "cv-inf",
        ]
    )
].copy()

antibiotics = pharma.loc[pharma["pharmaid"].isin(ANTIBIOTIC_IDS)].copy()

# Systemic IV or central venous routes only.
antibiotics = antibiotics.loc[
    antibiotics["route"]
    .astype(str)
    .str.lower()
    .isin(
        [
            "iv-inf",
            "iv-inj",
            "cv-inf",
            "cv-inj",
        ]
    )
].copy()

print(
    "\nPatients with continuous norepinephrine records:",
    norepi["patientid"].nunique(),
)

print(
    "Patients with IV antibiotic records:",
    antibiotics["patientid"].nunique(),
)

if norepi.empty:
    raise RuntimeError(
        "No norepinephrine infusion records remained " "after route filtering."
    )

# ------------------------------------------------------------
# 6. IDENTIFY CONTINUOUS NOREPINEPHRINE EPISODES
# ------------------------------------------------------------

# HiRID often records infusion updates approximately every
# 15 minutes. A gap >45 minutes defines a new episode.
MAX_GAP = pd.Timedelta(minutes=45)

MIN_DURATION = pd.Timedelta(minutes=60)

norepi = norepi.sort_values(
    [
        "patientid",
        "infusionid",
        "givenat",
    ]
).copy()

# Vectorized episode construction.
norepi["previous_time"] = norepi.groupby(
    [
        "patientid",
        "infusionid",
    ]
)[
    "givenat"
].shift(1)

norepi["time_gap"] = norepi["givenat"] - norepi["previous_time"]

norepi["new_episode"] = norepi["previous_time"].isna() | norepi["time_gap"].gt(MAX_GAP)

norepi["episode_number"] = norepi.groupby(
    [
        "patientid",
        "infusionid",
    ]
)["new_episode"].cumsum()

norepi["givendose_numeric"] = pd.to_numeric(
    norepi["givendose"],
    errors="coerce",
)

norepi_episodes = norepi.groupby(
    [
        "patientid",
        "infusionid",
        "episode_number",
    ],
    as_index=False,
    dropna=False,
).agg(
    norepi_start=(
        "givenat",
        "min",
    ),
    norepi_end=(
        "givenat",
        "max",
    ),
    n_records=(
        "givenat",
        "size",
    ),
    norepi_pharmaid=(
        "pharmaid",
        "first",
    ),
    main_doseunit=(
        "doseunit",
        lambda x: x.astype(str).mode().iloc[0] if not x.empty else np.nan,
    ),
    main_route=(
        "route",
        lambda x: x.astype(str).mode().iloc[0] if not x.empty else np.nan,
    ),
    first_given_dose=(
        "givendose_numeric",
        "first",
    ),
    median_given_dose=(
        "givendose_numeric",
        "median",
    ),
    maximum_given_dose=(
        "givendose_numeric",
        "max",
    ),
)

norepi_episodes["duration_hours"] = (
    norepi_episodes["norepi_end"] - norepi_episodes["norepi_start"]
).dt.total_seconds() / 3600

norepi_episodes = norepi_episodes.loc[
    norepi_episodes["duration_hours"].ge(MIN_DURATION.total_seconds() / 3600)
].copy()

if norepi_episodes.empty:
    raise RuntimeError(
        "No continuous norepinephrine episodes "
        "lasting at least 60 minutes were identified."
    )

print(
    "\nQualifying norepinephrine episodes:",
    len(norepi_episodes),
)

print(
    "Patients with qualifying episodes:",
    norepi_episodes["patientid"].nunique(),
)

# ------------------------------------------------------------
# 7. FIRST QUALIFYING EPISODE PER PATIENT
# ------------------------------------------------------------

index_episodes = (
    norepi_episodes.sort_values(
        [
            "patientid",
            "norepi_start",
        ]
    )
    .groupby(
        "patientid",
        as_index=False,
    )
    .first()
    .rename(
        columns={
            "norepi_start": "time_zero",
        }
    )
)

print(
    "\nPatients with first qualifying " "norepinephrine episode:",
    index_episodes["patientid"].nunique(),
)

# ------------------------------------------------------------
# 8. ANTIBIOTIC TREATMENT COURSE
# ------------------------------------------------------------

abx_join = index_episodes[
    [
        "patientid",
        "time_zero",
    ]
].merge(
    antibiotics[
        [
            "patientid",
            "pharmaid",
            "givenat",
            "route",
            "infusionid",
        ]
    ],
    on="patientid",
    how="left",
)

abx_join["hours_from_t0"] = (
    abx_join["givenat"] - abx_join["time_zero"]
).dt.total_seconds() / 3600

# At least one IV antibiotic must occur close to shock onset.
abx_index_window = abx_join.loc[
    abx_join["hours_from_t0"].between(
        -24,
        6,
        inclusive="both",
    )
].copy()

patients_with_index_abx = set(
    abx_index_window["patientid"].dropna().astype(int).unique()
)

# Review the antibiotic course from -24 h to +96 h.
# The wider window allows confirmation of treatment >24 h.
abx_course_window = abx_join.loc[
    abx_join["patientid"].isin(patients_with_index_abx)
    & abx_join["hours_from_t0"].between(
        -24,
        96,
        inclusive="both",
    )
].copy()

if not abx_course_window.empty:

    abx_summary = abx_course_window.groupby(
        "patientid",
        as_index=False,
    ).agg(
        n_antibiotic_records=(
            "pharmaid",
            "size",
        ),
        n_distinct_antibiotics=(
            "pharmaid",
            "nunique",
        ),
        first_antibiotic_time=(
            "givenat",
            "min",
        ),
        last_antibiotic_time=(
            "givenat",
            "max",
        ),
    )

    abx_summary["antibiotic_duration_hours"] = (
        abx_summary["last_antibiotic_time"] - abx_summary["first_antibiotic_time"]
    ).dt.total_seconds() / 3600

else:

    abx_summary = pd.DataFrame(
        columns=[
            "patientid",
            "n_antibiotic_records",
            "n_distinct_antibiotics",
            "first_antibiotic_time",
            "last_antibiotic_time",
            "antibiotic_duration_hours",
        ]
    )

cohort = index_episodes.merge(
    abx_summary,
    on="patientid",
    how="left",
)

cohort["has_iv_antibiotic_near_t0"] = cohort["patientid"].isin(patients_with_index_abx)

cohort["has_repeated_iv_antibiotic"] = cohort["n_antibiotic_records"].fillna(0).ge(2)

cohort["antibiotic_duration_gt_24h"] = (
    cohort["antibiotic_duration_hours"].fillna(0).gt(24)
)

cohort["has_iv_antibiotic"] = (
    cohort["has_iv_antibiotic_near_t0"]
    & cohort["has_repeated_iv_antibiotic"]
    & cohort["antibiotic_duration_gt_24h"]
)

n_abx_near_t0 = int(cohort["has_iv_antibiotic_near_t0"].sum())

n_abx_repeated = int(
    (cohort["has_iv_antibiotic_near_t0"] & cohort["has_repeated_iv_antibiotic"]).sum()
)

n_abx_gt24 = int(cohort["has_iv_antibiotic"].sum())

print(
    "\nPatients with IV antibiotic " "from -24 h to +6 h:",
    n_abx_near_t0,
)

print(
    "Patients with repeated IV antibiotic administrations:",
    n_abx_repeated,
)

print(
    "Patients with IV antibiotic treatment duration >24 h:",
    n_abx_gt24,
)

# ------------------------------------------------------------
# 9. PATIENTS REQUIRING OBSERVATION EXTRACTION
# ------------------------------------------------------------

eligible_patient_ids = set(
    cohort.loc[
        cohort["has_iv_antibiotic"],
        "patientid",
    ]
    .astype(int)
    .tolist()
)

if not eligible_patient_ids:
    raise RuntimeError("No patients fulfilled the antibiotic-course criteria.")

wanted_obs_ids = LACTATE_IDS | SBP_IDS | DBP_IDS | MAP_IDS | HR_IDS

print(
    "\nPatients selected for observation extraction:",
    len(eligible_patient_ids),
)

print(
    "Observation variable IDs:",
    sorted(wanted_obs_ids),
)

# ------------------------------------------------------------
# 10. INSPECT OBSERVATION SCHEMA
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

    if not obs_members:
        raise RuntimeError("No parquet files found inside observation archive.")

    first_obs_part = read_parquet_member(
        tar,
        obs_members[0],
    )

print("\nObservation columns:")
print(first_obs_part.columns.tolist())

OBS_PATIENT_COL = detect_column(
    first_obs_part.columns,
    [
        "patientid",
        "patient_id",
    ],
)

OBS_VARIABLE_COL = detect_column(
    first_obs_part.columns,
    [
        "variableid",
        "variable_id",
    ],
)

OBS_TIME_COL = detect_column(
    first_obs_part.columns,
    [
        "datetime",
        "observationtime",
        "observation_time",
        "recordedat",
        "recorded_at",
    ],
)

OBS_VALUE_COL = detect_column(
    first_obs_part.columns,
    [
        "value",
        "value_numeric",
        "numericvalue",
        "numeric_value",
    ],
)

detected_obs_schema = {
    "patient_column": OBS_PATIENT_COL,
    "variable_column": OBS_VARIABLE_COL,
    "time_column": OBS_TIME_COL,
    "value_column": OBS_VALUE_COL,
}

print("\nDetected observation schema:")
print(
    json.dumps(
        detected_obs_schema,
        indent=2,
    )
)

if any(value is None for value in detected_obs_schema.values()):
    raise RuntimeError(
        "Could not detect observation schema.\n"
        f"Available columns: "
        f"{first_obs_part.columns.tolist()}"
    )

# ------------------------------------------------------------
# 11. EXTRACT RELEVANT OBSERVATIONS
# ------------------------------------------------------------

obs_parts = []

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
        "\nNumber of observation partitions:",
        len(parquet_members),
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
            part["patientid"].isin(eligible_patient_ids)
            & part["variableid"].isin(wanted_obs_ids)
        ].copy()

        if not part.empty:
            obs_parts.append(
                part[
                    [
                        "patientid",
                        "variableid",
                        "datetime",
                        "value",
                    ]
                ]
            )

        if idx % 25 == 0 or idx == len(parquet_members):
            print(f"Observations: processed " f"{idx}/{len(parquet_members)}")

if not obs_parts:
    raise RuntimeError("No relevant observations were extracted.")

observations = pd.concat(
    obs_parts,
    ignore_index=True,
)

del obs_parts
gc.collect()

observations["patientid"] = pd.to_numeric(
    observations["patientid"],
    errors="coerce",
)

observations["variableid"] = pd.to_numeric(
    observations["variableid"],
    errors="coerce",
)

observations["datetime"] = pd.to_datetime(
    observations["datetime"],
    errors="coerce",
)

observations["value"] = pd.to_numeric(
    observations["value"],
    errors="coerce",
)

observations = observations.dropna(
    subset=[
        "patientid",
        "variableid",
        "datetime",
        "value",
    ]
)

observations["patientid"] = observations["patientid"].astype(int)

observations["variableid"] = observations["variableid"].astype(int)

print(
    "\nRelevant observations extracted:",
    len(observations),
)

print(
    "Patients with relevant observations:",
    observations["patientid"].nunique(),
)

# ------------------------------------------------------------
# 12. LINK OBSERVATIONS TO TIME ZERO
# ------------------------------------------------------------

obs_t0 = observations.merge(
    cohort[
        [
            "patientid",
            "time_zero",
        ]
    ],
    on="patientid",
    how="inner",
)

obs_t0["hours_from_t0"] = (
    obs_t0["datetime"] - obs_t0["time_zero"]
).dt.total_seconds() / 3600

obs_t0 = obs_t0.loc[
    obs_t0["hours_from_t0"].between(
        -24,
        24,
        inclusive="both",
    )
].copy()

# ------------------------------------------------------------
# 13. LACTATE WITHIN +/-6 HOURS
# ------------------------------------------------------------

lactate = obs_t0.loc[
    obs_t0["variableid"].isin(LACTATE_IDS)
    & obs_t0["hours_from_t0"].between(
        -6,
        6,
        inclusive="both",
    )
].copy()

lactate["abs_hours_from_t0"] = lactate["hours_from_t0"].abs()

if not lactate.empty:

    lactate_closest = (
        lactate.sort_values(
            [
                "patientid",
                "abs_hours_from_t0",
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
                "variableid",
            ]
        ]
        .rename(
            columns={
                "value": "lactate_t0",
                "datetime": "lactate_time",
                "hours_from_t0": "lactate_hours_from_t0",
                "variableid": "lactate_variableid",
            }
        )
    )

else:

    lactate_closest = pd.DataFrame(
        columns=[
            "patientid",
            "lactate_t0",
            "lactate_time",
            "lactate_hours_from_t0",
            "lactate_variableid",
        ]
    )

cohort = cohort.merge(
    lactate_closest,
    on="patientid",
    how="left",
)

cohort["has_lactate"] = cohort["lactate_t0"].notna()

cohort["lactate_gt_2"] = cohort["lactate_t0"].gt(2).fillna(False)

print(
    "\nPatients with blood lactate within +/-6 h:",
    int(cohort["has_lactate"].sum()),
)

print(
    "Patients with blood lactate >2 mmol/L:",
    int(cohort["lactate_gt_2"].sum()),
)

# ------------------------------------------------------------
# 14. INVASIVE BP AVAILABILITY NEAR TIME ZERO
# ------------------------------------------------------------

BP_IDS = SBP_IDS | DBP_IDS | MAP_IDS

bp_near_t0 = obs_t0.loc[
    obs_t0["variableid"].isin(BP_IDS)
    & obs_t0["hours_from_t0"].between(
        -1,
        1,
        inclusive="both",
    )
].copy()

bp_near_t0["is_sbp"] = bp_near_t0["variableid"].isin(SBP_IDS)

bp_near_t0["is_dbp"] = bp_near_t0["variableid"].isin(DBP_IDS)

bp_near_t0["is_map"] = bp_near_t0["variableid"].isin(MAP_IDS)

if not bp_near_t0.empty:

    bp_counts = bp_near_t0.groupby(
        "patientid",
        as_index=False,
    ).agg(
        n_bp_records=(
            "value",
            "count",
        ),
        n_sbp_records=(
            "is_sbp",
            "sum",
        ),
        n_dbp_records=(
            "is_dbp",
            "sum",
        ),
        n_map_records=(
            "is_map",
            "sum",
        ),
    )

else:

    bp_counts = pd.DataFrame(
        columns=[
            "patientid",
            "n_bp_records",
            "n_sbp_records",
            "n_dbp_records",
            "n_map_records",
        ]
    )

cohort = cohort.merge(
    bp_counts,
    on="patientid",
    how="left",
)

for col in [
    "n_bp_records",
    "n_sbp_records",
    "n_dbp_records",
    "n_map_records",
]:
    cohort[col] = cohort[col].fillna(0).astype(int)

cohort["has_invasive_bp_near_t0"] = (
    cohort["n_sbp_records"].gt(0)
    & cohort["n_dbp_records"].gt(0)
    & cohort["n_map_records"].gt(0)
)

print(
    "Patients with invasive SBP, DBP and MAP " "within +/-1 h:",
    int(cohort["has_invasive_bp_near_t0"].sum()),
)

# ------------------------------------------------------------
# 15. HEART RATE AVAILABILITY
# ------------------------------------------------------------

hr_near_t0 = obs_t0.loc[
    obs_t0["variableid"].isin(HR_IDS)
    & obs_t0["hours_from_t0"].between(
        -1,
        1,
        inclusive="both",
    )
].copy()

hr_counts = (
    hr_near_t0.groupby("patientid").size().rename("n_hr_records_near_t0").reset_index()
)

cohort = cohort.merge(
    hr_counts,
    on="patientid",
    how="left",
)

cohort["n_hr_records_near_t0"] = cohort["n_hr_records_near_t0"].fillna(0).astype(int)

cohort["has_hr_near_t0"] = cohort["n_hr_records_near_t0"].gt(0)

print(
    "Patients with heart rate within +/-1 h:",
    int(cohort["has_hr_near_t0"].sum()),
)

# ------------------------------------------------------------
# 16. SIX-HOUR INVASIVE BP FOLLOW-UP
# ------------------------------------------------------------

bp_followup = obs_t0.loc[
    obs_t0["variableid"].isin(BP_IDS)
    & obs_t0["hours_from_t0"].between(
        0,
        6,
        inclusive="both",
    )
].copy()

if not bp_followup.empty:

    followup_summary = bp_followup.groupby(
        "patientid",
        as_index=False,
    ).agg(
        last_bp_time=(
            "datetime",
            "max",
        ),
        max_followup_hours=(
            "hours_from_t0",
            "max",
        ),
        n_bp_records_0_6h=(
            "value",
            "count",
        ),
    )

else:

    followup_summary = pd.DataFrame(
        columns=[
            "patientid",
            "last_bp_time",
            "max_followup_hours",
            "n_bp_records_0_6h",
        ]
    )

cohort = cohort.merge(
    followup_summary,
    on="patientid",
    how="left",
)

cohort["has_6h_bp_followup"] = cohort["max_followup_hours"].fillna(-np.inf).ge(5.5)

print(
    "Patients with invasive BP extending " "to at least 5.5 h after time zero:",
    int(cohort["has_6h_bp_followup"].sum()),
)

# ------------------------------------------------------------
# 17. DEFINE BROAD AND STRICT COHORTS
# ------------------------------------------------------------

cohort["probable_septic_shock_broad"] = (
    cohort["has_iv_antibiotic"]
    & cohort["has_invasive_bp_near_t0"]
    & cohort["has_hr_near_t0"]
    & cohort["has_6h_bp_followup"]
)

cohort["probable_septic_shock_strict"] = (
    cohort["probable_septic_shock_broad"] & cohort["lactate_gt_2"]
)

# ------------------------------------------------------------
# 18. FLOWCHART COUNTS
# ------------------------------------------------------------

n_any_norepi = int(
    pharma.loc[
        pharma["pharmaid"].isin(NOREPI_IDS_ALL),
        "patientid",
    ].nunique()
)

n_continuous_norepi = int(index_episodes["patientid"].nunique())

n_with_invasive_bp = int(
    (cohort["has_iv_antibiotic"] & cohort["has_invasive_bp_near_t0"]).sum()
)

n_with_hr = int(
    (
        cohort["has_iv_antibiotic"]
        & cohort["has_invasive_bp_near_t0"]
        & cohort["has_hr_near_t0"]
    ).sum()
)

n_with_followup = int(
    (
        cohort["has_iv_antibiotic"]
        & cohort["has_invasive_bp_near_t0"]
        & cohort["has_hr_near_t0"]
        & cohort["has_6h_bp_followup"]
    ).sum()
)

n_with_lactate = int((cohort["has_iv_antibiotic"] & cohort["has_lactate"]).sum())

n_lactate_gt2 = int((cohort["has_iv_antibiotic"] & cohort["lactate_gt_2"]).sum())

n_broad = int(cohort["probable_septic_shock_broad"].sum())

n_strict = int(cohort["probable_septic_shock_strict"].sum())

flowchart = pd.DataFrame(
    {
        "step": [
            "Any norepinephrine administration",
            "Continuous norepinephrine infusion >=60 min",
            "IV antibiotic from -24 h to +6 h",
            "Repeated IV antibiotic administrations",
            "IV antibiotic treatment duration >24 h",
            "Invasive SBP, DBP and MAP near time zero",
            "Heart rate near time zero",
            "Approximately 6 h invasive BP follow-up",
            "Blood lactate available within +/-6 h",
            "Blood lactate >2 mmol/L",
            "Broad probable septic shock cohort",
            "Strict probable septic shock cohort",
        ],
        "n_patients": [
            n_any_norepi,
            n_continuous_norepi,
            n_abx_near_t0,
            n_abx_repeated,
            n_abx_gt24,
            n_with_invasive_bp,
            n_with_hr,
            n_with_followup,
            n_with_lactate,
            n_lactate_gt2,
            n_broad,
            n_strict,
        ],
    }
)

print("\n" + "=" * 80)
print("FEASIBILITY FLOWCHART")
print("=" * 80)

display(flowchart)

# ------------------------------------------------------------
# 19. CLINICAL SUMMARY
# ------------------------------------------------------------

clinical_summary = pd.DataFrame(
    {
        "metric": [
            "Median qualifying norepinephrine duration, h",
            "IQR qualifying norepinephrine duration, h",
            "Median antibiotic course duration, h",
            "IQR antibiotic course duration, h",
            "Median lactate near time zero, mmol/L",
            "IQR lactate near time zero, mmol/L",
            "Broad cohort size",
            "Strict cohort size",
        ],
        "value": [
            float(index_episodes["duration_hours"].median()),
            (
                f"{index_episodes['duration_hours'].quantile(0.25):.2f}"
                f"–"
                f"{index_episodes['duration_hours'].quantile(0.75):.2f}"
            ),
            (
                float(
                    cohort.loc[
                        cohort["has_iv_antibiotic"],
                        "antibiotic_duration_hours",
                    ].median()
                )
                if cohort["has_iv_antibiotic"].any()
                else np.nan
            ),
            (
                (
                    f"{cohort.loc[cohort['has_iv_antibiotic'], 'antibiotic_duration_hours'].quantile(0.25):.2f}"
                    f"–"
                    f"{cohort.loc[cohort['has_iv_antibiotic'], 'antibiotic_duration_hours'].quantile(0.75):.2f}"
                )
                if cohort["has_iv_antibiotic"].any()
                else np.nan
            ),
            (
                float(cohort["lactate_t0"].median())
                if cohort["lactate_t0"].notna().any()
                else np.nan
            ),
            (
                (
                    f"{cohort['lactate_t0'].quantile(0.25):.2f}"
                    f"–"
                    f"{cohort['lactate_t0'].quantile(0.75):.2f}"
                )
                if cohort["lactate_t0"].notna().any()
                else np.nan
            ),
            n_broad,
            n_strict,
        ],
    }
)

display(clinical_summary)

# ------------------------------------------------------------
# 20. SAVE OUTPUTS
# ------------------------------------------------------------

flowchart.to_csv(
    TABLE_DIR / "step02_septic_shock_feasibility_flowchart_atb_gt24h.csv",
    index=False,
)

clinical_summary.to_csv(
    TABLE_DIR / "step02_septic_shock_feasibility_summary_atb_gt24h.csv",
    index=False,
)

cohort.to_parquet(
    DERIVED_DIR / "step02_probable_septic_shock_candidates_atb_gt24h.parquet",
    index=False,
)

cohort.to_csv(
    TABLE_DIR / "step02_probable_septic_shock_candidates_atb_gt24h.csv",
    index=False,
)

norepi_episodes.to_parquet(
    DERIVED_DIR / "step02_norepinephrine_episodes.parquet",
    index=False,
)

obs_t0.to_parquet(
    DERIVED_DIR / "step02_relevant_observations_t0_minus24_plus24h_atb_gt24h.parquet",
    index=False,
)

abx_index_window.to_parquet(
    DERIVED_DIR / "step02_antibiotics_index_window_minus24_plus6h.parquet",
    index=False,
)

abx_course_window.to_parquet(
    DERIVED_DIR / "step02_antibiotic_courses_minus24_plus96h.parquet",
    index=False,
)

broad_cohort = cohort.loc[cohort["probable_septic_shock_broad"]].copy()

strict_cohort = cohort.loc[cohort["probable_septic_shock_strict"]].copy()

broad_cohort.to_parquet(
    DERIVED_DIR / "step02_broad_probable_septic_shock_cohort_atb_gt24h.parquet",
    index=False,
)

strict_cohort.to_parquet(
    DERIVED_DIR / "step02_strict_probable_septic_shock_cohort_atb_gt24h.parquet",
    index=False,
)

# ------------------------------------------------------------
# 21. SAVE RUN SUMMARY
# ------------------------------------------------------------

run_summary = {
    "any_norepinephrine_patients": n_any_norepi,
    "continuous_norepinephrine_patients": n_continuous_norepi,
    "patients_with_iv_antibiotic_near_t0": n_abx_near_t0,
    "patients_with_repeated_iv_antibiotic": n_abx_repeated,
    "patients_with_antibiotic_duration_gt_24h": n_abx_gt24,
    "patients_with_invasive_bp_near_t0": n_with_invasive_bp,
    "patients_with_hr_near_t0": n_with_hr,
    "patients_with_6h_bp_followup": n_with_followup,
    "patients_with_lactate": n_with_lactate,
    "patients_with_lactate_gt_2": n_lactate_gt2,
    "broad_probable_septic_shock": n_broad,
    "strict_probable_septic_shock": n_strict,
    "observation_schema": detected_obs_schema,
}

with open(
    LOG_DIR / "step02_septic_shock_feasibility_atb_gt24h.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        run_summary,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n" + "=" * 80)
print("STEP 02 COMPLETE")
print("=" * 80)

print(
    "\nBroad probable septic shock cohort:",
    n_broad,
)

print(
    "Strict probable septic shock cohort:",
    n_strict,
)

print(
    "\nBroad cohort saved to:\n"
    f"{DERIVED_DIR / 'step02_broad_probable_septic_shock_cohort_atb_gt24h.parquet'}"
)

print(
    "\nStrict cohort saved to:\n"
    f"{DERIVED_DIR / 'step02_strict_probable_septic_shock_cohort_atb_gt24h.parquet'}"
)

print(
    "\nFlowchart saved to:\n"
    f"{TABLE_DIR / 'step02_septic_shock_feasibility_flowchart_atb_gt24h.csv'}"
)
