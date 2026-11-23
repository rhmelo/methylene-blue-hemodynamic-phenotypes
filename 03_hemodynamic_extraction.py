"""Source: original notebook cell 10. Run through run.py."""

from common import display, hirid_source, phase_dir

# ============================================================
# HiRID Methylene Blue Hemodynamic Phenotypes
# STEP 03 — Build analysis-ready hemodynamic dataset
#
# PHENOTYPE-DERIVATION WINDOW:
#   0 to 1 hour after time zero
#
# LONGITUDINAL VALIDATION WINDOW:
#   >1 to 6 hours after time zero
#
# Baseline phenotype variables:
# - Invasive DBP
# - Pulse pressure
# - Heart rate
# - Norepinephrine dose rate
# - Lactate
#
# Variables NOT used to derive the phenotype:
# - Subsequent trajectories
# - Hypotension burden
# - Vasopressor persistence
# - AKI
# - Mortality
#
# These will be used later for physiological and prognostic
# validation, avoiding circularity.
# ============================================================

import pandas as pd
import numpy as np
import tarfile
import io
import gc
import json
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

COHORT_PATH = DERIVED_DIR / "step02_probable_septic_shock_candidates_atb_gt24h.parquet"

OBS_PATH = (
    DERIVED_DIR / "step02_relevant_observations_t0_minus24_plus24h_atb_gt24h.parquet"
)

PHARMA_ARCHIVE = HIRID_DIR / "pharma_records_parquet.tar.gz"

ID_MAP_PATH = LOG_DIR / "step01c_final_variable_id_map.json"

for required_path in [
    COHORT_PATH,
    OBS_PATH,
    PHARMA_ARCHIVE,
    ID_MAP_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required file not found:\n{required_path}")

# ------------------------------------------------------------
# 2. LOAD VARIABLE MAP
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

NOREPI_BOLUS_IDS = {1000658}

NOREPI_INFUSION_IDS = NOREPI_IDS_ALL - NOREPI_BOLUS_IDS

SBP_IDS = set(map(int, id_map["sbp_ids"]))

DBP_IDS = set(map(int, id_map["dbp_ids"]))

MAP_IDS = set(map(int, id_map["map_ids"]))

HR_IDS = set(map(int, id_map["heart_rate_ids"]))

LACTATE_IDS = set(map(int, id_map["lactate_ids"]))

# ------------------------------------------------------------
# 3. LOAD COHORT
# ------------------------------------------------------------

cohort_all = pd.read_parquet(COHORT_PATH)

broad_cohort = cohort_all.loc[cohort_all["probable_septic_shock_broad"]].copy()

broad_cohort["patientid"] = pd.to_numeric(
    broad_cohort["patientid"],
    errors="coerce",
).astype(int)

broad_cohort["time_zero"] = pd.to_datetime(
    broad_cohort["time_zero"],
    errors="coerce",
)

COHORT_PATIENTS = set(broad_cohort["patientid"].astype(int))

print("=" * 80)
print("COHORT")
print("=" * 80)

print(
    "Broad probable septic shock cohort:",
    len(broad_cohort),
)

print(
    "Strict probable septic shock patients:",
    int(broad_cohort["probable_septic_shock_strict"].sum()),
)

# ------------------------------------------------------------
# 4. READ OBSERVATIONS
# ------------------------------------------------------------

# Try to apply a parquet filter so only the 0–6 h period is
# loaded. If the parquet engine cannot apply it, read the file
# and filter immediately afterward.

obs_columns = [
    "patientid",
    "variableid",
    "datetime",
    "value",
    "time_zero",
    "hours_from_t0",
]

try:

    obs = pd.read_parquet(
        OBS_PATH,
        columns=obs_columns,
        filters=[
            (
                "hours_from_t0",
                ">=",
                0.0,
            ),
            (
                "hours_from_t0",
                "<=",
                6.0,
            ),
        ],
    )

except Exception as exc:

    print("\nParquet filter could not be applied directly.")

    print("Reading selected columns and filtering afterward.")

    print(
        "Reason:",
        type(exc).__name__,
        str(exc)[:200],
    )

    obs = pd.read_parquet(
        OBS_PATH,
        columns=obs_columns,
    )

    obs = obs.loc[
        obs["hours_from_t0"].between(
            0,
            6,
            inclusive="both",
        )
    ].copy()

obs = obs.loc[obs["patientid"].isin(COHORT_PATIENTS)].copy()

obs["patientid"] = pd.to_numeric(
    obs["patientid"],
    errors="coerce",
).astype(int)

obs["variableid"] = pd.to_numeric(
    obs["variableid"],
    errors="coerce",
).astype(int)

obs["value"] = pd.to_numeric(
    obs["value"],
    errors="coerce",
)

obs["hours_from_t0"] = pd.to_numeric(
    obs["hours_from_t0"],
    errors="coerce",
)

obs["datetime"] = pd.to_datetime(
    obs["datetime"],
    errors="coerce",
)

obs = obs.dropna(
    subset=[
        "patientid",
        "variableid",
        "value",
        "hours_from_t0",
    ]
)

print("\nObservations from 0 to 6 h:", len(obs))

print(
    "Patients represented:",
    obs["patientid"].nunique(),
)

# ------------------------------------------------------------
# 5. ASSIGN CLINICAL VARIABLE NAMES
# ------------------------------------------------------------


def assign_variable_name(variableid):
    if variableid in SBP_IDS:
        return "sbp"

    if variableid in DBP_IDS:
        return "dbp"

    if variableid in MAP_IDS:
        return "map"

    if variableid in HR_IDS:
        return "heart_rate"

    if variableid in LACTATE_IDS:
        return "lactate"

    return np.nan


obs["variable"] = obs["variableid"].map(assign_variable_name)

obs = obs.dropna(subset=["variable"])

# ------------------------------------------------------------
# 6. PHYSIOLOGICAL RANGE CLEANING
# ------------------------------------------------------------

# Conservative ranges intended only to remove obvious
# artifacts. These are not clinical inclusion thresholds.

VALID_RANGES = {
    "sbp": (30, 300),
    "dbp": (15, 200),
    "map": (20, 250),
    "heart_rate": (20, 250),
    "lactate": (0.1, 30),
}

before_cleaning = len(obs)

range_valid = pd.Series(
    False,
    index=obs.index,
)

for variable, (
    lower,
    upper,
) in VALID_RANGES.items():

    variable_mask = obs["variable"].eq(variable)

    range_valid.loc[variable_mask] = obs.loc[
        variable_mask,
        "value",
    ].between(
        lower,
        upper,
        inclusive="both",
    )

obs = obs.loc[range_valid].copy()

print(
    "\nObservations removed as implausible:",
    before_cleaning - len(obs),
)

# ------------------------------------------------------------
# 7. BASELINE HEMODYNAMICS: 0–1 HOUR
# ------------------------------------------------------------

baseline_obs = obs.loc[
    obs["hours_from_t0"].between(
        0,
        1,
        inclusive="both",
    )
    & obs["variable"].isin(
        [
            "sbp",
            "dbp",
            "map",
            "heart_rate",
        ]
    )
].copy()

baseline_summary = baseline_obs.groupby(
    [
        "patientid",
        "variable",
    ],
    as_index=False,
).agg(
    median_value=(
        "value",
        "median",
    ),
    mean_value=(
        "value",
        "mean",
    ),
    n_values=(
        "value",
        "size",
    ),
    q25_value=(
        "value",
        lambda x: x.quantile(0.25),
    ),
    q75_value=(
        "value",
        lambda x: x.quantile(0.75),
    ),
)

baseline_median = (
    baseline_summary.pivot(
        index="patientid",
        columns="variable",
        values="median_value",
    )
    .reset_index()
    .rename(
        columns={
            "sbp": "sbp_baseline",
            "dbp": "dbp_baseline",
            "map": "map_baseline",
            "heart_rate": "heart_rate_baseline",
        }
    )
)

baseline_counts = (
    baseline_summary.pivot(
        index="patientid",
        columns="variable",
        values="n_values",
    )
    .reset_index()
    .rename(
        columns={
            "sbp": "n_sbp_baseline",
            "dbp": "n_dbp_baseline",
            "map": "n_map_baseline",
            "heart_rate": "n_heart_rate_baseline",
        }
    )
)

baseline_hemodynamics = baseline_median.merge(
    baseline_counts,
    on="patientid",
    how="outer",
)

# ------------------------------------------------------------
# 8. DERIVED BASELINE INDICES
# ------------------------------------------------------------

baseline_hemodynamics["pulse_pressure_baseline"] = (
    baseline_hemodynamics["sbp_baseline"] - baseline_hemodynamics["dbp_baseline"]
)

baseline_hemodynamics["shock_index_baseline"] = (
    baseline_hemodynamics["heart_rate_baseline"] / baseline_hemodynamics["sbp_baseline"]
)

baseline_hemodynamics["modified_shock_index_baseline"] = (
    baseline_hemodynamics["heart_rate_baseline"] / baseline_hemodynamics["map_baseline"]
)

baseline_hemodynamics["diastolic_shock_index_baseline"] = (
    baseline_hemodynamics["heart_rate_baseline"] / baseline_hemodynamics["dbp_baseline"]
)

# Remove non-physiological derived pressure differences.
baseline_hemodynamics.loc[
    baseline_hemodynamics["pulse_pressure_baseline"].le(0),
    "pulse_pressure_baseline",
] = np.nan

# ------------------------------------------------------------
# 9. LACTATE NEAR PHENOTYPE WINDOW
# ------------------------------------------------------------

# For phenotype derivation, use the lactate closest to the
# beginning of the first hour, from -6 to +1 h, as already
# identified in the cohort dataset.

lactate_baseline = broad_cohort[
    [
        "patientid",
        "lactate_t0",
        "lactate_time",
        "lactate_hours_from_t0",
        "lactate_variableid",
        "lactate_gt_2",
    ]
].copy()

lactate_baseline = lactate_baseline.rename(
    columns={
        "lactate_t0": "lactate_baseline",
    }
)

# ------------------------------------------------------------
# 10. READ NOREPINEPHRINE RECORDS FOR COHORT
# ------------------------------------------------------------


def read_parquet_member(
    tar_object,
    member,
    columns=None,
):
    extracted = tar_object.extractfile(member)

    if extracted is None:
        return None

    raw_bytes = extracted.read()

    return pd.read_parquet(
        io.BytesIO(raw_bytes),
        columns=columns,
    )


norepi_parts = []

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
        "\nReading norepinephrine records from",
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
                "patientid",
                "pharmaid",
                "givenat",
                "givendose",
                "doseunit",
                "route",
                "infusionid",
                "recordstatus",
            ],
        )

        if part is None or part.empty:
            continue

        part = part.loc[
            part["patientid"].isin(COHORT_PATIENTS)
            & part["pharmaid"].isin(NOREPI_INFUSION_IDS)
            & part["route"]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "iv-inf",
                    "cv-inf",
                ]
            )
        ].copy()

        if not part.empty:
            norepi_parts.append(part)

        if idx % 25 == 0 or idx == len(parquet_members):
            print(f"Norepinephrine: processed " f"{idx}/{len(parquet_members)}")

if not norepi_parts:
    raise RuntimeError("No norepinephrine records found " "for the broad cohort.")

norepi = pd.concat(
    norepi_parts,
    ignore_index=True,
)

del norepi_parts
gc.collect()

norepi["patientid"] = pd.to_numeric(
    norepi["patientid"],
    errors="coerce",
)

norepi["pharmaid"] = pd.to_numeric(
    norepi["pharmaid"],
    errors="coerce",
)

norepi["givenat"] = pd.to_datetime(
    norepi["givenat"],
    errors="coerce",
)

norepi["givendose"] = pd.to_numeric(
    norepi["givendose"],
    errors="coerce",
)

norepi = norepi.dropna(
    subset=[
        "patientid",
        "pharmaid",
        "givenat",
        "givendose",
    ]
)

norepi["patientid"] = norepi["patientid"].astype(int)

# ------------------------------------------------------------
# 11. LINK NOREPINEPHRINE TO TIME ZERO
# ------------------------------------------------------------

norepi = norepi.merge(
    broad_cohort[
        [
            "patientid",
            "time_zero",
        ]
    ],
    on="patientid",
    how="inner",
)

norepi["hours_from_t0"] = (
    norepi["givenat"] - norepi["time_zero"]
).dt.total_seconds() / 3600

# Retain a small pre-window so that the first post-t0 dose
# interval can be calculated correctly.
norepi = norepi.loc[
    norepi["hours_from_t0"].between(
        -1,
        6.5,
        inclusive="both",
    )
].copy()

print(
    "\nNorepinephrine records in -1 to +6.5 h:",
    len(norepi),
)

print(
    "Patients with norepinephrine records:",
    norepi["patientid"].nunique(),
)

# ------------------------------------------------------------
# 12. AUDIT NOREPINEPHRINE UNITS
# ------------------------------------------------------------

norepi_unit_summary = (
    norepi.groupby(
        [
            "pharmaid",
            "doseunit",
            "route",
        ],
        dropna=False,
    )
    .agg(
        n_records=(
            "givendose",
            "size",
        ),
        n_patients=(
            "patientid",
            "nunique",
        ),
        median_given_dose=(
            "givendose",
            "median",
        ),
        q25_given_dose=(
            "givendose",
            lambda x: x.quantile(0.25),
        ),
        q75_given_dose=(
            "givendose",
            lambda x: x.quantile(0.75),
        ),
    )
    .reset_index()
)

print("\n" + "=" * 80)
print("NOREPINEPHRINE UNIT AUDIT")
print("=" * 80)

display(norepi_unit_summary)

norepi_unit_summary.to_csv(
    TABLE_DIR / "step03_norepinephrine_unit_audit.csv",
    index=False,
)

# ------------------------------------------------------------
# 13. RECONSTRUCT NOREPINEPHRINE RATE
# ------------------------------------------------------------

# HiRID givendose represents the quantity administered over
# an interval. We divide it by the elapsed time since the
# preceding record in the same infusion.
#
# Records with intervals >45 min are not used to reconstruct
# a continuous dose rate.

norepi = norepi.sort_values(
    [
        "patientid",
        "infusionid",
        "givenat",
    ]
).copy()

norepi["previous_givenat"] = norepi.groupby(
    [
        "patientid",
        "infusionid",
    ],
    dropna=False,
)["givenat"].shift(1)

norepi["interval_minutes"] = (
    norepi["givenat"] - norepi["previous_givenat"]
).dt.total_seconds() / 60

valid_interval = norepi["interval_minutes"].gt(0) & norepi["interval_minutes"].le(45)

# All identified norepinephrine formulations should use µg.
# Values recorded in another unit are not converted silently.
norepi["doseunit_normalized"] = norepi["doseunit"].astype(str).str.strip().str.lower()

microgram_units = {
    "µg",
    "μg",
    "ug",
    "mcg",
}

valid_microgram_unit = norepi["doseunit_normalized"].isin(microgram_units)

norepi["norepi_ug_min"] = np.nan

norepi.loc[
    valid_interval & valid_microgram_unit,
    "norepi_ug_min",
] = (
    norepi.loc[
        valid_interval & valid_microgram_unit,
        "givendose",
    ]
    / norepi.loc[
        valid_interval & valid_microgram_unit,
        "interval_minutes",
    ]
)

# Remove impossible or extreme values only for data-quality
# inspection. The upper limit is deliberately permissive.
norepi.loc[
    (norepi["norepi_ug_min"] < 0) | (norepi["norepi_ug_min"] > 500),
    "norepi_ug_min",
] = np.nan

rate_distribution = (
    norepi["norepi_ug_min"]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .rename_axis("statistic")
    .reset_index(name="value")
)

print("\n" + "=" * 80)
print("RECONSTRUCTED NOREPINEPHRINE RATE, µg/min")
print("=" * 80)

display(rate_distribution)

rate_distribution.to_csv(
    TABLE_DIR / "step03_norepinephrine_rate_distribution.csv",
    index=False,
)

# ------------------------------------------------------------
# 14. BASELINE NOREPINEPHRINE: 0–1 HOUR
# ------------------------------------------------------------

norepi_baseline_records = norepi.loc[
    norepi["hours_from_t0"].between(
        0,
        1,
        inclusive="both",
    )
    & norepi["norepi_ug_min"].notna()
].copy()

norepi_baseline = norepi_baseline_records.groupby(
    "patientid",
    as_index=False,
).agg(
    norepi_ug_min_baseline=(
        "norepi_ug_min",
        "median",
    ),
    norepi_ug_min_baseline_mean=(
        "norepi_ug_min",
        "mean",
    ),
    norepi_ug_min_baseline_max=(
        "norepi_ug_min",
        "max",
    ),
    n_norepi_records_baseline=(
        "norepi_ug_min",
        "size",
    ),
)

# ------------------------------------------------------------
# 15. CREATE HOURLY HEMODYNAMIC DATA
# ------------------------------------------------------------

# Hour 0 represents 0–1 h, hour 1 represents >1–2 h, etc.
obs["hour_bin"] = np.floor(obs["hours_from_t0"]).astype(int)

obs.loc[
    obs["hour_bin"].lt(0),
    "hour_bin",
] = 0

obs.loc[
    obs["hour_bin"].gt(5),
    "hour_bin",
] = 5

hourly_obs = (
    obs.loc[
        obs["variable"].isin(
            [
                "sbp",
                "dbp",
                "map",
                "heart_rate",
            ]
        )
    ]
    .groupby(
        [
            "patientid",
            "hour_bin",
            "variable",
        ],
        as_index=False,
    )
    .agg(
        hourly_median=(
            "value",
            "median",
        ),
        hourly_mean=(
            "value",
            "mean",
        ),
        n_values=(
            "value",
            "size",
        ),
    )
)

hourly_hemodynamics = hourly_obs.pivot_table(
    index=[
        "patientid",
        "hour_bin",
    ],
    columns="variable",
    values="hourly_median",
    aggfunc="first",
).reset_index()

hourly_hemodynamics["pulse_pressure"] = (
    hourly_hemodynamics["sbp"] - hourly_hemodynamics["dbp"]
)

hourly_hemodynamics.loc[
    hourly_hemodynamics["pulse_pressure"].le(0),
    "pulse_pressure",
] = np.nan

# ------------------------------------------------------------
# 16. CREATE HOURLY NOREPINEPHRINE DATA
# ------------------------------------------------------------

norepi_valid = norepi.loc[
    norepi["hours_from_t0"].between(
        0,
        6,
        inclusive="both",
    )
    & norepi["norepi_ug_min"].notna()
].copy()

norepi_valid["hour_bin"] = np.floor(norepi_valid["hours_from_t0"]).astype(int)

norepi_valid.loc[
    norepi_valid["hour_bin"].gt(5),
    "hour_bin",
] = 5

hourly_norepi = norepi_valid.groupby(
    [
        "patientid",
        "hour_bin",
    ],
    as_index=False,
).agg(
    norepi_ug_min=(
        "norepi_ug_min",
        "median",
    ),
    norepi_ug_min_mean=(
        "norepi_ug_min",
        "mean",
    ),
    norepi_ug_min_max=(
        "norepi_ug_min",
        "max",
    ),
    n_norepi_records=(
        "norepi_ug_min",
        "size",
    ),
)

# ------------------------------------------------------------
# 17. COMBINE HOURLY HEMODYNAMICS AND NOREPINEPHRINE
# ------------------------------------------------------------

hourly = hourly_hemodynamics.merge(
    hourly_norepi,
    on=[
        "patientid",
        "hour_bin",
    ],
    how="outer",
)

hourly = hourly.sort_values(
    [
        "patientid",
        "hour_bin",
    ]
)

# ------------------------------------------------------------
# 18. HELPER FUNCTIONS FOR LONGITUDINAL METRICS
# ------------------------------------------------------------


def patient_slope(
    group,
    value_column,
    minimum_points=3,
):
    """
    Linear slope per hour using hourly medians.
    """

    valid = group[
        [
            "hour_bin",
            value_column,
        ]
    ].dropna()

    if len(valid) < minimum_points:
        return np.nan

    if valid["hour_bin"].nunique() < minimum_points:
        return np.nan

    slope = np.polyfit(
        valid["hour_bin"].astype(float),
        valid[value_column].astype(float),
        1,
    )[0]

    return float(slope)


def trapezoidal_auc(
    group,
    value_column,
    minimum_points=2,
):
    """
    Trapezoidal area under the hourly curve.
    """

    valid = (
        group[
            [
                "hour_bin",
                value_column,
            ]
        ]
        .dropna()
        .sort_values("hour_bin")
    )

    if len(valid) < minimum_points:
        return np.nan

    return float(
        np.trapz(
            valid[value_column].astype(float),
            valid["hour_bin"].astype(float),
        )
    )


# ------------------------------------------------------------
# 19. LONGITUDINAL VALIDATION WINDOW: HOURS 1–6
# ------------------------------------------------------------

validation_hourly = hourly.loc[
    hourly["hour_bin"].between(
        1,
        5,
        inclusive="both",
    )
].copy()

longitudinal_rows = []

for patientid, group in validation_hourly.groupby(
    "patientid",
    sort=False,
):

    group = group.sort_values("hour_bin")

    dbp_values = group["dbp"].dropna()
    map_values = group["map"].dropna()
    pp_values = group["pulse_pressure"].dropna()

    ne_values = group["norepi_ug_min"].dropna()

    last_hour = group["hour_bin"].max() if not group.empty else np.nan

    final_row = group.loc[group["hour_bin"].eq(last_hour)].tail(1)

    norepi_at_last_hour = (
        final_row["norepi_ug_min"].iloc[0]
        if (not final_row.empty and pd.notna(final_row["norepi_ug_min"].iloc[0]))
        else np.nan
    )

    longitudinal_rows.append(
        {
            "patientid": int(patientid),
            # Trajectories
            "dbp_slope_h1_h6": patient_slope(
                group,
                "dbp",
            ),
            "map_slope_h1_h6": patient_slope(
                group,
                "map",
            ),
            "pulse_pressure_slope_h1_h6": patient_slope(
                group,
                "pulse_pressure",
            ),
            "heart_rate_slope_h1_h6": patient_slope(
                group,
                "heart_rate",
            ),
            "norepi_slope_h1_h6": patient_slope(
                group,
                "norepi_ug_min",
            ),
            # Median state during follow-up
            "dbp_median_h1_h6": (
                float(dbp_values.median()) if not dbp_values.empty else np.nan
            ),
            "map_median_h1_h6": (
                float(map_values.median()) if not map_values.empty else np.nan
            ),
            "pulse_pressure_median_h1_h6": (
                float(pp_values.median()) if not pp_values.empty else np.nan
            ),
            "norepi_median_h1_h6": (
                float(ne_values.median()) if not ne_values.empty else np.nan
            ),
            "norepi_max_h1_h6": (
                float(ne_values.max()) if not ne_values.empty else np.nan
            ),
            # Approximate hourly burden
            "proportion_hours_dbp_lt_50_h1_h6": (
                float((dbp_values < 50).mean()) if not dbp_values.empty else np.nan
            ),
            "proportion_hours_dbp_lt_45_h1_h6": (
                float((dbp_values < 45).mean()) if not dbp_values.empty else np.nan
            ),
            "proportion_hours_map_lt_65_h1_h6": (
                float((map_values < 65).mean()) if not map_values.empty else np.nan
            ),
            # AUCs
            "dbp_auc_h1_h6": trapezoidal_auc(
                group,
                "dbp",
            ),
            "map_auc_h1_h6": trapezoidal_auc(
                group,
                "map",
            ),
            "norepi_auc_h1_h6": trapezoidal_auc(
                group,
                "norepi_ug_min",
            ),
            # Vasopressor persistence
            "norepi_present_last_observed_hour": (
                int(pd.notna(norepi_at_last_hour) and norepi_at_last_hour > 0)
                if pd.notna(last_hour)
                else np.nan
            ),
            "last_observed_hour": last_hour,
            "n_hourly_dbp_values_h1_h6": int(dbp_values.notna().sum()),
            "n_hourly_map_values_h1_h6": int(map_values.notna().sum()),
            "n_hourly_norepi_values_h1_h6": int(ne_values.notna().sum()),
        }
    )

longitudinal_metrics = pd.DataFrame(longitudinal_rows)

# ------------------------------------------------------------
# 20. CHANGE FROM BASELINE TO FOLLOW-UP
# ------------------------------------------------------------

longitudinal_metrics = longitudinal_metrics.merge(
    baseline_hemodynamics[
        [
            "patientid",
            "dbp_baseline",
            "map_baseline",
            "pulse_pressure_baseline",
        ]
    ],
    on="patientid",
    how="left",
)

longitudinal_metrics["delta_dbp_followup_minus_baseline"] = (
    longitudinal_metrics["dbp_median_h1_h6"] - longitudinal_metrics["dbp_baseline"]
)

longitudinal_metrics["delta_map_followup_minus_baseline"] = (
    longitudinal_metrics["map_median_h1_h6"] - longitudinal_metrics["map_baseline"]
)

longitudinal_metrics["delta_pp_followup_minus_baseline"] = (
    longitudinal_metrics["pulse_pressure_median_h1_h6"]
    - longitudinal_metrics["pulse_pressure_baseline"]
)

longitudinal_metrics = longitudinal_metrics.drop(
    columns=[
        "dbp_baseline",
        "map_baseline",
        "pulse_pressure_baseline",
    ],
    errors="ignore",
)

# ------------------------------------------------------------
# 21. BUILD FINAL PATIENT-LEVEL DATASET
# ------------------------------------------------------------

analysis_dataset = (
    broad_cohort.merge(
        baseline_hemodynamics,
        on="patientid",
        how="left",
    )
    .merge(
        lactate_baseline,
        on=[
            "patientid",
            "lactate_time",
            "lactate_hours_from_t0",
            "lactate_variableid",
            "lactate_gt_2",
        ],
        how="left",
    )
    .merge(
        norepi_baseline,
        on="patientid",
        how="left",
    )
    .merge(
        longitudinal_metrics,
        on="patientid",
        how="left",
    )
)

# ------------------------------------------------------------
# 22. PHENOTYPE-DERIVATION COMPLETENESS
# ------------------------------------------------------------

PHENOTYPE_VARIABLES = [
    "dbp_baseline",
    "pulse_pressure_baseline",
    "heart_rate_baseline",
    "norepi_ug_min_baseline",
    "lactate_baseline",
]

analysis_dataset["n_complete_phenotype_variables"] = (
    analysis_dataset[PHENOTYPE_VARIABLES].notna().sum(axis=1)
)

analysis_dataset["complete_all_phenotype_variables"] = (
    analysis_dataset[PHENOTYPE_VARIABLES].notna().all(axis=1)
)

analysis_dataset["complete_core_hemodynamic_variables"] = (
    analysis_dataset[
        [
            "dbp_baseline",
            "pulse_pressure_baseline",
            "heart_rate_baseline",
            "norepi_ug_min_baseline",
        ]
    ]
    .notna()
    .all(axis=1)
)

# ------------------------------------------------------------
# 23. COMPLETENESS TABLE
# ------------------------------------------------------------

completeness_rows = []

for variable in PHENOTYPE_VARIABLES + [
    "shock_index_baseline",
    "modified_shock_index_baseline",
    "diastolic_shock_index_baseline",
    "dbp_slope_h1_h6",
    "map_slope_h1_h6",
    "norepi_slope_h1_h6",
    "proportion_hours_dbp_lt_50_h1_h6",
    "proportion_hours_map_lt_65_h1_h6",
    "norepi_auc_h1_h6",
]:

    if variable not in analysis_dataset.columns:
        continue

    n_available = int(analysis_dataset[variable].notna().sum())

    completeness_rows.append(
        {
            "variable": variable,
            "n_available": n_available,
            "n_total": len(analysis_dataset),
            "percent_available": (100 * n_available / len(analysis_dataset)),
        }
    )

completeness = pd.DataFrame(completeness_rows)

print("\n" + "=" * 80)
print("ANALYSIS DATASET COMPLETENESS")
print("=" * 80)

display(completeness)

# ------------------------------------------------------------
# 24. BASELINE DISTRIBUTION AUDIT
# ------------------------------------------------------------

distribution_variables = [
    "sbp_baseline",
    "dbp_baseline",
    "map_baseline",
    "pulse_pressure_baseline",
    "heart_rate_baseline",
    "shock_index_baseline",
    "modified_shock_index_baseline",
    "diastolic_shock_index_baseline",
    "norepi_ug_min_baseline",
    "lactate_baseline",
]

distribution_rows = []

for variable in distribution_variables:

    if variable not in analysis_dataset.columns:
        continue

    values = pd.to_numeric(
        analysis_dataset[variable],
        errors="coerce",
    ).dropna()

    if values.empty:
        continue

    distribution_rows.append(
        {
            "variable": variable,
            "n": int(values.size),
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
    )

baseline_distribution = pd.DataFrame(distribution_rows)

print("\n" + "=" * 80)
print("BASELINE DISTRIBUTION AUDIT")
print("=" * 80)

display(baseline_distribution)

# ------------------------------------------------------------
# 25. FINAL COUNTS
# ------------------------------------------------------------

final_counts = pd.DataFrame(
    {
        "criterion": [
            "Broad cohort",
            "Strict cohort",
            "Complete core hemodynamic phenotype variables",
            "Complete all five phenotype variables",
            "At least 4 of 5 phenotype variables",
        ],
        "n": [
            int(len(analysis_dataset)),
            int(analysis_dataset["probable_septic_shock_strict"].sum()),
            int(analysis_dataset["complete_core_hemodynamic_variables"].sum()),
            int(analysis_dataset["complete_all_phenotype_variables"].sum()),
            int(analysis_dataset["n_complete_phenotype_variables"].ge(4).sum()),
        ],
    }
)

print("\n" + "=" * 80)
print("FINAL ANALYSIS COUNTS")
print("=" * 80)

display(final_counts)

# ------------------------------------------------------------
# 26. SAVE OUTPUTS
# ------------------------------------------------------------

analysis_dataset.to_parquet(
    DERIVED_DIR / "step03_analysis_ready_hemodynamic_dataset.parquet",
    index=False,
)

analysis_dataset.to_csv(
    TABLE_DIR / "step03_analysis_ready_hemodynamic_dataset.csv",
    index=False,
)

hourly.to_parquet(
    DERIVED_DIR / "step03_hourly_hemodynamic_trajectories.parquet",
    index=False,
)

baseline_hemodynamics.to_parquet(
    DERIVED_DIR / "step03_baseline_hemodynamics_0_1h.parquet",
    index=False,
)

norepi_baseline.to_parquet(
    DERIVED_DIR / "step03_baseline_norepinephrine_0_1h.parquet",
    index=False,
)

longitudinal_metrics.to_parquet(
    DERIVED_DIR / "step03_longitudinal_validation_metrics_h1_h6.parquet",
    index=False,
)

completeness.to_csv(
    TABLE_DIR / "step03_variable_completeness.csv",
    index=False,
)

baseline_distribution.to_csv(
    TABLE_DIR / "step03_baseline_distribution_audit.csv",
    index=False,
)

final_counts.to_csv(
    TABLE_DIR / "step03_final_analysis_counts.csv",
    index=False,
)

run_summary = {
    "broad_cohort_n": int(len(analysis_dataset)),
    "strict_cohort_n": int(analysis_dataset["probable_septic_shock_strict"].sum()),
    "complete_core_hemodynamic_n": int(
        analysis_dataset["complete_core_hemodynamic_variables"].sum()
    ),
    "complete_all_five_variables_n": int(
        analysis_dataset["complete_all_phenotype_variables"].sum()
    ),
    "at_least_four_of_five_n": int(
        analysis_dataset["n_complete_phenotype_variables"].ge(4).sum()
    ),
    "phenotype_variables": PHENOTYPE_VARIABLES,
    "phenotype_window": "0 to 1 hour after time zero",
    "validation_window": (">1 to 6 hours after time zero"),
}

with open(
    LOG_DIR / "step03_analysis_ready_dataset_summary.json",
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
print("STEP 03 COMPLETE")
print("=" * 80)

print(
    "\nMain patient-level dataset:\n"
    f"{DERIVED_DIR / 'step03_analysis_ready_hemodynamic_dataset.parquet'}"
)

print(
    "\nHourly longitudinal dataset:\n"
    f"{DERIVED_DIR / 'step03_hourly_hemodynamic_trajectories.parquet'}"
)
