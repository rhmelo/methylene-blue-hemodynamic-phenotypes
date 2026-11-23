"""Source: original notebook cell 11. Run through run.py."""

from common import display, phase_dir

# ============================================================
# STEP 03B — Clean reconstructed norepinephrine dose
# ============================================================

import pandas as pd
import numpy as np

PROJECT_DIR = phase_dir("hirid")

TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
DERIVED_DIR = PROJECT_DIR / "derived_data"

ANALYSIS_PATH = DERIVED_DIR / "step03_analysis_ready_hemodynamic_dataset.parquet"

HOURLY_PATH = DERIVED_DIR / "step03_hourly_hemodynamic_trajectories.parquet"

analysis_dataset = pd.read_parquet(ANALYSIS_PATH)
hourly = pd.read_parquet(HOURLY_PATH)

# ------------------------------------------------------------
# 1. AUDIT BASELINE NOREPINEPHRINE
# ------------------------------------------------------------

baseline_ne = pd.to_numeric(analysis_dataset["norepi_ug_min_baseline"], errors="coerce")

audit_before = pd.DataFrame(
    {
        "statistic": [
            "n available",
            "n equal to zero",
            "n greater than 20 µg/min",
            "n greater than 50 µg/min",
            "minimum",
            "p01",
            "p05",
            "p25",
            "median",
            "p75",
            "p95",
            "p99",
            "maximum",
        ],
        "value": [
            baseline_ne.notna().sum(),
            baseline_ne.eq(0).sum(),
            baseline_ne.gt(20).sum(),
            baseline_ne.gt(50).sum(),
            baseline_ne.min(),
            baseline_ne.quantile(0.01),
            baseline_ne.quantile(0.05),
            baseline_ne.quantile(0.25),
            baseline_ne.median(),
            baseline_ne.quantile(0.75),
            baseline_ne.quantile(0.95),
            baseline_ne.quantile(0.99),
            baseline_ne.max(),
        ],
    }
)

print("=" * 80)
print("BASELINE NOREPINEPHRINE AUDIT — BEFORE CLEANING")
print("=" * 80)

display(audit_before)

# ------------------------------------------------------------
# 2. DEFINE DATA-DRIVEN EXTREME LIMIT
# ------------------------------------------------------------

positive_ne = baseline_ne.loc[baseline_ne.gt(0)].dropna()

if positive_ne.empty:
    raise RuntimeError("No positive baseline norepinephrine values found.")

upper_limit_p995 = positive_ne.quantile(0.995)

print(
    "\n99.5th percentile of positive baseline dose:",
    round(float(upper_limit_p995), 3),
    "µg/min",
)

# ------------------------------------------------------------
# 3. CLEAN BASELINE DOSE
# ------------------------------------------------------------

analysis_dataset["norepi_ug_min_baseline_raw"] = baseline_ne

analysis_dataset["norepi_ug_min_baseline_clean"] = baseline_ne.copy()

# Zero or negative reconstructed rates do not represent
# an active therapeutic dose for phenotype derivation.
analysis_dataset.loc[
    analysis_dataset["norepi_ug_min_baseline_clean"].le(0),
    "norepi_ug_min_baseline_clean",
] = np.nan

# Winsorize extreme values at the empirical 99.5th percentile.
analysis_dataset["norepi_ug_min_baseline_winsor"] = analysis_dataset[
    "norepi_ug_min_baseline_clean"
].clip(upper=upper_limit_p995)

# Log-transformed variable for clustering/modeling.
analysis_dataset["log_norepi_baseline"] = np.log(
    analysis_dataset["norepi_ug_min_baseline_winsor"] + 0.1
)

# ------------------------------------------------------------
# 4. CLEAN HOURLY NOREPINEPHRINE
# ------------------------------------------------------------

hourly["norepi_ug_min_raw"] = hourly["norepi_ug_min"]

hourly["norepi_ug_min_clean"] = hourly["norepi_ug_min"]

hourly.loc[
    hourly["norepi_ug_min_clean"].le(0),
    "norepi_ug_min_clean",
] = np.nan

hourly["norepi_ug_min_winsor"] = hourly["norepi_ug_min_clean"].clip(
    upper=upper_limit_p995
)

# ------------------------------------------------------------
# 5. AUDIT AFTER CLEANING
# ------------------------------------------------------------

clean_ne = analysis_dataset["norepi_ug_min_baseline_winsor"]

audit_after = pd.DataFrame(
    {
        "statistic": [
            "n available",
            "percent available",
            "minimum",
            "p01",
            "p05",
            "p25",
            "median",
            "p75",
            "p95",
            "p99",
            "maximum after winsorization",
        ],
        "value": [
            clean_ne.notna().sum(),
            100 * clean_ne.notna().mean(),
            clean_ne.min(),
            clean_ne.quantile(0.01),
            clean_ne.quantile(0.05),
            clean_ne.quantile(0.25),
            clean_ne.median(),
            clean_ne.quantile(0.75),
            clean_ne.quantile(0.95),
            clean_ne.quantile(0.99),
            clean_ne.max(),
        ],
    }
)

print("\n" + "=" * 80)
print("BASELINE NOREPINEPHRINE AUDIT — AFTER CLEANING")
print("=" * 80)

display(audit_after)

# ------------------------------------------------------------
# 6. UPDATE PHENOTYPE COMPLETENESS
# ------------------------------------------------------------

phenotype_variables_clean = [
    "dbp_baseline",
    "pulse_pressure_baseline",
    "heart_rate_baseline",
    "log_norepi_baseline",
    "lactate_baseline",
]

analysis_dataset["n_complete_phenotype_variables_clean"] = (
    analysis_dataset[phenotype_variables_clean].notna().sum(axis=1)
)

analysis_dataset["complete_all_phenotype_variables_clean"] = (
    analysis_dataset[phenotype_variables_clean].notna().all(axis=1)
)

analysis_dataset["complete_core_hemodynamic_variables_clean"] = (
    analysis_dataset[
        [
            "dbp_baseline",
            "pulse_pressure_baseline",
            "heart_rate_baseline",
            "log_norepi_baseline",
        ]
    ]
    .notna()
    .all(axis=1)
)

updated_counts = pd.DataFrame(
    {
        "criterion": [
            "Broad cohort",
            "Complete core phenotype variables after NE cleaning",
            "Complete all five phenotype variables after NE cleaning",
            "At least 4 of 5 phenotype variables after NE cleaning",
        ],
        "n": [
            len(analysis_dataset),
            int(analysis_dataset["complete_core_hemodynamic_variables_clean"].sum()),
            int(analysis_dataset["complete_all_phenotype_variables_clean"].sum()),
            int(analysis_dataset["n_complete_phenotype_variables_clean"].ge(4).sum()),
        ],
    }
)

print("\n" + "=" * 80)
print("UPDATED PHENOTYPE COUNTS")
print("=" * 80)

display(updated_counts)

# ------------------------------------------------------------
# 7. SAVE
# ------------------------------------------------------------

analysis_dataset.to_parquet(
    DERIVED_DIR / "step03b_analysis_ready_dataset_clean_norepinephrine.parquet",
    index=False,
)

hourly.to_parquet(
    DERIVED_DIR / "step03b_hourly_trajectories_clean_norepinephrine.parquet",
    index=False,
)

audit_before.to_csv(
    TABLE_DIR / "step03b_norepinephrine_audit_before_cleaning.csv",
    index=False,
)

audit_after.to_csv(
    TABLE_DIR / "step03b_norepinephrine_audit_after_cleaning.csv",
    index=False,
)

updated_counts.to_csv(
    TABLE_DIR / "step03b_updated_phenotype_counts.csv",
    index=False,
)

print("\nSTEP 03B COMPLETE")
