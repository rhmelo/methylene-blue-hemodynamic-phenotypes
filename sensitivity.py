"""Source: revision notebook cells 90–94 (93 supplies covariate preparation only). Run through run.py."""

from common import artifact, display, phase_dir

import pandas as pd

PROJECT_DIR = phase_dir("mimic_iv")
REVISION_DIR = PROJECT_DIR / "revision_analyses"
corrected_data = pd.read_csv(REVISION_DIR / "GEE_model_data_without_post_death.csv.gz")
baseline = pd.read_csv(
    artifact(PROJECT_DIR, "METHYLENE_BLUE_ANALYTIC_COHORT_FROZEN_FINAL_*.csv.gz")
)
bp_data = pd.read_csv(
    artifact(PROJECT_DIR, "TABLE1_BASE_FINAL_combined_bp_phenotype_*.csv.gz")
)
primary = pd.read_csv(
    REVISION_DIR / "GEE_global_interaction_original_vs_no_post_death.csv"
).iloc[1]
# ============================================================
# 16F — SENSITIVITY ANALYSIS: INVASIVE ARTERIAL PRESSURE ONLY
#
# Purpose:
# Repeat the primary longitudinal GEE after excluding patients
# whose pre-MB phenotype classification relied on non-invasive
# blood pressure measurements.
#
# Same outcome, transformation, time structure and GEE
# specification as the primary analysis.
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ------------------------------------------------------------
# 1. OUTPUT DIRECTORY
# ------------------------------------------------------------

REVISION_DIR = PROJECT_DIR / "revision_analyses"

REVISION_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. AUDIT BP MODALITY
# ------------------------------------------------------------

print("=" * 80)
print("16F — INVASIVE-ONLY SENSITIVITY ANALYSIS")
print("=" * 80)

print("\nBP modality in the 59-patient analytic cohort:")

bp_modality_audit = (
    baseline["bp_modality_selected"]
    .value_counts(dropna=False)
    .rename_axis("bp_modality_selected")
    .reset_index(name="n")
)

bp_modality_audit["percent"] = 100 * bp_modality_audit["n"] / len(baseline)

display(bp_modality_audit)

# ------------------------------------------------------------
# 3. IDENTIFY INVASIVE PATIENTS
# ------------------------------------------------------------

print(
    "\nRaw values of bp_modality_selected:",
    baseline["bp_modality_selected"].dropna().unique(),
)

# Determine invasive label robustly
modality_text = baseline["bp_modality_selected"].astype(str).str.strip().str.lower()

invasive_mask = modality_text.str.contains(
    "invasive",
    na=False,
) & ~modality_text.str.contains(
    "non",
    na=False,
)

invasive_patients = (
    baseline.loc[
        invasive_mask,
        [
            "subject_id",
            "stay_id",
            "hadm_id",
            "hemodynamic_phenotype",
            "bp_modality_selected",
        ],
    ]
    .drop_duplicates(
        subset=[
            "subject_id",
            "stay_id",
        ]
    )
    .copy()
)

print("\nPatients classified using invasive BP:", len(invasive_patients))

print(
    "Patients excluded because classification was not invasive:",
    len(baseline) - len(invasive_patients),
)

print("\nPhenotype distribution among invasive-only patients:")

invasive_phenotype_counts = (
    invasive_patients["hemodynamic_phenotype"]
    .value_counts()
    .rename_axis("phenotype")
    .reset_index(name="n")
)

display(invasive_phenotype_counts)

# ------------------------------------------------------------
# 4. RESTRICT CORRECTED LONGITUDINAL DATASET
# ------------------------------------------------------------

invasive_ids = set(invasive_patients["stay_id"])

gee_invasive = corrected_data.loc[corrected_data["stay_id"].isin(invasive_ids)].copy()

print("\nLongitudinal invasive-only dataset:")

print("Patients:", gee_invasive["stay_id"].nunique())

print("Observations:", len(gee_invasive))

print("\nPatients contributing at each time point:")

invasive_by_time = (
    gee_invasive.groupby(
        [
            "hour_index",
            "interval_label",
        ],
        observed=True,
    )
    .agg(
        n_patients=(
            "stay_id",
            "nunique",
        ),
        n_observations=(
            "stay_id",
            "size",
        ),
    )
    .reset_index()
)

display(invasive_by_time)

# ------------------------------------------------------------
# 5. ENSURE MODEL VARIABLES ARE CORRECT
# ------------------------------------------------------------

gee_invasive["hour_index"] = pd.to_numeric(
    gee_invasive["hour_index"],
    errors="coerce",
)

gee_invasive["nee_median"] = pd.to_numeric(
    gee_invasive["nee_median"],
    errors="coerce",
)

gee_invasive["log1p_nee"] = np.log1p(gee_invasive["nee_median"])

# Explicit reference category:
# Preserved diastolic pressure

PHENOTYPE_REFERENCE = "Preserved diastolic pressure"

phenotype_categories = [
    "Preserved diastolic pressure",
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
]

gee_invasive["phenotype_model"] = pd.Categorical(
    gee_invasive["hemodynamic_phenotype"],
    categories=phenotype_categories,
    ordered=False,
)

# Treat time categorically, exactly as in primary model

gee_invasive["time_factor"] = pd.Categorical(
    gee_invasive["hour_index"],
    categories=[
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
    ],
    ordered=False,
)

# ------------------------------------------------------------
# 6. FIT FULL GEE WITH PHENOTYPE × TIME
# ------------------------------------------------------------

formula_full = (
    "log1p_nee ~ "
    "C(phenotype_model, Treatment(reference='Preserved diastolic pressure')) "
    "* C(time_factor, Treatment(reference=-1))"
)

gee_invasive_full = smf.gee(
    formula=formula_full,
    groups="subject_id",
    data=gee_invasive,
    family=sm.families.Gaussian(sm.families.links.Identity()),
    cov_struct=sm.cov_struct.Exchangeable(),
)

gee_invasive_full_result = gee_invasive_full.fit()

print("\n" + "=" * 80)
print("INVASIVE-ONLY GEE — FULL MODEL")
print("=" * 80)

print(gee_invasive_full_result.summary())

# ------------------------------------------------------------
# 7. FIT REDUCED MODEL WITHOUT INTERACTION
# ------------------------------------------------------------

formula_reduced = (
    "log1p_nee ~ "
    "C(phenotype_model, Treatment(reference='Preserved diastolic pressure')) "
    "+ C(time_factor, Treatment(reference=-1))"
)

gee_invasive_reduced = smf.gee(
    formula=formula_reduced,
    groups="subject_id",
    data=gee_invasive,
    family=sm.families.Gaussian(sm.families.links.Identity()),
    cov_struct=sm.cov_struct.Exchangeable(),
)

gee_invasive_reduced_result = gee_invasive_reduced.fit()

# ------------------------------------------------------------
# 8. GLOBAL PHENOTYPE × TIME INTERACTION WALD TEST
# ------------------------------------------------------------

interaction_terms = [
    term
    for term in gee_invasive_full_result.params.index
    if ":" in term and "phenotype_model" in term and "time_factor" in term
]

print("\nNumber of phenotype × time interaction terms:", len(interaction_terms))

if len(interaction_terms) == 0:

    raise RuntimeError("No interaction coefficients were identified.")

parameter_names = list(gee_invasive_full_result.params.index)

interaction_indices = [parameter_names.index(term) for term in interaction_terms]

R = np.zeros(
    (
        len(interaction_indices),
        len(parameter_names),
    )
)

for row_index, parameter_index in enumerate(interaction_indices):

    R[row_index, parameter_index] = 1

wald_result = gee_invasive_full_result.wald_test(
    R,
    scalar=True,
)

interaction_chi2 = float(np.asarray(wald_result.statistic).squeeze())

interaction_df = len(interaction_terms)

interaction_p = float(np.asarray(wald_result.pvalue).squeeze())

print("\n" + "=" * 80)
print("GLOBAL PHENOTYPE × TIME INTERACTION")
print("=" * 80)

print(f"Wald χ²({interaction_df}) = " f"{interaction_chi2:.3f}")

print(f"P = {interaction_p:.6f}")

# ------------------------------------------------------------
# 9. COMPARE WITH PRIMARY RESULT
# ------------------------------------------------------------

PRIMARY_INTERACTION_P = float(primary["p_value"])

comparison = pd.DataFrame(
    {
        "analysis": [
            "Primary GEE",
            "Invasive-only GEE",
        ],
        "n_patients": [
            corrected_data["subject_id"].nunique(),
            gee_invasive["stay_id"].nunique(),
        ],
        "n_observations": [
            len(corrected_data),
            len(gee_invasive),
        ],
        "phenotype_time_interaction_p": [
            PRIMARY_INTERACTION_P,
            interaction_p,
        ],
    }
)

print("\n" + "=" * 80)
print("PRIMARY VS INVASIVE-ONLY")
print("=" * 80)

display(comparison)

# ------------------------------------------------------------
# 10. MODEL COEFFICIENTS
# ------------------------------------------------------------

coefficients = pd.DataFrame(
    {
        "term": gee_invasive_full_result.params.index,
        "estimate": gee_invasive_full_result.params.values,
        "standard_error": gee_invasive_full_result.bse.values,
        "z": gee_invasive_full_result.tvalues.values,
        "p_value": gee_invasive_full_result.pvalues.values,
    }
)

conf_int = gee_invasive_full_result.conf_int()

coefficients["ci_lower"] = conf_int[0].values

coefficients["ci_upper"] = conf_int[1].values

# ------------------------------------------------------------
# 11. SAVE OUTPUTS
# ------------------------------------------------------------

bp_modality_audit.to_csv(
    REVISION_DIR / "sensitivity_invasive_only_BP_modality_audit.csv",
    index=False,
)

invasive_phenotype_counts.to_csv(
    REVISION_DIR / "sensitivity_invasive_only_phenotype_counts.csv",
    index=False,
)

invasive_by_time.to_csv(
    REVISION_DIR / "sensitivity_invasive_only_patients_by_time.csv",
    index=False,
)

comparison.to_csv(
    REVISION_DIR / "sensitivity_primary_vs_invasive_only.csv",
    index=False,
)

coefficients.to_csv(
    REVISION_DIR / "sensitivity_invasive_only_GEE_coefficients.csv",
    index=False,
)

gee_invasive.to_csv(
    REVISION_DIR / "sensitivity_invasive_only_GEE_dataset.csv.gz",
    index=False,
    compression="gzip",
)

with open(
    REVISION_DIR / "sensitivity_invasive_only_GEE_summary.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(gee_invasive_full_result.summary().as_text())

    f.write("\n\nGLOBAL PHENOTYPE × TIME INTERACTION\n")

    f.write(f"Wald chi-square({interaction_df}) = " f"{interaction_chi2:.6f}\n")

    f.write(f"P = {interaction_p:.8f}\n")

# ------------------------------------------------------------
# 12. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("16F COMPLETE")
print("=" * 80)

print("Invasive-only patients:", gee_invasive["stay_id"].nunique())

print("Longitudinal observations:", len(gee_invasive))

print("Phenotype × time P:", interaction_p)

print("\nOutputs saved to:\n", REVISION_DIR)
invasive_result = {
    "n_patients": gee_invasive["stay_id"].nunique(),
    "n_observations": int(gee_invasive_full_result.nobs),
    "wald_chi2": interaction_chi2,
    "df": interaction_df,
    "phenotype_time_interaction_p": interaction_p,
}
# ============================================================
# 16G — SENSITIVITY ANALYSIS:
# COMPLETE 6-HOUR LONGITUDINAL FOLLOW-UP
#
# Purpose:
# Repeat the primary GEE among patients contributing valid
# observations at ALL seven model timepoints:
#
# - Baseline (-1 to 0 h)
# - 0 to <1 h
# - 1 to <2 h
# - 2 to <3 h
# - 3 to <4 h
# - 4 to <5 h
# - 5 to <6 h
#
# This directly evaluates robustness to longitudinal dropout.
#
# IMPORTANT:
# Uses corrected_data, i.e. the dataset after exclusion of
# intervals occurring entirely after death.
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ------------------------------------------------------------
# 1. OUTPUT DIRECTORY
# ------------------------------------------------------------

REVISION_DIR = PROJECT_DIR / "revision_analyses"

REVISION_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("16G — COMPLETE 6-HOUR FOLLOW-UP SENSITIVITY")
print("=" * 80)

# ------------------------------------------------------------
# 2. DEFINE REQUIRED MODEL TIMEPOINTS
# ------------------------------------------------------------

REQUIRED_TIMEPOINTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
]

print("\nRequired timepoints:", REQUIRED_TIMEPOINTS)

# ------------------------------------------------------------
# 3. AUDIT TIMEPOINT COMPLETENESS PER PATIENT
# ------------------------------------------------------------

timepoint_audit = (
    corrected_data.groupby(
        [
            "subject_id",
            "stay_id",
            "hemodynamic_phenotype",
        ],
        observed=True,
    )
    .agg(
        n_timepoints=(
            "hour_index",
            "nunique",
        ),
        min_hour_index=(
            "hour_index",
            "min",
        ),
        max_hour_index=(
            "hour_index",
            "max",
        ),
        n_observations=(
            "hour_index",
            "size",
        ),
    )
    .reset_index()
)

# Explicitly verify presence of each required timepoint
for timepoint in REQUIRED_TIMEPOINTS:

    patient_ids_at_time = set(
        corrected_data.loc[
            corrected_data["hour_index"].eq(timepoint),
            "stay_id",
        ]
    )

    timepoint_audit[f"has_time_{timepoint}"] = timepoint_audit["stay_id"].isin(
        patient_ids_at_time
    )

required_columns = [f"has_time_{timepoint}" for timepoint in REQUIRED_TIMEPOINTS]

timepoint_audit["complete_6h_followup"] = timepoint_audit[required_columns].all(axis=1)

print("\nPatient-level completeness:")

display(
    timepoint_audit[
        [
            "subject_id",
            "stay_id",
            "hemodynamic_phenotype",
            "n_timepoints",
            "min_hour_index",
            "max_hour_index",
            "complete_6h_followup",
        ]
    ]
)

# ------------------------------------------------------------
# 4. NUMBER OF COMPLETE PATIENTS
# ------------------------------------------------------------

n_total = len(timepoint_audit)

n_complete = int(timepoint_audit["complete_6h_followup"].sum())

n_incomplete = n_total - n_complete

print("\n" + "=" * 80)
print("COMPLETE FOLLOW-UP SUMMARY")
print("=" * 80)

print("Total patients:", n_total)

print("Complete through all 7 model timepoints:", n_complete)

print("Incomplete:", n_incomplete)

print(
    "Percent complete:",
    round(
        100 * n_complete / n_total,
        1,
    ),
)

# ------------------------------------------------------------
# 5. PHENOTYPE DISTRIBUTION
# ------------------------------------------------------------

complete_patient_table = timepoint_audit.loc[
    timepoint_audit["complete_6h_followup"]
].copy()

complete_phenotype_counts = (
    complete_patient_table["hemodynamic_phenotype"]
    .value_counts()
    .rename_axis("phenotype")
    .reset_index(name="n")
)

complete_phenotype_counts["percent"] = 100 * complete_phenotype_counts["n"] / n_complete

print("\nPhenotype distribution among " "complete-follow-up patients:")

display(complete_phenotype_counts)

# ------------------------------------------------------------
# 6. AUDIT EXCLUDED PATIENTS
# ------------------------------------------------------------

excluded_patients = timepoint_audit.loc[~timepoint_audit["complete_6h_followup"]].copy()

print("\nPatients excluded because of incomplete " "6-hour longitudinal follow-up:")

display(
    excluded_patients[
        [
            "subject_id",
            "stay_id",
            "hemodynamic_phenotype",
            "n_timepoints",
            "min_hour_index",
            "max_hour_index",
        ]
    ]
)

# ------------------------------------------------------------
# 7. RESTRICT LONGITUDINAL DATASET
# ------------------------------------------------------------

complete_stay_ids = set(complete_patient_table["stay_id"])

gee_complete = corrected_data.loc[
    corrected_data["stay_id"].isin(complete_stay_ids)
].copy()

print("\nComplete-follow-up GEE dataset:")

print("Patients:", gee_complete["stay_id"].nunique())

print("Observations:", len(gee_complete))

# Every patient should have exactly 7 observations
observation_count_check = gee_complete.groupby("stay_id").size()

print("\nObservations per patient:")

display(
    observation_count_check.value_counts()
    .sort_index()
    .rename_axis("observations_per_patient")
    .reset_index(name="n_patients")
)

# ------------------------------------------------------------
# 8. PATIENTS AT EACH TIMEPOINT
# ------------------------------------------------------------

complete_by_time = (
    gee_complete.groupby(
        [
            "hour_index",
            "interval_label",
        ],
        observed=True,
    )
    .agg(
        n_patients=(
            "stay_id",
            "nunique",
        ),
        n_observations=(
            "stay_id",
            "size",
        ),
        nee_median=(
            "nee_median",
            "median",
        ),
    )
    .reset_index()
)

print("\nPatients contributing at each time point:")

display(complete_by_time)

# ------------------------------------------------------------
# 9. PREPARE MODEL VARIABLES
# ------------------------------------------------------------

gee_complete["hour_index"] = pd.to_numeric(
    gee_complete["hour_index"],
    errors="coerce",
)

gee_complete["nee_median"] = pd.to_numeric(
    gee_complete["nee_median"],
    errors="coerce",
)

gee_complete["log1p_nee"] = np.log1p(gee_complete["nee_median"])

phenotype_categories = [
    "Preserved diastolic pressure",
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
]

gee_complete["phenotype_model"] = pd.Categorical(
    gee_complete["hemodynamic_phenotype"],
    categories=phenotype_categories,
    ordered=False,
)

gee_complete["time_factor"] = pd.Categorical(
    gee_complete["hour_index"],
    categories=REQUIRED_TIMEPOINTS,
    ordered=False,
)

# ------------------------------------------------------------
# 10. FIT PRIMARY-SPECIFICATION GEE
# ------------------------------------------------------------

formula_complete = (
    "log1p_nee ~ "
    "C(phenotype_model, "
    "Treatment(reference='Preserved diastolic pressure')) "
    "* C(time_factor, Treatment(reference=-1))"
)

gee_complete_model = smf.gee(
    formula=formula_complete,
    groups="subject_id",
    data=gee_complete,
    family=sm.families.Gaussian(sm.families.links.Identity()),
    cov_struct=sm.cov_struct.Exchangeable(),
)

gee_complete_result = gee_complete_model.fit()

print("\n" + "=" * 80)
print("COMPLETE 6-HOUR FOLLOW-UP GEE")
print("=" * 80)

print(gee_complete_result.summary())

# ------------------------------------------------------------
# 11. GLOBAL PHENOTYPE × TIME WALD TEST
# ------------------------------------------------------------

interaction_terms = [
    term
    for term in gee_complete_result.params.index
    if (":" in term and "phenotype_model" in term and "time_factor" in term)
]

print("\nNumber of phenotype × time " "interaction terms:", len(interaction_terms))

if len(interaction_terms) == 0:

    raise RuntimeError("No phenotype × time interaction " "coefficients identified.")

parameter_names = list(gee_complete_result.params.index)

interaction_indices = [parameter_names.index(term) for term in interaction_terms]

R = np.zeros(
    (
        len(interaction_indices),
        len(parameter_names),
    )
)

for row_index, parameter_index in enumerate(interaction_indices):

    R[row_index, parameter_index] = 1

wald_result = gee_complete_result.wald_test(
    R,
    scalar=True,
)

interaction_chi2 = float(np.asarray(wald_result.statistic).squeeze())

interaction_df = len(interaction_terms)

interaction_p = float(np.asarray(wald_result.pvalue).squeeze())

print("\n" + "=" * 80)
print("GLOBAL PHENOTYPE × TIME INTERACTION")
print("=" * 80)

print(f"Wald χ²({interaction_df}) = " f"{interaction_chi2:.3f}")

print(f"P = {interaction_p:.6f}")

# ------------------------------------------------------------
# 12. PRIMARY / INVASIVE / COMPLETE COMPARISON
# ------------------------------------------------------------

# We already know the first two results.
# This creates the accumulating robustness table.

robustness_comparison = pd.DataFrame(
    {
        "analysis": [
            "Primary GEE",
            "Invasive-only GEE",
            "Complete 6-h follow-up GEE",
        ],
        "n_patients": [
            corrected_data["subject_id"].nunique(),
            invasive_result["n_patients"],
            gee_complete["stay_id"].nunique(),
        ],
        "n_observations": [
            len(corrected_data),
            invasive_result["n_observations"],
            len(gee_complete),
        ],
        "wald_chi2": [
            np.nan,
            invasive_result["wald_chi2"],
            interaction_chi2,
        ],
        "df": [
            12,
            12,
            interaction_df,
        ],
        "phenotype_time_interaction_p": [
            primary["p_value"],
            invasive_result["phenotype_time_interaction_p"],
            interaction_p,
        ],
    }
)

print("\n" + "=" * 80)
print("ROBUSTNESS ANALYSES — CURRENT RESULTS")
print("=" * 80)

display(robustness_comparison)

# ------------------------------------------------------------
# 13. COEFFICIENT TABLE
# ------------------------------------------------------------

coefficients_complete = pd.DataFrame(
    {
        "term": gee_complete_result.params.index,
        "estimate": gee_complete_result.params.values,
        "standard_error": gee_complete_result.bse.values,
        "z": gee_complete_result.tvalues.values,
        "p_value": gee_complete_result.pvalues.values,
    }
)

confidence_intervals = gee_complete_result.conf_int()

coefficients_complete["ci_lower"] = confidence_intervals[0].values

coefficients_complete["ci_upper"] = confidence_intervals[1].values

# ------------------------------------------------------------
# 14. SAVE OUTPUTS
# ------------------------------------------------------------

timepoint_audit.to_csv(
    REVISION_DIR / "sensitivity_complete_6h_patient_audit.csv",
    index=False,
)

complete_phenotype_counts.to_csv(
    REVISION_DIR / "sensitivity_complete_6h_phenotype_counts.csv",
    index=False,
)

excluded_patients.to_csv(
    REVISION_DIR / "sensitivity_complete_6h_excluded_patients.csv",
    index=False,
)

complete_by_time.to_csv(
    REVISION_DIR / "sensitivity_complete_6h_patients_by_time.csv",
    index=False,
)

gee_complete.to_csv(
    REVISION_DIR / "sensitivity_complete_6h_GEE_dataset.csv.gz",
    index=False,
    compression="gzip",
)

coefficients_complete.to_csv(
    REVISION_DIR / "sensitivity_complete_6h_GEE_coefficients.csv",
    index=False,
)

robustness_comparison.to_csv(
    REVISION_DIR / "robustness_analyses_interim_summary.csv",
    index=False,
)

with open(
    REVISION_DIR / "sensitivity_complete_6h_GEE_summary.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(gee_complete_result.summary().as_text())

    f.write("\n\nGLOBAL PHENOTYPE × TIME INTERACTION\n")

    f.write(f"Wald chi-square({interaction_df}) = " f"{interaction_chi2:.6f}\n")

    f.write(f"P = {interaction_p:.8f}\n")

# ------------------------------------------------------------
# 15. FINAL DISPLAY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("16G COMPLETE")
print("=" * 80)

print("Complete-follow-up patients:", gee_complete["stay_id"].nunique())

print("Longitudinal observations:", len(gee_complete))

print("Phenotype × time P:", interaction_p)

print("\nOutputs saved to:\n", REVISION_DIR)
complete_result = {
    "n_patients": gee_complete["stay_id"].nunique(),
    "n_observations": int(gee_complete_result.nobs),
    "wald_chi2": interaction_chi2,
    "df": interaction_df,
    "phenotype_time_interaction_p": interaction_p,
}
# ============================================================
# 16H — SENSITIVITY ANALYSIS:
# PARSIMONIOUSLY ADJUSTED POST-MB GEE
#
# PURPOSE
#
# Evaluate whether phenotype-specific post-MB NEE trajectories
# remain different after adjustment for:
#
#   1. Baseline NEE
#   2. Pre-MB lactate
#   3. Time from ICU admission to methylene blue
#
# IMPORTANT
#
# - Baseline (-1 to 0 h) is NOT included as a longitudinal
#   outcome in this model.
#
# - Post-MB outcome timepoints:
#       0 to <1 h
#       1 to <2 h
#       2 to <3 h
#       3 to <4 h
#       4 to <5 h
#       5 to <6 h
#
# - Baseline NEE enters as an adjustment covariate.
#
# - Phenotype definition is unchanged.
#
# - Uses corrected_data, where intervals entirely after death
#   have already been removed.
#
# - Primary inferential test remains the GLOBAL
#   phenotype × time interaction.
# ============================================================

import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

# ------------------------------------------------------------
# 1. OUTPUT DIRECTORY
# ------------------------------------------------------------

REVISION_DIR = PROJECT_DIR / "revision_analyses"

REVISION_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("16H — ADJUSTED POST-MB GEE")
print("=" * 80)

# ------------------------------------------------------------
# 2. SELECT BASELINE COVARIATES
# ------------------------------------------------------------

baseline_covariates = (
    baseline[
        [
            "subject_id",
            "stay_id",
            "hadm_id",
            "hemodynamic_phenotype",
            "nee_baseline_median",
            "lactate_pre_mb",
            "hours_icu_to_mb",
        ]
    ]
    .drop_duplicates(
        subset=[
            "subject_id",
            "stay_id",
        ]
    )
    .copy()
)

# Convert to numeric
for column in [
    "nee_baseline_median",
    "lactate_pre_mb",
    "hours_icu_to_mb",
]:

    baseline_covariates[column] = pd.to_numeric(
        baseline_covariates[column],
        errors="coerce",
    )

# ------------------------------------------------------------
# 3. CREATE TRANSFORMED BASELINE NEE
# ------------------------------------------------------------

baseline_covariates["log1p_nee_baseline"] = np.log1p(
    baseline_covariates["nee_baseline_median"]
)

# ------------------------------------------------------------
# 4. COVARIATE AVAILABILITY AUDIT
# ------------------------------------------------------------

all_three_available = (
    baseline_covariates[
        [
            "nee_baseline_median",
            "lactate_pre_mb",
            "hours_icu_to_mb",
        ]
    ]
    .notna()
    .all(axis=1)
)

covariate_audit = pd.DataFrame(
    {
        "variable": [
            "Baseline NEE",
            "Pre-MB lactate",
            "ICU admission to MB",
            "All three covariates",
        ],
        "n_available": [
            int(baseline_covariates["nee_baseline_median"].notna().sum()),
            int(baseline_covariates["lactate_pre_mb"].notna().sum()),
            int(baseline_covariates["hours_icu_to_mb"].notna().sum()),
            int(all_three_available.sum()),
        ],
    }
)

covariate_audit["percent_available"] = (
    100 * covariate_audit["n_available"] / len(baseline_covariates)
)

print("\n" + "=" * 80)
print("BASELINE COVARIATE AVAILABILITY")
print("=" * 80)

display(covariate_audit)

# ------------------------------------------------------------
# 5. COVARIATE DISTRIBUTIONS
# ------------------------------------------------------------

covariate_descriptive_rows = []

for variable in [
    "nee_baseline_median",
    "lactate_pre_mb",
    "hours_icu_to_mb",
]:

    values = baseline_covariates[variable].dropna()

    covariate_descriptive_rows.append(
        {
            "variable": variable,
            "n": int(len(values)),
            "median": float(values.median()) if not values.empty else np.nan,
            "q25": float(values.quantile(0.25)) if not values.empty else np.nan,
            "q75": float(values.quantile(0.75)) if not values.empty else np.nan,
            "min": float(values.min()) if not values.empty else np.nan,
            "max": float(values.max()) if not values.empty else np.nan,
        }
    )

covariate_descriptives = pd.DataFrame(covariate_descriptive_rows)

print("\n" + "=" * 80)
print("COVARIATE DISTRIBUTIONS")
print("=" * 80)

display(covariate_descriptives)

# ------------------------------------------------------------
# 6. ICU-TO-MB TIME AUDIT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TIME ICU → MB AUDIT")
print("=" * 80)

display(
    baseline_covariates["hours_icu_to_mb"].describe(
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
)

# ------------------------------------------------------------
# 7. RESTRICT LONGITUDINAL OUTCOME TO POST-MB
# ------------------------------------------------------------

# hour_index:
# -1 = Baseline (-1 to 0 h)
#  0 = 0 to <1 h
#  1 = 1 to <2 h
#  ...
#  5 = 5 to <6 h
#
# Baseline is removed because baseline NEE is now an
# adjustment covariate.

post_mb = corrected_data.loc[
    corrected_data["hour_index"].isin(
        [
            0,
            1,
            2,
            3,
            4,
            5,
        ]
    )
].copy()

print("\n" + "=" * 80)
print("POST-MB LONGITUDINAL DATA BEFORE COVARIATE MERGE")
print("=" * 80)

print("Patients:", post_mb["stay_id"].nunique())

print("Observations:", len(post_mb))

# ------------------------------------------------------------
# 8. MERGE BASELINE COVARIATES
# ------------------------------------------------------------

baseline_for_merge = baseline_covariates[
    [
        "subject_id",
        "stay_id",
        "nee_baseline_median",
        "log1p_nee_baseline",
        "lactate_pre_mb",
        "hours_icu_to_mb",
    ]
].copy()

adjusted_data = post_mb.merge(
    baseline_for_merge,
    on=[
        "subject_id",
        "stay_id",
    ],
    how="left",
    validate="many_to_one",
)

print("\nAfter baseline covariate merge:")

print("Patients:", adjusted_data["stay_id"].nunique())

print("Observations:", len(adjusted_data))

# ------------------------------------------------------------
# 9. COMPLETE-CASE SAMPLE
# ------------------------------------------------------------

required_model_variables = [
    "log1p_nee",
    "hemodynamic_phenotype",
    "hour_index",
    "log1p_nee_baseline",
    "lactate_pre_mb",
    "hours_icu_to_mb",
]

adjusted_complete = adjusted_data.dropna(subset=required_model_variables).copy()

print("\n" + "=" * 80)
print("ADJUSTED MODEL COMPLETE-CASE SAMPLE")
print("=" * 80)

print("Patients:", adjusted_complete["stay_id"].nunique())

print("Observations:", len(adjusted_complete))

# ------------------------------------------------------------
# 10. PATIENTS EXCLUDED DUE TO MISSING COVARIATES
# ------------------------------------------------------------

all_post_mb_stays = set(post_mb["stay_id"].unique())

adjusted_stays = set(adjusted_complete["stay_id"].unique())

excluded_adjusted_ids = all_post_mb_stays - adjusted_stays

excluded_adjusted = baseline_covariates.loc[
    baseline_covariates["stay_id"].isin(excluded_adjusted_ids),
    [
        "subject_id",
        "stay_id",
        "hemodynamic_phenotype",
        "nee_baseline_median",
        "lactate_pre_mb",
        "hours_icu_to_mb",
    ],
].copy()

print(
    "\nPatients excluded because at least one " "adjustment covariate was unavailable:"
)

display(excluded_adjusted)

# ------------------------------------------------------------
# 11. PHENOTYPE DISTRIBUTION
# ------------------------------------------------------------

adjusted_patient_level = adjusted_complete[
    [
        "subject_id",
        "stay_id",
        "hemodynamic_phenotype",
    ]
].drop_duplicates()

adjusted_phenotype_counts = (
    adjusted_patient_level["hemodynamic_phenotype"]
    .value_counts()
    .rename_axis("phenotype")
    .reset_index(name="n")
)

adjusted_phenotype_counts["percent"] = (
    100 * adjusted_phenotype_counts["n"] / len(adjusted_patient_level)
)

print("\n" + "=" * 80)
print("PHENOTYPE DISTRIBUTION IN ADJUSTED MODEL")
print("=" * 80)

display(adjusted_phenotype_counts)

# ------------------------------------------------------------
# 12. PATIENTS BY POST-MB TIMEPOINT
# ------------------------------------------------------------

adjusted_by_time = (
    adjusted_complete.groupby(
        [
            "hour_index",
            "interval_label",
        ],
        observed=True,
    )
    .agg(
        n_patients=(
            "stay_id",
            "nunique",
        ),
        n_observations=(
            "stay_id",
            "size",
        ),
        median_nee=(
            "nee_median",
            "median",
        ),
    )
    .reset_index()
)

print("\n" + "=" * 80)
print("PATIENTS BY POST-MB TIMEPOINT")
print("=" * 80)

display(adjusted_by_time)

# ------------------------------------------------------------
# 13. PREPARE PHENOTYPE AND TIME FACTORS
# ------------------------------------------------------------

phenotype_categories = [
    "Preserved diastolic pressure",
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
]

adjusted_complete["phenotype_model"] = pd.Categorical(
    adjusted_complete["hemodynamic_phenotype"],
    categories=phenotype_categories,
    ordered=False,
)

adjusted_complete["time_factor_post"] = pd.Categorical(
    adjusted_complete["hour_index"],
    categories=[
        0,
        1,
        2,
        3,
        4,
        5,
    ],
    ordered=False,
)

# ============================================================
# 16H.1 — FINAL CORRECTION:
# log-transform ICU-to-MB time and refit adjusted GEE
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ------------------------------------------------------------
# 1. LOG-TRANSFORM ICU → MB TIME
# ------------------------------------------------------------

adjusted_complete["log1p_hours_icu_to_mb"] = np.log1p(
    adjusted_complete["hours_icu_to_mb"]
)

# ------------------------------------------------------------
# 2. STANDARDIZE THE THREE ADJUSTMENT VARIABLES
# ------------------------------------------------------------

for variable in [
    "log1p_nee_baseline",
    "lactate_pre_mb",
    "log1p_hours_icu_to_mb",
]:

    mean_value = adjusted_complete[variable].mean()

    sd_value = adjusted_complete[variable].std()

    adjusted_complete[f"{variable}_z"] = (
        adjusted_complete[variable] - mean_value
    ) / sd_value

# ------------------------------------------------------------
# 3. FINAL ADJUSTED MODEL
# ------------------------------------------------------------

formula_adjusted_final = (
    "log1p_nee ~ "
    "C(phenotype_model, "
    "Treatment(reference='Preserved diastolic pressure')) "
    "* C(time_factor_post, Treatment(reference=0)) "
    "+ log1p_nee_baseline_z "
    "+ lactate_pre_mb_z "
    "+ log1p_hours_icu_to_mb_z"
)

gee_adjusted_final_model = smf.gee(
    formula=formula_adjusted_final,
    groups="subject_id",
    data=adjusted_complete,
    family=sm.families.Gaussian(sm.families.links.Identity()),
    cov_struct=sm.cov_struct.Exchangeable(),
)

gee_adjusted_final_result = gee_adjusted_final_model.fit()

print("=" * 80)
print("FINAL ADJUSTED POST-MB GEE")
print("=" * 80)

print(gee_adjusted_final_result.summary())

# ------------------------------------------------------------
# 4. GLOBAL PHENOTYPE × TIME WALD TEST
# ------------------------------------------------------------

interaction_terms_final = [
    term
    for term in gee_adjusted_final_result.params.index
    if (":" in term and "phenotype_model" in term and "time_factor_post" in term)
]

parameter_names_final = list(gee_adjusted_final_result.params.index)

interaction_indices_final = [
    parameter_names_final.index(term) for term in interaction_terms_final
]

R_final = np.zeros(
    (
        len(interaction_indices_final),
        len(parameter_names_final),
    )
)

for row_index, parameter_index in enumerate(interaction_indices_final):

    R_final[row_index, parameter_index] = 1

wald_final = gee_adjusted_final_result.wald_test(
    R_final,
    scalar=True,
)

chi2_final = float(np.asarray(wald_final.statistic).squeeze())

df_final = len(interaction_terms_final)

p_final = float(np.asarray(wald_final.pvalue).squeeze())

print("\n" + "=" * 80)
print("FINAL GLOBAL ADJUSTED PHENOTYPE × TIME INTERACTION")
print("=" * 80)

print(f"Wald χ²({df_final}) = {chi2_final:.3f}")

print(f"P = {p_final:.6f}")

# ------------------------------------------------------------
# 5. ADJUSTMENT COVARIATES
# ------------------------------------------------------------

final_covariates = [
    "log1p_nee_baseline_z",
    "lactate_pre_mb_z",
    "log1p_hours_icu_to_mb_z",
]

ci_final = gee_adjusted_final_result.conf_int()

final_covariate_results = []

for term in final_covariates:

    final_covariate_results.append(
        {
            "covariate": term,
            "estimate": gee_adjusted_final_result.params[term],
            "standard_error": gee_adjusted_final_result.bse[term],
            "ci_lower": ci_final.loc[term, 0],
            "ci_upper": ci_final.loc[term, 1],
            "p_value": gee_adjusted_final_result.pvalues[term],
        }
    )

final_covariate_results = pd.DataFrame(final_covariate_results)

print("\n" + "=" * 80)
print("FINAL ADJUSTMENT COVARIATE EFFECTS")
print("=" * 80)

display(final_covariate_results)

# ------------------------------------------------------------
# 6. FINAL ROBUSTNESS TABLE
# ------------------------------------------------------------

robustness_final_corrected = pd.DataFrame(
    {
        "analysis": [
            "Primary GEE",
            "Invasive-only GEE",
            "Complete 6-h follow-up GEE",
            "Adjusted post-MB GEE",
        ],
        "n_patients": [
            corrected_data["subject_id"].nunique(),
            invasive_result["n_patients"],
            complete_result["n_patients"],
            adjusted_complete["subject_id"].nunique(),
        ],
        "n_observations": [
            len(corrected_data),
            invasive_result["n_observations"],
            complete_result["n_observations"],
            int(gee_adjusted_final_result.nobs),
        ],
        "wald_chi2": [
            np.nan,
            invasive_result["wald_chi2"],
            complete_result["wald_chi2"],
            chi2_final,
        ],
        "df": [
            int(primary["df"]),
            invasive_result["df"],
            complete_result["df"],
            df_final,
        ],
        "phenotype_time_interaction_p": [
            primary["p_value"],
            invasive_result["phenotype_time_interaction_p"],
            complete_result["phenotype_time_interaction_p"],
            p_final,
        ],
        "adjustment": [
            "None",
            "None",
            "None",
            ("Baseline NEE, pre-MB lactate, " "log-transformed ICU-to-MB time"),
        ],
    }
)

print("\n" + "=" * 80)
print("FINAL CORRECTED ROBUSTNESS ANALYSES")
print("=" * 80)

display(robustness_final_corrected)

# ------------------------------------------------------------
# 7. SAVE
# ------------------------------------------------------------

robustness_final_corrected.to_csv(
    REVISION_DIR / "robustness_analyses_FINAL_CORRECTED.csv",
    index=False,
)

final_covariate_results.to_csv(
    REVISION_DIR / "sensitivity_adjusted_GEE_covariate_effects_FINAL.csv",
    index=False,
)

with open(
    REVISION_DIR / "sensitivity_adjusted_post_MB_GEE_FINAL_summary.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(gee_adjusted_final_result.summary().as_text())

    f.write("\n\nGLOBAL PHENOTYPE × TIME INTERACTION\n")

    f.write(f"Wald chi-square({df_final}) = " f"{chi2_final:.6f}\n")

    f.write(f"P = {p_final:.8f}\n")

    f.write("\n\nMODEL FORMULA\n")

    f.write(formula_adjusted_final)

adjusted_complete.to_csv(
    REVISION_DIR / "sensitivity_adjusted_post_MB_GEE_model_data.csv.gz",
    index=False,
    compression="gzip",
)
print("\nSaved final corrected outputs.")
