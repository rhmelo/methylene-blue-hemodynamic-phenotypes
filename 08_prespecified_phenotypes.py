"""Source: original notebook cell 15. Run through run.py."""

from common import display, phase_dir

# ============================================================
# HiRID Methylene Blue Hemodynamic Phenotypes
# STEP 07 — Prespecified physiological phenotypes
#
# PRIMARY PHENOTYPE DEFINITION
#
# Phenotype 1 — Predominant vasoplegic phenotype:
#   DBP < 50 mmHg AND pulse pressure >= 40 mmHg
#
# Phenotype 2 — Mixed / low-pulsatility shock:
#   DBP < 50 mmHg AND pulse pressure < 40 mmHg
#
# Phenotype 3 — Preserved diastolic pressure:
#   DBP >= 50 mmHg
#
# Heart rate, norepinephrine, lactate and outcomes are NOT
# used to define the phenotypes.
#
# They are used only for:
# - physiological characterization
# - longitudinal validation
# - comparison with DSI, SI and MSI
# - future external validation in MIMIC-IV
# ============================================================

import pandas as pd
import numpy as np
import json
import warnings

from scipy.stats import (
    kruskal,
    chi2_contingency,
)

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_DIR = phase_dir("hirid")
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
DERIVED_DIR = PROJECT_DIR / "derived_data"
MODEL_DIR = PROJECT_DIR / "models"

for folder in [
    PROJECT_DIR,
    OUTPUT_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    LOG_DIR,
    DERIVED_DIR,
    MODEL_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

ANALYSIS_PATH = (
    DERIVED_DIR
    / "step06_analysis_dataset_with_three_macrohemodynamic_phenotypes.parquet"
)

if not ANALYSIS_PATH.exists():

    ANALYSIS_PATH = (
        DERIVED_DIR / "step05_analysis_dataset_with_macrohemodynamic_phenotypes.parquet"
    )

if not ANALYSIS_PATH.exists():

    ANALYSIS_PATH = (
        DERIVED_DIR / "step04_analysis_dataset_weight_normalized_norepinephrine.parquet"
    )

if not ANALYSIS_PATH.exists():
    raise FileNotFoundError(f"Analysis dataset not found:\n{ANALYSIS_PATH}")

# ------------------------------------------------------------
# 2. LOAD DATA
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
# 3. CHECK REQUIRED VARIABLES
# ------------------------------------------------------------

REQUIRED_VARIABLES = [
    "dbp_baseline",
    "pulse_pressure_baseline",
    "heart_rate_baseline",
]

missing_variables = [
    variable for variable in REQUIRED_VARIABLES if variable not in analysis.columns
]

if missing_variables:
    raise RuntimeError(f"Missing required variables: {missing_variables}")

# ------------------------------------------------------------
# 4. CALCULATE COMPARISON INDICES IF NECESSARY
# ------------------------------------------------------------

if "diastolic_shock_index" not in analysis.columns:

    analysis["diastolic_shock_index"] = (
        analysis["heart_rate_baseline"] / analysis["dbp_baseline"]
    )

if "shock_index" not in analysis.columns:

    analysis["shock_index"] = analysis["heart_rate_baseline"] / analysis["sbp_baseline"]

if "modified_shock_index" not in analysis.columns:

    analysis["modified_shock_index"] = (
        analysis["heart_rate_baseline"] / analysis["map_baseline"]
    )

analysis.loc[
    ~np.isfinite(analysis["diastolic_shock_index"]),
    "diastolic_shock_index",
] = np.nan

analysis.loc[
    ~np.isfinite(analysis["shock_index"]),
    "shock_index",
] = np.nan

analysis.loc[
    ~np.isfinite(analysis["modified_shock_index"]),
    "modified_shock_index",
] = np.nan

# ------------------------------------------------------------
# 5. PRIMARY PRESPECIFIED PHENOTYPE
# ------------------------------------------------------------

PRIMARY_DBP_CUTOFF = 50.0
PRIMARY_PP_CUTOFF = 40.0


def classify_physiological_phenotype(
    dbp,
    pulse_pressure,
    dbp_cutoff=50.0,
    pp_cutoff=40.0,
):

    if pd.isna(dbp) or pd.isna(pulse_pressure):
        return np.nan

    if dbp < dbp_cutoff:

        if pulse_pressure >= pp_cutoff:
            return 1

        return 2

    return 3


analysis["physiological_phenotype"] = analysis.apply(
    lambda row: classify_physiological_phenotype(
        dbp=row["dbp_baseline"],
        pulse_pressure=row["pulse_pressure_baseline"],
        dbp_cutoff=PRIMARY_DBP_CUTOFF,
        pp_cutoff=PRIMARY_PP_CUTOFF,
    ),
    axis=1,
)

PHENOTYPE_LABELS = {
    1: "Predominant vasoplegic",
    2: "Mixed / low pulsatility",
    3: "Preserved diastolic pressure",
}

analysis["physiological_phenotype_label"] = analysis["physiological_phenotype"].map(
    PHENOTYPE_LABELS
)

analysis["candidate_vasoplegic_phenotype"] = analysis["physiological_phenotype"].eq(1)

# ------------------------------------------------------------
# 6. PRIMARY PHENOTYPE COUNTS
# ------------------------------------------------------------

phenotype_counts = (
    analysis.dropna(subset=["physiological_phenotype"])
    .groupby(
        [
            "physiological_phenotype",
            "physiological_phenotype_label",
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

print("\n" + "=" * 80)
print("PRIMARY PHYSIOLOGICAL PHENOTYPE COUNTS")
print("=" * 80)

display(phenotype_counts)

# ------------------------------------------------------------
# 7. BASELINE CHARACTERIZATION
# ------------------------------------------------------------

BASELINE_CHARACTERIZATION_VARIABLES = [
    # Defining variables
    "dbp_baseline",
    "pulse_pressure_baseline",
    # Not used for defining phenotypes
    "heart_rate_baseline",
    "sbp_baseline",
    "map_baseline",
    "lactate_baseline",
    "norepi_ug_min_baseline_winsor",
    "norepi_ug_kg_min_baseline_winsor",
    "diastolic_shock_index",
    "shock_index",
    "modified_shock_index",
    "weight_kg",
]

available_baseline_variables = [
    variable
    for variable in BASELINE_CHARACTERIZATION_VARIABLES
    if variable in analysis.columns
]

baseline_rows = []

for variable in available_baseline_variables:

    for phenotype, group in analysis.dropna(subset=["physiological_phenotype"]).groupby(
        "physiological_phenotype"
    ):

        values = pd.to_numeric(
            group[variable],
            errors="coerce",
        ).dropna()

        baseline_rows.append(
            {
                "variable": variable,
                "phenotype": int(phenotype),
                "phenotype_label": PHENOTYPE_LABELS[int(phenotype)],
                "n": int(values.size),
                "mean": (float(values.mean()) if not values.empty else np.nan),
                "sd": (float(values.std()) if len(values) > 1 else np.nan),
                "median": (float(values.median()) if not values.empty else np.nan),
                "q25": (float(values.quantile(0.25)) if not values.empty else np.nan),
                "q75": (float(values.quantile(0.75)) if not values.empty else np.nan),
            }
        )

baseline_characterization = pd.DataFrame(baseline_rows)

print("\n" + "=" * 80)
print("BASELINE PHYSIOLOGICAL CHARACTERIZATION")
print("=" * 80)

display(
    baseline_characterization[
        [
            "variable",
            "phenotype",
            "phenotype_label",
            "n",
            "median",
            "q25",
            "q75",
        ]
    ]
)

# ------------------------------------------------------------
# 8. LONGITUDINAL VALIDATION VARIABLES
# ------------------------------------------------------------

LONGITUDINAL_VARIABLES = [
    # Persistent arterial-pressure profile
    "dbp_median_h1_h6_weighted_validation",
    "map_median_h1_h6_weighted_validation",
    "pulse_pressure_median_h1_h6_validation",
    "proportion_hours_dbp_lt_50_h1_h6_validation",
    "proportion_hours_dbp_lt_45_h1_h6_validation",
    "proportion_hours_map_lt_65_h1_h6_validation",
    # Vasopressor burden
    "norepi_ug_kg_min_median_h1_h6",
    "norepi_ug_kg_min_max_h1_h6",
    "norepi_ug_kg_min_auc_h1_h6",
    "norepi_ug_kg_min_slope_h1_h6",
    # Existing longitudinal variables
    "delta_dbp_followup_minus_baseline",
    "delta_map_followup_minus_baseline",
    "delta_pp_followup_minus_baseline",
    "norepi_present_last_observed_hour",
]

available_longitudinal_variables = [
    variable for variable in LONGITUDINAL_VARIABLES if variable in analysis.columns
]

longitudinal_rows = []

for variable in available_longitudinal_variables:

    for phenotype, group in analysis.dropna(subset=["physiological_phenotype"]).groupby(
        "physiological_phenotype"
    ):

        values = pd.to_numeric(
            group[variable],
            errors="coerce",
        ).dropna()

        longitudinal_rows.append(
            {
                "variable": variable,
                "phenotype": int(phenotype),
                "phenotype_label": PHENOTYPE_LABELS[int(phenotype)],
                "n": int(values.size),
                "mean": (float(values.mean()) if not values.empty else np.nan),
                "sd": (float(values.std()) if len(values) > 1 else np.nan),
                "median": (float(values.median()) if not values.empty else np.nan),
                "q25": (float(values.quantile(0.25)) if not values.empty else np.nan),
                "q75": (float(values.quantile(0.75)) if not values.empty else np.nan),
            }
        )

longitudinal_validation = pd.DataFrame(longitudinal_rows)

print("\n" + "=" * 80)
print("LONGITUDINAL PHYSIOLOGICAL VALIDATION")
print("=" * 80)

display(
    longitudinal_validation[
        [
            "variable",
            "phenotype",
            "phenotype_label",
            "n",
            "median",
            "q25",
            "q75",
        ]
    ]
)

# ------------------------------------------------------------
# 9. GLOBAL CONTINUOUS TESTS
# ------------------------------------------------------------

CONTINUOUS_TEST_VARIABLES = list(
    dict.fromkeys(available_baseline_variables + available_longitudinal_variables)
)

global_test_rows = []

for variable in CONTINUOUS_TEST_VARIABLES:

    groups = []

    for phenotype in [1, 2, 3]:

        values = pd.to_numeric(
            analysis.loc[
                analysis["physiological_phenotype"].eq(phenotype),
                variable,
            ],
            errors="coerce",
        ).dropna()

        if not values.empty:
            groups.append(values)

    if len(groups) == 3:

        statistic, p_value = kruskal(*groups)

        global_test_rows.append(
            {
                "variable": variable,
                "test": "Kruskal-Wallis",
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

global_continuous_tests = pd.DataFrame(global_test_rows)

# ------------------------------------------------------------
# 10. BINARY VARIABLES
# ------------------------------------------------------------

BINARY_VARIABLES = [
    "probable_septic_shock_strict",
    "lactate_gt_2",
    "norepi_present_last_observed_hour",
]

available_binary_variables = [
    variable for variable in BINARY_VARIABLES if variable in analysis.columns
]

binary_rows = []
binary_test_rows = []

for variable in available_binary_variables:

    temporary = analysis.loc[
        analysis["physiological_phenotype"].notna() & analysis[variable].notna(),
        [
            "physiological_phenotype",
            variable,
        ],
    ].copy()

    for phenotype, group in temporary.groupby("physiological_phenotype"):

        events = group[variable].astype(bool).sum()

        binary_rows.append(
            {
                "variable": variable,
                "phenotype": int(phenotype),
                "phenotype_label": PHENOTYPE_LABELS[int(phenotype)],
                "n": int(len(group)),
                "events": int(events),
                "percent": float(100 * events / len(group)),
            }
        )

    contingency = pd.crosstab(
        temporary["physiological_phenotype"],
        temporary[variable],
    )

    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:

        chi2, p_value, _, _ = chi2_contingency(contingency)

        binary_test_rows.append(
            {
                "variable": variable,
                "test": "Chi-square",
                "statistic": float(chi2),
                "p_value": float(p_value),
            }
        )

binary_characterization = pd.DataFrame(binary_rows)

binary_tests = pd.DataFrame(binary_test_rows)

# ------------------------------------------------------------
# 11. COMPARE WITH 3-SPHERICAL GMM
# ------------------------------------------------------------

comparison_with_gmm_available = "phenotype_3class" in analysis.columns

if comparison_with_gmm_available:

    comparison_data = analysis.loc[
        analysis["physiological_phenotype"].notna()
        & analysis["phenotype_3class"].notna(),
        [
            "patientid",
            "physiological_phenotype",
            "phenotype_3class",
        ],
    ].copy()

    physiological_labels = comparison_data["physiological_phenotype"].astype(int)

    gmm_labels = comparison_data["phenotype_3class"].astype(int)

    agreement_ari = adjusted_rand_score(
        physiological_labels,
        gmm_labels,
    )

    agreement_nmi = normalized_mutual_info_score(
        physiological_labels,
        gmm_labels,
    )

    gmm_cross_table = pd.crosstab(
        comparison_data["physiological_phenotype"],
        comparison_data["phenotype_3class"],
        margins=True,
    )

    agreement_summary = pd.DataFrame(
        {
            "metric": [
                "Patients compared",
                "Adjusted Rand index",
                "Normalized mutual information",
            ],
            "value": [
                len(comparison_data),
                agreement_ari,
                agreement_nmi,
            ],
        }
    )

    print("\n" + "=" * 80)
    print("AGREEMENT WITH 3-SPHERICAL GMM")
    print("=" * 80)

    display(agreement_summary)

    print("\nCross-tabulation:")

    display(gmm_cross_table)

else:

    comparison_data = pd.DataFrame()

    agreement_summary = pd.DataFrame(
        {
            "metric": ["GMM comparison available"],
            "value": [False],
        }
    )

    gmm_cross_table = pd.DataFrame()

    print("\nNo phenotype_3class column was found. " "GMM comparison was skipped.")

# ------------------------------------------------------------
# 12. SENSITIVITY DEFINITIONS
# ------------------------------------------------------------

SENSITIVITY_DEFINITIONS = [
    {
        "definition_name": "Primary_DBP50_PP40",
        "dbp_cutoff": 50.0,
        "pp_cutoff": 40.0,
    },
    {
        "definition_name": "DBP50_PP50",
        "dbp_cutoff": 50.0,
        "pp_cutoff": 50.0,
    },
    {
        "definition_name": "DBP45_PP40",
        "dbp_cutoff": 45.0,
        "pp_cutoff": 40.0,
    },
    {
        "definition_name": "DBP40_PP40",
        "dbp_cutoff": 40.0,
        "pp_cutoff": 40.0,
    },
    {
        "definition_name": "DBP50_PP30",
        "dbp_cutoff": 50.0,
        "pp_cutoff": 30.0,
    },
]

sensitivity_count_rows = []

for definition in SENSITIVITY_DEFINITIONS:

    column_name = "phenotype_" + definition["definition_name"]

    analysis[column_name] = analysis.apply(
        lambda row: classify_physiological_phenotype(
            dbp=row["dbp_baseline"],
            pulse_pressure=row["pulse_pressure_baseline"],
            dbp_cutoff=definition["dbp_cutoff"],
            pp_cutoff=definition["pp_cutoff"],
        ),
        axis=1,
    )

    counts = analysis[column_name].value_counts(dropna=False)

    for phenotype in [1, 2, 3]:

        n = int(counts.get(phenotype, 0))

        sensitivity_count_rows.append(
            {
                "definition_name": definition["definition_name"],
                "dbp_cutoff": definition["dbp_cutoff"],
                "pp_cutoff": definition["pp_cutoff"],
                "phenotype": phenotype,
                "phenotype_label": PHENOTYPE_LABELS[phenotype],
                "n": n,
                "percent": float(100 * n / analysis[column_name].notna().sum()),
            }
        )

sensitivity_counts = pd.DataFrame(sensitivity_count_rows)

print("\n" + "=" * 80)
print("SENSITIVITY DEFINITION COUNTS")
print("=" * 80)

display(sensitivity_counts)

# ------------------------------------------------------------
# 13. SENSITIVITY PHYSIOLOGICAL VALIDATION
# ------------------------------------------------------------

SENSITIVITY_VALIDATION_VARIABLES = [
    "norepi_ug_kg_min_baseline_winsor",
    "norepi_ug_kg_min_median_h1_h6",
    "norepi_ug_kg_min_auc_h1_h6",
    "dbp_median_h1_h6_weighted_validation",
    "pulse_pressure_median_h1_h6_validation",
    "proportion_hours_dbp_lt_50_h1_h6_validation",
    "proportion_hours_map_lt_65_h1_h6_validation",
    "lactate_baseline",
]

available_sensitivity_validation_variables = [
    variable
    for variable in SENSITIVITY_VALIDATION_VARIABLES
    if variable in analysis.columns
]

sensitivity_validation_rows = []

for definition in SENSITIVITY_DEFINITIONS:

    column_name = "phenotype_" + definition["definition_name"]

    for variable in available_sensitivity_validation_variables:

        for phenotype in [1, 2, 3]:

            values = pd.to_numeric(
                analysis.loc[
                    analysis[column_name].eq(phenotype),
                    variable,
                ],
                errors="coerce",
            ).dropna()

            sensitivity_validation_rows.append(
                {
                    "definition_name": definition["definition_name"],
                    "dbp_cutoff": definition["dbp_cutoff"],
                    "pp_cutoff": definition["pp_cutoff"],
                    "variable": variable,
                    "phenotype": phenotype,
                    "phenotype_label": PHENOTYPE_LABELS[phenotype],
                    "n": int(values.size),
                    "median": (float(values.median()) if not values.empty else np.nan),
                    "q25": (
                        float(values.quantile(0.25)) if not values.empty else np.nan
                    ),
                    "q75": (
                        float(values.quantile(0.75)) if not values.empty else np.nan
                    ),
                }
            )

sensitivity_validation = pd.DataFrame(sensitivity_validation_rows)

# ------------------------------------------------------------
# 14. PRIMARY SCATTERPLOT
# ------------------------------------------------------------

plot_data = analysis.loc[analysis["physiological_phenotype"].notna()].copy()

plt.figure(figsize=(10, 7))

for phenotype in [1, 2, 3]:

    subset = plot_data.loc[plot_data["physiological_phenotype"].eq(phenotype)]

    plt.scatter(
        subset["dbp_baseline"],
        subset["pulse_pressure_baseline"],
        s=25,
        alpha=0.5,
        label=(f"{phenotype}: " f"{PHENOTYPE_LABELS[phenotype]}"),
    )

plt.axvline(
    PRIMARY_DBP_CUTOFF,
    linestyle="--",
    linewidth=1.5,
)

plt.axhline(
    PRIMARY_PP_CUTOFF,
    linestyle="--",
    linewidth=1.5,
)

plt.xlabel("Baseline diastolic arterial pressure (mmHg)")

plt.ylabel("Baseline pulse pressure (mmHg)")

plt.title("Prespecified physiological phenotypes")

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "step07_prespecified_physiological_phenotypes.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# ------------------------------------------------------------
# 15. PHENOTYPE PROFILE FIGURE
# ------------------------------------------------------------

profile_variables = [
    "dbp_baseline",
    "pulse_pressure_baseline",
    "heart_rate_baseline",
    "norepi_ug_kg_min_baseline_winsor",
    "lactate_baseline",
]

available_profile_variables = [
    variable for variable in profile_variables if variable in analysis.columns
]

profile_for_plot = (
    analysis.dropna(subset=["physiological_phenotype"])
    .groupby("physiological_phenotype")[available_profile_variables]
    .median()
)

profile_standardized = (
    profile_for_plot - analysis[available_profile_variables].median()
) / analysis[available_profile_variables].std()

plt.figure(figsize=(11, 6))

for phenotype in profile_standardized.index:

    plt.plot(
        available_profile_variables,
        profile_standardized.loc[phenotype],
        marker="o",
        label=(f"{int(phenotype)}: " f"{PHENOTYPE_LABELS[int(phenotype)]}"),
    )

plt.axhline(
    0,
    linewidth=1,
)

plt.ylabel("Standardized median")

plt.xlabel("Variable")

plt.title("Physiological profiles of prespecified phenotypes")

plt.xticks(
    rotation=30,
    ha="right",
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "step07_prespecified_phenotype_profiles.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# ------------------------------------------------------------
# 16. SAVE TRANSPORTABLE DEFINITION
# ------------------------------------------------------------

transportable_definition = {
    "primary_definition": {
        "dbp_cutoff_mmHg": PRIMARY_DBP_CUTOFF,
        "pulse_pressure_cutoff_mmHg": PRIMARY_PP_CUTOFF,
        "phenotype_1": {
            "label": "Predominant vasoplegic",
            "definition": "DBP <50 mmHg and pulse pressure >=40 mmHg",
        },
        "phenotype_2": {
            "label": "Mixed / low pulsatility",
            "definition": "DBP <50 mmHg and pulse pressure <40 mmHg",
        },
        "phenotype_3": {
            "label": "Preserved diastolic pressure",
            "definition": "DBP >=50 mmHg",
        },
    },
    "baseline_window": "Median invasive hemodynamic values from 0 to 1 hour after time zero",
    "phenotype_variables": [
        "invasive diastolic arterial pressure",
        "invasive pulse pressure",
    ],
    "heart_rate_used_in_definition": False,
    "norepinephrine_used_in_definition": False,
    "lactate_used_in_definition": False,
    "sensitivity_definitions": SENSITIVITY_DEFINITIONS,
    "intended_external_validation": "Apply unchanged to MIMIC-IV patients before methylene blue administration",
}

with open(
    MODEL_DIR / "step07_transportable_physiological_phenotype_definition.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        transportable_definition,
        f,
        indent=2,
        ensure_ascii=False,
    )

# ------------------------------------------------------------
# 17. SAVE OUTPUTS
# ------------------------------------------------------------

analysis.to_parquet(
    DERIVED_DIR
    / "step07_analysis_dataset_with_prespecified_physiological_phenotypes.parquet",
    index=False,
)

analysis.to_csv(
    TABLE_DIR
    / "step07_analysis_dataset_with_prespecified_physiological_phenotypes.csv",
    index=False,
)

phenotype_counts.to_csv(
    TABLE_DIR / "step07_primary_phenotype_counts.csv",
    index=False,
)

baseline_characterization.to_csv(
    TABLE_DIR / "step07_baseline_phenotype_characterization.csv",
    index=False,
)

longitudinal_validation.to_csv(
    TABLE_DIR / "step07_longitudinal_physiological_validation.csv",
    index=False,
)

global_continuous_tests.to_csv(
    TABLE_DIR / "step07_global_continuous_tests.csv",
    index=False,
)

binary_characterization.to_csv(
    TABLE_DIR / "step07_binary_characterization.csv",
    index=False,
)

binary_tests.to_csv(
    TABLE_DIR / "step07_binary_tests.csv",
    index=False,
)

agreement_summary.to_csv(
    TABLE_DIR / "step07_agreement_with_gmm_summary.csv",
    index=False,
)

if not gmm_cross_table.empty:

    gmm_cross_table.to_csv(TABLE_DIR / "step07_cross_tabulation_with_gmm.csv")

sensitivity_counts.to_csv(
    TABLE_DIR / "step07_sensitivity_definition_counts.csv",
    index=False,
)

sensitivity_validation.to_csv(
    TABLE_DIR / "step07_sensitivity_physiological_validation.csv",
    index=False,
)

run_summary = {
    "input_n": int(len(analysis)),
    "classified_n": int(analysis["physiological_phenotype"].notna().sum()),
    "primary_dbp_cutoff": PRIMARY_DBP_CUTOFF,
    "primary_pulse_pressure_cutoff": PRIMARY_PP_CUTOFF,
    "phenotype_1_n": int(analysis["physiological_phenotype"].eq(1).sum()),
    "phenotype_2_n": int(analysis["physiological_phenotype"].eq(2).sum()),
    "phenotype_3_n": int(analysis["physiological_phenotype"].eq(3).sum()),
    "gmm_comparison_available": bool(comparison_with_gmm_available),
    "adjusted_rand_index_vs_gmm": (
        float(agreement_ari) if comparison_with_gmm_available else None
    ),
    "normalized_mutual_information_vs_gmm": (
        float(agreement_nmi) if comparison_with_gmm_available else None
    ),
}

with open(
    LOG_DIR / "step07_prespecified_physiological_phenotype_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        run_summary,
        f,
        indent=2,
        ensure_ascii=False,
    )

# ------------------------------------------------------------
# 18. FINAL DISPLAY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 07 COMPLETE")
print("=" * 80)

print("\nPrimary definition:")

print("Phenotype 1: DBP <50 and PP >=40")

print("Phenotype 2: DBP <50 and PP <40")

print("Phenotype 3: DBP >=50")

print("\nHeart rate was NOT used to define phenotypes.")

print("Norepinephrine was NOT used to define phenotypes.")

print("Lactate was NOT used to define phenotypes.")

print(
    "\nMain dataset saved to:\n"
    f"{DERIVED_DIR / 'step07_analysis_dataset_with_prespecified_physiological_phenotypes.parquet'}"
)

print(
    "\nTransportable definition saved to:\n"
    f"{MODEL_DIR / 'step07_transportable_physiological_phenotype_definition.json'}"
)
