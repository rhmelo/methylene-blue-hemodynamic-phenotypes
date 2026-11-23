"""Source: original notebook cell 13. Run through run.py."""

from common import display, phase_dir

# ============================================================
# HiRID Methylene Blue Hemodynamic Phenotypes
# STEP 05 — Macrohemodynamic phenotype derivation
#
# PHENOTYPE VARIABLES:
# - Baseline diastolic arterial pressure
# - Baseline pulse pressure
# - Baseline heart rate
#
# NOT USED TO DEFINE PHENOTYPES:
# - Norepinephrine dose
# - Lactate
# - Longitudinal trajectories
# - Outcomes
#
# These variables are used only for:
# - Physiological characterization
# - Longitudinal validation
# - Comparison with DSI, SI and MSI
#
# Candidate vasoplegic phenotype:
# - Lower DBP
# - Preserved or higher pulse pressure
#
# Tested solutions:
# - 2, 3 and 4 Gaussian mixture components
# ============================================================

import pandas as pd
import numpy as np
import json
import warnings
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
)

from scipy.stats import (
    kruskal,
    chi2_contingency,
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

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
    folder.mkdir(parents=True, exist_ok=True)

ANALYSIS_PATH = (
    DERIVED_DIR / "step04_analysis_dataset_weight_normalized_norepinephrine.parquet"
)

HOURLY_PATH_CLEAN = (
    DERIVED_DIR / "step03b_hourly_trajectories_clean_norepinephrine.parquet"
)

HOURLY_PATH_ORIGINAL = DERIVED_DIR / "step03_hourly_hemodynamic_trajectories.parquet"

if not ANALYSIS_PATH.exists():
    raise FileNotFoundError(f"Analysis dataset not found:\n{ANALYSIS_PATH}")

if HOURLY_PATH_CLEAN.exists():
    HOURLY_PATH = HOURLY_PATH_CLEAN
elif HOURLY_PATH_ORIGINAL.exists():
    HOURLY_PATH = HOURLY_PATH_ORIGINAL
else:
    raise FileNotFoundError("Neither cleaned nor original hourly dataset was found.")

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

analysis = pd.read_parquet(ANALYSIS_PATH)

hourly = pd.read_parquet(HOURLY_PATH)

analysis["patientid"] = pd.to_numeric(
    analysis["patientid"],
    errors="coerce",
).astype(int)

hourly["patientid"] = pd.to_numeric(
    hourly["patientid"],
    errors="coerce",
).astype(int)

print("=" * 80)
print("INPUT DATA")
print("=" * 80)

print("Patient-level dataset:", len(analysis))
print("Hourly dataset rows:", len(hourly))
print("Hourly dataset patients:", hourly["patientid"].nunique())

# ------------------------------------------------------------
# 3. PHENOTYPE VARIABLES
# ------------------------------------------------------------

PHENOTYPE_VARIABLES = [
    "dbp_baseline",
    "pulse_pressure_baseline",
    "heart_rate_baseline",
]

missing_columns = [
    column for column in PHENOTYPE_VARIABLES if column not in analysis.columns
]

if missing_columns:
    raise RuntimeError(f"Missing phenotype columns: {missing_columns}")

phenotype_data = analysis.loc[analysis[PHENOTYPE_VARIABLES].notna().all(axis=1)].copy()

print(
    "\nPatients with complete macrohemodynamic variables:",
    len(phenotype_data),
)

# ------------------------------------------------------------
# 4. CONSERVATIVE PLAUSIBILITY CHECK
# ------------------------------------------------------------

phenotype_data = phenotype_data.loc[
    phenotype_data["dbp_baseline"].between(
        15,
        150,
        inclusive="both",
    )
    & phenotype_data["pulse_pressure_baseline"].between(
        5,
        200,
        inclusive="both",
    )
    & phenotype_data["heart_rate_baseline"].between(
        20,
        220,
        inclusive="both",
    )
].copy()

print(
    "Patients after plausibility checks:",
    len(phenotype_data),
)

# ------------------------------------------------------------
# 5. WINSORIZE PHENOTYPE VARIABLES
# ------------------------------------------------------------

winsor_limits = {}

for variable in PHENOTYPE_VARIABLES:

    lower = phenotype_data[variable].quantile(0.005)

    upper = phenotype_data[variable].quantile(0.995)

    winsor_limits[variable] = {
        "lower": float(lower),
        "upper": float(upper),
    }

    phenotype_data[f"{variable}_winsor"] = phenotype_data[variable].clip(
        lower=lower,
        upper=upper,
    )

PHENOTYPE_VARIABLES_WINSOR = [f"{variable}_winsor" for variable in PHENOTYPE_VARIABLES]

# ------------------------------------------------------------
# 6. STANDARDIZE VARIABLES
# ------------------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(phenotype_data[PHENOTYPE_VARIABLES_WINSOR])

# ------------------------------------------------------------
# 7. FIT GMM SOLUTIONS WITH 2–4 CLUSTERS
# ------------------------------------------------------------

candidate_models = {}
model_results = []

for n_clusters in [2, 3, 4]:

    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=50,
        max_iter=1000,
        random_state=RANDOM_STATE,
        reg_covar=1e-6,
    )

    labels = model.fit_predict(X)

    cluster_counts = pd.Series(labels).value_counts()

    minimum_cluster_n = int(cluster_counts.min())

    minimum_cluster_percent = float(100 * cluster_counts.min() / len(labels))

    silhouette = silhouette_score(
        X,
        labels,
    )

    candidate_models[n_clusters] = {
        "model": model,
        "labels": labels,
    }

    model_results.append(
        {
            "n_clusters": n_clusters,
            "bic": float(model.bic(X)),
            "aic": float(model.aic(X)),
            "silhouette": float(silhouette),
            "minimum_cluster_n": minimum_cluster_n,
            "minimum_cluster_percent": minimum_cluster_percent,
            "converged": bool(model.converged_),
            "iterations": int(model.n_iter_),
        }
    )

model_comparison = pd.DataFrame(model_results)

# Differences relative to the lowest BIC.
model_comparison["delta_bic"] = model_comparison["bic"] - model_comparison["bic"].min()

print("\n" + "=" * 80)
print("GMM MODEL COMPARISON")
print("=" * 80)

display(model_comparison.sort_values("n_clusters"))

# ------------------------------------------------------------
# 8. BOOTSTRAP STABILITY
# ------------------------------------------------------------

N_BOOTSTRAPS = 50

stability_rows = []

rng = np.random.default_rng(RANDOM_STATE)

for n_clusters in [2, 3, 4]:

    full_model = candidate_models[n_clusters]["model"]

    full_labels = candidate_models[n_clusters]["labels"]

    bootstrap_ari = []

    for bootstrap_iteration in range(N_BOOTSTRAPS):

        bootstrap_indices = rng.choice(
            len(X),
            size=len(X),
            replace=True,
        )

        X_bootstrap = X[bootstrap_indices]

        bootstrap_model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            n_init=10,
            max_iter=1000,
            random_state=(RANDOM_STATE + bootstrap_iteration + n_clusters * 100),
            reg_covar=1e-6,
        )

        bootstrap_model.fit(X_bootstrap)

        predicted_full = bootstrap_model.predict(X)

        ari = adjusted_rand_score(
            full_labels,
            predicted_full,
        )

        bootstrap_ari.append(ari)

    stability_rows.append(
        {
            "n_clusters": n_clusters,
            "mean_bootstrap_ari": float(np.mean(bootstrap_ari)),
            "median_bootstrap_ari": float(np.median(bootstrap_ari)),
            "p05_bootstrap_ari": float(
                np.quantile(
                    bootstrap_ari,
                    0.05,
                )
            ),
            "minimum_bootstrap_ari": float(np.min(bootstrap_ari)),
        }
    )

stability = pd.DataFrame(stability_rows)

model_comparison = model_comparison.merge(
    stability,
    on="n_clusters",
    how="left",
)

print("\n" + "=" * 80)
print("MODEL STABILITY")
print("=" * 80)

display(model_comparison.sort_values("n_clusters"))

# ------------------------------------------------------------
# 9. SELECT PREFERRED SOLUTION
# ------------------------------------------------------------

# Selection hierarchy:
# 1. No cluster smaller than 10% of the cohort
# 2. Median bootstrap ARI >=0.75
# 3. Lowest BIC among eligible models
#
# If none fulfills both criteria, choose the model with the
# highest median bootstrap ARI, then the lowest BIC.

eligible_models = model_comparison.loc[
    (model_comparison["minimum_cluster_percent"] >= 10)
    & (model_comparison["median_bootstrap_ari"] >= 0.75)
].copy()

if not eligible_models.empty:

    preferred_n_clusters = int(
        eligible_models.sort_values(
            [
                "bic",
                "silhouette",
            ],
            ascending=[
                True,
                False,
            ],
        ).iloc[0]["n_clusters"]
    )

    selection_reason = (
        "Lowest BIC among solutions with "
        "minimum cluster size >=10% and "
        "median bootstrap ARI >=0.75."
    )

else:

    preferred_n_clusters = int(
        model_comparison.sort_values(
            [
                "median_bootstrap_ari",
                "bic",
            ],
            ascending=[
                False,
                True,
            ],
        ).iloc[0]["n_clusters"]
    )

    selection_reason = (
        "No solution fulfilled all prespecified "
        "stability and size criteria. The solution "
        "with the highest median bootstrap ARI was selected."
    )

print("\nPreferred number of clusters:", preferred_n_clusters)
print("Selection reason:", selection_reason)

# ------------------------------------------------------------
# 10. ASSIGN PREFERRED PHENOTYPES
# ------------------------------------------------------------

preferred_model = candidate_models[preferred_n_clusters]["model"]

raw_labels = preferred_model.predict(X)

posterior_probabilities = preferred_model.predict_proba(X)

phenotype_data["phenotype_raw"] = raw_labels

phenotype_data["phenotype_max_probability"] = posterior_probabilities.max(axis=1)

phenotype_data["phenotype_assignment_uncertain"] = (
    phenotype_data["phenotype_max_probability"] < 0.70
)

# ------------------------------------------------------------
# 11. CLUSTER CHARACTERISTICS
# ------------------------------------------------------------

cluster_characteristics = phenotype_data.groupby(
    "phenotype_raw",
    as_index=False,
).agg(
    n=(
        "patientid",
        "size",
    ),
    dbp_median=(
        "dbp_baseline",
        "median",
    ),
    dbp_mean=(
        "dbp_baseline",
        "mean",
    ),
    pulse_pressure_median=(
        "pulse_pressure_baseline",
        "median",
    ),
    pulse_pressure_mean=(
        "pulse_pressure_baseline",
        "mean",
    ),
    heart_rate_median=(
        "heart_rate_baseline",
        "median",
    ),
    heart_rate_mean=(
        "heart_rate_baseline",
        "mean",
    ),
    assignment_probability_median=(
        "phenotype_max_probability",
        "median",
    ),
    uncertain_assignments=(
        "phenotype_assignment_uncertain",
        "sum",
    ),
)

cluster_characteristics["percent"] = (
    100 * cluster_characteristics["n"] / len(phenotype_data)
)

# ------------------------------------------------------------
# 12. IDENTIFY THE MOST VASOPLEGIA-COMPATIBLE PHENOTYPE
# ------------------------------------------------------------

# A cluster-level physiological score is used only to order
# and interpret the already-derived clusters.
#
# It does NOT participate in cluster formation.
#
# More vasoplegia-compatible:
# - Lower DBP
# - Preserved/higher pulse pressure
#
# Heart rate is not included in this post-hoc score because
# tachycardia may reflect severity, fever, pain, sedation,
# atrial arrhythmia or catecholamine exposure.

cluster_characteristics["dbp_cluster_z"] = (
    cluster_characteristics["dbp_median"] - cluster_characteristics["dbp_median"].mean()
) / cluster_characteristics["dbp_median"].std(ddof=0)

cluster_characteristics["pulse_pressure_cluster_z"] = (
    cluster_characteristics["pulse_pressure_median"]
    - cluster_characteristics["pulse_pressure_median"].mean()
) / cluster_characteristics["pulse_pressure_median"].std(ddof=0)

cluster_characteristics["vasoplegia_compatibility_score"] = (
    -cluster_characteristics["dbp_cluster_z"]
    + cluster_characteristics["pulse_pressure_cluster_z"]
)

cluster_characteristics = cluster_characteristics.sort_values(
    "vasoplegia_compatibility_score",
    ascending=False,
).reset_index(drop=True)

# Ordered labels:
# Phenotype 1 = most compatible with vasoplegia.
cluster_characteristics["phenotype_ordered"] = np.arange(
    1,
    len(cluster_characteristics) + 1,
)

raw_to_ordered = dict(
    zip(
        cluster_characteristics["phenotype_raw"],
        cluster_characteristics["phenotype_ordered"],
    )
)

phenotype_data["phenotype"] = phenotype_data["phenotype_raw"].map(raw_to_ordered)

phenotype_data["candidate_vasoplegic_phenotype"] = phenotype_data["phenotype"].eq(1)

print("\n" + "=" * 80)
print("CLUSTER MACROHEMODYNAMIC CHARACTERISTICS")
print("=" * 80)

display(
    cluster_characteristics[
        [
            "phenotype_ordered",
            "phenotype_raw",
            "n",
            "percent",
            "dbp_median",
            "pulse_pressure_median",
            "heart_rate_median",
            "assignment_probability_median",
            "uncertain_assignments",
            "vasoplegia_compatibility_score",
        ]
    ]
)

# ------------------------------------------------------------
# 13. MERGE PHENOTYPES INTO FULL DATASET
# ------------------------------------------------------------

phenotype_assignments = phenotype_data[
    [
        "patientid",
        "phenotype",
        "phenotype_raw",
        "candidate_vasoplegic_phenotype",
        "phenotype_max_probability",
        "phenotype_assignment_uncertain",
    ]
].copy()

analysis_with_phenotype = analysis.merge(
    phenotype_assignments,
    on="patientid",
    how="left",
)

# ------------------------------------------------------------
# 14. CALCULATE CLASSIC INDICES
# ------------------------------------------------------------

analysis_with_phenotype["diastolic_shock_index"] = (
    analysis_with_phenotype["heart_rate_baseline"]
    / analysis_with_phenotype["dbp_baseline"]
)

analysis_with_phenotype["shock_index"] = (
    analysis_with_phenotype["heart_rate_baseline"]
    / analysis_with_phenotype["sbp_baseline"]
)

analysis_with_phenotype["modified_shock_index"] = (
    analysis_with_phenotype["heart_rate_baseline"]
    / analysis_with_phenotype["map_baseline"]
)

# ------------------------------------------------------------
# 15. NORMALIZE HOURLY NOREPINEPHRINE BY WEIGHT
# ------------------------------------------------------------

weight_data = analysis_with_phenotype[
    [
        "patientid",
        "weight_kg",
    ]
].drop_duplicates()

hourly = hourly.merge(
    weight_data,
    on="patientid",
    how="left",
)

if "norepi_ug_min_winsor" in hourly.columns:

    hourly_ne_column = "norepi_ug_min_winsor"

elif "norepi_ug_min_clean" in hourly.columns:

    hourly_ne_column = "norepi_ug_min_clean"

elif "norepi_ug_min" in hourly.columns:

    hourly_ne_column = "norepi_ug_min"

else:

    raise RuntimeError("No hourly norepinephrine column was found.")

hourly["norepi_ug_kg_min"] = hourly[hourly_ne_column] / hourly["weight_kg"]

hourly.loc[
    ~hourly["norepi_ug_kg_min"].between(
        0.0001,
        5.0,
        inclusive="both",
    ),
    "norepi_ug_kg_min",
] = np.nan

hourly = hourly.merge(
    phenotype_assignments[
        [
            "patientid",
            "phenotype",
            "candidate_vasoplegic_phenotype",
        ]
    ],
    on="patientid",
    how="left",
)

# ------------------------------------------------------------
# 16. LONGITUDINAL WEIGHTED NOREPINEPHRINE METRICS
# ------------------------------------------------------------


def calculate_slope(
    group,
    value_column,
    minimum_points=3,
):

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

    if len(valid) < minimum_points or valid["hour_bin"].nunique() < minimum_points:
        return np.nan

    return float(
        np.polyfit(
            valid["hour_bin"].astype(float),
            valid[value_column].astype(float),
            1,
        )[0]
    )


def calculate_auc(
    group,
    value_column,
    minimum_points=2,
):

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

    if len(valid) < minimum_points or valid["hour_bin"].nunique() < minimum_points:
        return np.nan

    return float(
        np.trapezoid(
            valid[value_column].astype(float),
            valid["hour_bin"].astype(float),
        )
    )


validation_hourly = hourly.loc[
    hourly["hour_bin"].between(
        1,
        5,
        inclusive="both",
    )
].copy()

weighted_longitudinal_rows = []

for patientid, group in validation_hourly.groupby(
    "patientid",
    sort=False,
):

    ne_values = group["norepi_ug_kg_min"].dropna()

    dbp_values = group["dbp"].dropna()

    map_values = group["map"].dropna()

    pp_values = group["pulse_pressure"].dropna()

    weighted_longitudinal_rows.append(
        {
            "patientid": int(patientid),
            "norepi_ug_kg_min_median_h1_h6": (
                float(ne_values.median()) if not ne_values.empty else np.nan
            ),
            "norepi_ug_kg_min_max_h1_h6": (
                float(ne_values.max()) if not ne_values.empty else np.nan
            ),
            "norepi_ug_kg_min_slope_h1_h6": calculate_slope(
                group,
                "norepi_ug_kg_min",
            ),
            "norepi_ug_kg_min_auc_h1_h6": calculate_auc(
                group,
                "norepi_ug_kg_min",
            ),
            "dbp_median_h1_h6_weighted_validation": (
                float(dbp_values.median()) if not dbp_values.empty else np.nan
            ),
            "map_median_h1_h6_weighted_validation": (
                float(map_values.median()) if not map_values.empty else np.nan
            ),
            "pulse_pressure_median_h1_h6_validation": (
                float(pp_values.median()) if not pp_values.empty else np.nan
            ),
            "proportion_hours_dbp_lt_50_h1_h6_validation": (
                float((dbp_values < 50).mean()) if not dbp_values.empty else np.nan
            ),
            "proportion_hours_dbp_lt_45_h1_h6_validation": (
                float((dbp_values < 45).mean()) if not dbp_values.empty else np.nan
            ),
            "proportion_hours_map_lt_65_h1_h6_validation": (
                float((map_values < 65).mean()) if not map_values.empty else np.nan
            ),
            "n_valid_ne_hours_h1_h6": int(ne_values.size),
            "n_valid_dbp_hours_h1_h6": int(dbp_values.size),
        }
    )

weighted_longitudinal = pd.DataFrame(weighted_longitudinal_rows)

analysis_with_phenotype = analysis_with_phenotype.merge(
    weighted_longitudinal,
    on="patientid",
    how="left",
)

# ------------------------------------------------------------
# 17. PHYSIOLOGICAL CHARACTERIZATION BY PHENOTYPE
# ------------------------------------------------------------

continuous_characterization_variables = [
    # Macrohemodynamics used in clustering
    "dbp_baseline",
    "pulse_pressure_baseline",
    "heart_rate_baseline",
    # Variables not used in clustering
    "map_baseline",
    "sbp_baseline",
    "lactate_baseline",
    "norepi_ug_kg_min_baseline_winsor",
    "diastolic_shock_index",
    "shock_index",
    "modified_shock_index",
    # Longitudinal physiological validation
    "dbp_median_h1_h6_weighted_validation",
    "map_median_h1_h6_weighted_validation",
    "pulse_pressure_median_h1_h6_validation",
    "norepi_ug_kg_min_median_h1_h6",
    "norepi_ug_kg_min_max_h1_h6",
    "norepi_ug_kg_min_slope_h1_h6",
    "norepi_ug_kg_min_auc_h1_h6",
    "proportion_hours_dbp_lt_50_h1_h6_validation",
    "proportion_hours_dbp_lt_45_h1_h6_validation",
    "proportion_hours_map_lt_65_h1_h6_validation",
    # Existing longitudinal variables
    "delta_dbp_followup_minus_baseline",
    "delta_map_followup_minus_baseline",
    "delta_pp_followup_minus_baseline",
]

characterization_rows = []

for variable in continuous_characterization_variables:

    if variable not in analysis_with_phenotype.columns:
        continue

    for phenotype, group in analysis_with_phenotype.dropna(
        subset=["phenotype"]
    ).groupby("phenotype"):

        values = pd.to_numeric(
            group[variable],
            errors="coerce",
        ).dropna()

        characterization_rows.append(
            {
                "variable": variable,
                "phenotype": int(phenotype),
                "n": int(values.size),
                "mean": (float(values.mean()) if not values.empty else np.nan),
                "sd": (float(values.std()) if len(values) > 1 else np.nan),
                "median": (float(values.median()) if not values.empty else np.nan),
                "q25": (float(values.quantile(0.25)) if not values.empty else np.nan),
                "q75": (float(values.quantile(0.75)) if not values.empty else np.nan),
            }
        )

phenotype_characterization = pd.DataFrame(characterization_rows)

# ------------------------------------------------------------
# 18. GLOBAL BETWEEN-PHENOTYPE TESTS
# ------------------------------------------------------------

global_test_rows = []

for variable in continuous_characterization_variables:

    if variable not in analysis_with_phenotype.columns:
        continue

    groups = []

    for phenotype in sorted(analysis_with_phenotype["phenotype"].dropna().unique()):

        values = pd.to_numeric(
            analysis_with_phenotype.loc[
                analysis_with_phenotype["phenotype"].eq(phenotype),
                variable,
            ],
            errors="coerce",
        ).dropna()

        if not values.empty:
            groups.append(values)

    if len(groups) >= 2:

        statistic, p_value = kruskal(*groups)

        global_test_rows.append(
            {
                "variable": variable,
                "test": "Kruskal-Wallis",
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

global_tests = pd.DataFrame(global_test_rows)

# ------------------------------------------------------------
# 19. BINARY CHARACTERIZATION
# ------------------------------------------------------------

binary_variables = [
    "probable_septic_shock_strict",
    "lactate_gt_2",
    "candidate_vasoplegic_phenotype",
]

binary_rows = []
binary_test_rows = []

for variable in binary_variables:

    if variable not in analysis_with_phenotype.columns:
        continue

    temp = analysis_with_phenotype.loc[
        analysis_with_phenotype["phenotype"].notna()
        & analysis_with_phenotype[variable].notna(),
        [
            "phenotype",
            variable,
        ],
    ].copy()

    for phenotype, group in temp.groupby("phenotype"):

        binary_rows.append(
            {
                "variable": variable,
                "phenotype": int(phenotype),
                "n": int(len(group)),
                "events": int(group[variable].astype(bool).sum()),
                "percent": float(100 * group[variable].astype(bool).mean()),
            }
        )

    contingency = pd.crosstab(
        temp["phenotype"],
        temp[variable],
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
# 20. CREATE PHYSIOLOGICAL VALIDATION SUMMARY
# ------------------------------------------------------------

validation_variables = [
    "dbp_median_h1_h6_weighted_validation",
    "pulse_pressure_median_h1_h6_validation",
    "norepi_ug_kg_min_median_h1_h6",
    "norepi_ug_kg_min_auc_h1_h6",
    "norepi_ug_kg_min_slope_h1_h6",
    "proportion_hours_dbp_lt_50_h1_h6_validation",
    "proportion_hours_dbp_lt_45_h1_h6_validation",
    "proportion_hours_map_lt_65_h1_h6_validation",
]

validation_summary = (
    phenotype_characterization.loc[
        phenotype_characterization["variable"].isin(validation_variables)
    ]
    .copy()
    .sort_values(
        [
            "variable",
            "phenotype",
        ]
    )
)

print("\n" + "=" * 80)
print("LONGITUDINAL PHYSIOLOGICAL VALIDATION")
print("=" * 80)

display(
    validation_summary[
        [
            "variable",
            "phenotype",
            "n",
            "median",
            "q25",
            "q75",
        ]
    ]
)

# ------------------------------------------------------------
# 21. MACROHEMODYNAMIC SCATTERPLOT
# ------------------------------------------------------------

plot_data = phenotype_data.loc[phenotype_data["phenotype"].notna()].copy()

plt.figure(figsize=(9, 7))

for phenotype in sorted(plot_data["phenotype"].unique()):

    subset = plot_data.loc[plot_data["phenotype"].eq(phenotype)]

    plt.scatter(
        subset["dbp_baseline"],
        subset["pulse_pressure_baseline"],
        alpha=0.45,
        s=22,
        label=f"Phenotype {int(phenotype)}",
    )

plt.xlabel("Baseline diastolic arterial pressure (mmHg)")

plt.ylabel("Baseline pulse pressure (mmHg)")

plt.title("Macrohemodynamic phenotypes in probable septic shock")

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "step05_macrohemodynamic_phenotype_scatterplot.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# ------------------------------------------------------------
# 22. MODEL-DIAGNOSTIC FIGURE
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    model_comparison["n_clusters"],
    model_comparison["bic"],
    marker="o",
)

plt.xlabel("Number of Gaussian mixture components")

plt.ylabel("Bayesian information criterion")

plt.title("Model selection for macrohemodynamic phenotypes")

plt.xticks([2, 3, 4])

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "step05_gmm_bic_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# ------------------------------------------------------------
# 23. SAVE TRANSPORTABLE MODEL
# ------------------------------------------------------------

joblib.dump(
    scaler,
    MODEL_DIR / "step05_macrohemodynamic_standard_scaler.joblib",
)

joblib.dump(
    preferred_model,
    MODEL_DIR / "step05_preferred_macrohemodynamic_gmm.joblib",
)

transport_parameters = {
    "phenotype_variables_original": PHENOTYPE_VARIABLES,
    "phenotype_variables_winsorized": PHENOTYPE_VARIABLES_WINSOR,
    "winsor_limits": winsor_limits,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "n_clusters": int(preferred_n_clusters),
    "raw_to_ordered_phenotype_mapping": {
        str(key): int(value) for key, value in raw_to_ordered.items()
    },
    "candidate_vasoplegic_phenotype": 1,
    "selection_reason": selection_reason,
    "phenotype_window": "Median values from 0 to 1 hour after norepinephrine time zero",
    "validation_window": "Hourly values from >1 to 6 hours after time zero",
}

with open(
    MODEL_DIR / "step05_transportable_phenotype_parameters.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        transport_parameters,
        f,
        indent=2,
        ensure_ascii=False,
    )

# ------------------------------------------------------------
# 24. SAVE OUTPUTS
# ------------------------------------------------------------

analysis_with_phenotype.to_parquet(
    DERIVED_DIR / "step05_analysis_dataset_with_macrohemodynamic_phenotypes.parquet",
    index=False,
)

analysis_with_phenotype.to_csv(
    TABLE_DIR / "step05_analysis_dataset_with_macrohemodynamic_phenotypes.csv",
    index=False,
)

phenotype_assignments.to_csv(
    TABLE_DIR / "step05_phenotype_assignments.csv",
    index=False,
)

model_comparison.to_csv(
    TABLE_DIR / "step05_gmm_model_comparison.csv",
    index=False,
)

cluster_characteristics.to_csv(
    TABLE_DIR / "step05_cluster_macrohemodynamic_characteristics.csv",
    index=False,
)

phenotype_characterization.to_csv(
    TABLE_DIR / "step05_full_phenotype_characterization.csv",
    index=False,
)

validation_summary.to_csv(
    TABLE_DIR / "step05_longitudinal_physiological_validation.csv",
    index=False,
)

global_tests.to_csv(
    TABLE_DIR / "step05_global_continuous_tests.csv",
    index=False,
)

binary_characterization.to_csv(
    TABLE_DIR / "step05_binary_phenotype_characterization.csv",
    index=False,
)

binary_tests.to_csv(
    TABLE_DIR / "step05_binary_global_tests.csv",
    index=False,
)

run_summary = {
    "input_cohort_n": int(len(analysis)),
    "phenotype_derivation_n": int(len(phenotype_data)),
    "preferred_n_clusters": int(preferred_n_clusters),
    "selection_reason": selection_reason,
    "candidate_vasoplegic_phenotype": 1,
    "uncertain_assignments_n": int(
        phenotype_data["phenotype_assignment_uncertain"].sum()
    ),
    "uncertain_assignments_percent": float(
        100 * phenotype_data["phenotype_assignment_uncertain"].mean()
    ),
    "phenotype_variables": PHENOTYPE_VARIABLES,
    "norepinephrine_used_in_clustering": False,
    "lactate_used_in_clustering": False,
    "outcomes_used_in_clustering": False,
}

with open(
    LOG_DIR / "step05_macrohemodynamic_phenotype_summary.json",
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
# 25. FINAL DISPLAY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 05 COMPLETE")
print("=" * 80)

print(
    "\nPreferred solution:",
    preferred_n_clusters,
    "phenotypes",
)

print("Candidate vasoplegic phenotype: Phenotype 1")

print("\nPhenotype derivation variables:")

for variable in PHENOTYPE_VARIABLES:
    print(" -", variable)

print("\nNorepinephrine was NOT used to create the phenotypes.")

print("Norepinephrine was used only for longitudinal validation.")

print(
    "\nMain dataset saved to:\n"
    f"{DERIVED_DIR / 'step05_analysis_dataset_with_macrohemodynamic_phenotypes.parquet'}"
)

print(
    "\nTransportable model parameters saved to:\n"
    f"{MODEL_DIR / 'step05_transportable_phenotype_parameters.json'}"
)
