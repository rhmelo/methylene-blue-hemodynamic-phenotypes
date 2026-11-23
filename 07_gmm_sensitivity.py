"""Source: original notebook cell 14. Run through run.py."""

from common import display, phase_dir

# ============================================================
# HiRID Methylene Blue Hemodynamic Phenotypes
# STEP 06 — Three-phenotype macrohemodynamic models
#
# Main objective:
# Compare 3-cluster Gaussian mixture models using different
# covariance structures:
# - full
# - tied
# - diag
# - spherical
#
# Reference model:
# - 2 clusters, covariance_type="full"
#
# Variables used to CREATE phenotypes:
# - Baseline diastolic arterial pressure
# - Baseline pulse pressure
# - Baseline heart rate
#
# Variables NOT used to create phenotypes:
# - Norepinephrine
# - Lactate
# - DSI / SI / MSI
# - Longitudinal trajectories
# - Outcomes
#
# The purpose is to determine whether a stable and
# physiologically meaningful 3-phenotype solution exists.
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

from scipy.stats import kruskal

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_BOOTSTRAPS = 100

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
# 3. PHENOTYPE VARIABLES
# ------------------------------------------------------------

PHENOTYPE_VARIABLES = [
    "dbp_baseline",
    "pulse_pressure_baseline",
    "heart_rate_baseline",
]

missing_variables = [
    variable for variable in PHENOTYPE_VARIABLES if variable not in analysis.columns
]

if missing_variables:
    raise RuntimeError(f"Missing variables: {missing_variables}")

phenotype_data = analysis.loc[analysis[PHENOTYPE_VARIABLES].notna().all(axis=1)].copy()

# Conservative plausibility ranges
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
    "Patients available for phenotype derivation:",
    len(phenotype_data),
)

# ------------------------------------------------------------
# 4. WINSORIZATION
# ------------------------------------------------------------

winsor_limits = {}

for variable in PHENOTYPE_VARIABLES:

    lower_limit = phenotype_data[variable].quantile(0.005)

    upper_limit = phenotype_data[variable].quantile(0.995)

    winsor_limits[variable] = {
        "lower": float(lower_limit),
        "upper": float(upper_limit),
    }

    phenotype_data[f"{variable}_winsor"] = phenotype_data[variable].clip(
        lower=lower_limit,
        upper=upper_limit,
    )

PHENOTYPE_VARIABLES_WINSOR = [f"{variable}_winsor" for variable in PHENOTYPE_VARIABLES]

# ------------------------------------------------------------
# 5. STANDARDIZATION
# ------------------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(phenotype_data[PHENOTYPE_VARIABLES_WINSOR])

# ------------------------------------------------------------
# 6. MODEL SPECIFICATIONS
# ------------------------------------------------------------

MODEL_SPECIFICATIONS = [
    {
        "model_name": "2_full_reference",
        "n_clusters": 2,
        "covariance_type": "full",
    },
    {
        "model_name": "3_full",
        "n_clusters": 3,
        "covariance_type": "full",
    },
    {
        "model_name": "3_tied",
        "n_clusters": 3,
        "covariance_type": "tied",
    },
    {
        "model_name": "3_diag",
        "n_clusters": 3,
        "covariance_type": "diag",
    },
    {
        "model_name": "3_spherical",
        "n_clusters": 3,
        "covariance_type": "spherical",
    },
]

fitted_models = {}
model_results = []

# ------------------------------------------------------------
# 7. FIT MODELS
# ------------------------------------------------------------

for specification in MODEL_SPECIFICATIONS:

    model_name = specification["model_name"]

    n_clusters = specification["n_clusters"]

    covariance_type = specification["covariance_type"]

    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        n_init=100,
        max_iter=2000,
        random_state=RANDOM_STATE,
        reg_covar=1e-6,
    )

    labels = model.fit_predict(X)

    posterior_probabilities = model.predict_proba(X)

    cluster_counts = pd.Series(labels).value_counts().sort_index()

    minimum_cluster_n = int(cluster_counts.min())

    minimum_cluster_percent = float(100 * minimum_cluster_n / len(labels))

    median_max_probability = float(np.median(posterior_probabilities.max(axis=1)))

    uncertain_assignment_percent = float(
        100 * (posterior_probabilities.max(axis=1) < 0.70).mean()
    )

    silhouette = float(
        silhouette_score(
            X,
            labels,
        )
    )

    fitted_models[model_name] = {
        "model": model,
        "labels": labels,
        "posterior_probabilities": posterior_probabilities,
    }

    model_results.append(
        {
            "model_name": model_name,
            "n_clusters": n_clusters,
            "covariance_type": covariance_type,
            "bic": float(model.bic(X)),
            "aic": float(model.aic(X)),
            "silhouette": silhouette,
            "minimum_cluster_n": minimum_cluster_n,
            "minimum_cluster_percent": minimum_cluster_percent,
            "median_max_probability": median_max_probability,
            "uncertain_assignment_percent": uncertain_assignment_percent,
            "converged": bool(model.converged_),
            "iterations": int(model.n_iter_),
        }
    )

model_comparison = pd.DataFrame(model_results)

model_comparison["delta_bic_vs_best"] = (
    model_comparison["bic"] - model_comparison["bic"].min()
)

reference_bic = float(
    model_comparison.loc[
        model_comparison["model_name"].eq("2_full_reference"),
        "bic",
    ].iloc[0]
)

model_comparison["delta_bic_vs_2_cluster_reference"] = (
    model_comparison["bic"] - reference_bic
)

print("\n" + "=" * 80)
print("INITIAL MODEL COMPARISON")
print("=" * 80)

display(
    model_comparison.sort_values(
        [
            "n_clusters",
            "covariance_type",
        ]
    )
)

# ------------------------------------------------------------
# 8. BOOTSTRAP STABILITY
# ------------------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

stability_results = []

for specification in MODEL_SPECIFICATIONS:

    model_name = specification["model_name"]

    n_clusters = specification["n_clusters"]

    covariance_type = specification["covariance_type"]

    full_labels = fitted_models[model_name]["labels"]

    ari_values = []

    for bootstrap_number in range(N_BOOTSTRAPS):

        bootstrap_indices = rng.choice(
            len(X),
            size=len(X),
            replace=True,
        )

        X_bootstrap = X[bootstrap_indices]

        bootstrap_model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            n_init=20,
            max_iter=1500,
            random_state=(RANDOM_STATE + bootstrap_number + n_clusters * 1000),
            reg_covar=1e-6,
        )

        bootstrap_model.fit(X_bootstrap)

        predicted_full_data = bootstrap_model.predict(X)

        ari = adjusted_rand_score(
            full_labels,
            predicted_full_data,
        )

        ari_values.append(ari)

    stability_results.append(
        {
            "model_name": model_name,
            "mean_bootstrap_ari": float(np.mean(ari_values)),
            "median_bootstrap_ari": float(np.median(ari_values)),
            "p05_bootstrap_ari": float(
                np.quantile(
                    ari_values,
                    0.05,
                )
            ),
            "p25_bootstrap_ari": float(
                np.quantile(
                    ari_values,
                    0.25,
                )
            ),
            "p75_bootstrap_ari": float(
                np.quantile(
                    ari_values,
                    0.75,
                )
            ),
            "minimum_bootstrap_ari": float(np.min(ari_values)),
            "maximum_bootstrap_ari": float(np.max(ari_values)),
        }
    )

stability_table = pd.DataFrame(stability_results)

model_comparison = model_comparison.merge(
    stability_table,
    on="model_name",
    how="left",
)

print("\n" + "=" * 80)
print("MODEL COMPARISON WITH BOOTSTRAP STABILITY")
print("=" * 80)

display(
    model_comparison.sort_values(
        [
            "n_clusters",
            "covariance_type",
        ]
    )
)

# ------------------------------------------------------------
# 9. CREATE CLUSTER PROFILES FOR EVERY MODEL
# ------------------------------------------------------------

all_model_profiles = []

for specification in MODEL_SPECIFICATIONS:

    model_name = specification["model_name"]

    labels = fitted_models[model_name]["labels"]

    posterior_probabilities = fitted_models[model_name]["posterior_probabilities"]

    temporary_data = phenotype_data[
        [
            "patientid",
            "dbp_baseline",
            "pulse_pressure_baseline",
            "heart_rate_baseline",
        ]
    ].copy()

    temporary_data["cluster_raw"] = labels

    temporary_data["maximum_posterior_probability"] = posterior_probabilities.max(
        axis=1
    )

    temporary_data["uncertain_assignment"] = (
        temporary_data["maximum_posterior_probability"] < 0.70
    )

    profile = temporary_data.groupby(
        "cluster_raw",
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
        dbp_q25=(
            "dbp_baseline",
            lambda x: x.quantile(0.25),
        ),
        dbp_q75=(
            "dbp_baseline",
            lambda x: x.quantile(0.75),
        ),
        pulse_pressure_median=(
            "pulse_pressure_baseline",
            "median",
        ),
        pulse_pressure_q25=(
            "pulse_pressure_baseline",
            lambda x: x.quantile(0.25),
        ),
        pulse_pressure_q75=(
            "pulse_pressure_baseline",
            lambda x: x.quantile(0.75),
        ),
        heart_rate_median=(
            "heart_rate_baseline",
            "median",
        ),
        heart_rate_q25=(
            "heart_rate_baseline",
            lambda x: x.quantile(0.25),
        ),
        heart_rate_q75=(
            "heart_rate_baseline",
            lambda x: x.quantile(0.75),
        ),
        median_assignment_probability=(
            "maximum_posterior_probability",
            "median",
        ),
        uncertain_assignments=(
            "uncertain_assignment",
            "sum",
        ),
    )

    profile["percent"] = 100 * profile["n"] / len(temporary_data)

    profile["model_name"] = model_name

    profile["n_clusters"] = specification["n_clusters"]

    profile["covariance_type"] = specification["covariance_type"]

    all_model_profiles.append(profile)

all_profiles = pd.concat(
    all_model_profiles,
    ignore_index=True,
)

print("\n" + "=" * 80)
print("ALL CLUSTER PROFILES")
print("=" * 80)

display(
    all_profiles[
        [
            "model_name",
            "cluster_raw",
            "n",
            "percent",
            "dbp_median",
            "pulse_pressure_median",
            "heart_rate_median",
            "median_assignment_probability",
            "uncertain_assignments",
        ]
    ].sort_values(
        [
            "model_name",
            "cluster_raw",
        ]
    )
)

# ------------------------------------------------------------
# 10. PHYSIOLOGICAL SCREENING OF 3-CLUSTER MODELS
# ------------------------------------------------------------

# Post-hoc screening:
#
# A potentially vasoplegia-compatible cluster should show:
# - comparatively low DBP
# - PP not simultaneously the lowest among all clusters
#
# This is not used to fit the model.
# It is used only to assess physiological interpretability.

three_cluster_profiles = all_profiles.loc[all_profiles["n_clusters"].eq(3)].copy()

physiological_screen_rows = []

for model_name, group in three_cluster_profiles.groupby("model_name"):

    group = group.copy()

    group["dbp_rank_ascending"] = group["dbp_median"].rank(
        method="min",
        ascending=True,
    )

    group["pp_rank_descending"] = group["pulse_pressure_median"].rank(
        method="min",
        ascending=False,
    )

    lowest_dbp_cluster = int(
        group.loc[
            group["dbp_median"].idxmin(),
            "cluster_raw",
        ]
    )

    lowest_dbp_row = group.loc[group["cluster_raw"].eq(lowest_dbp_cluster)].iloc[0]

    pp_values = group["pulse_pressure_median"]

    has_low_dbp_preserved_pp = bool(
        lowest_dbp_row["pulse_pressure_median"] > pp_values.min()
    )

    physiological_screen_rows.append(
        {
            "model_name": model_name,
            "lowest_dbp_cluster": lowest_dbp_cluster,
            "lowest_dbp_median": float(lowest_dbp_row["dbp_median"]),
            "lowest_dbp_cluster_pp_median": float(
                lowest_dbp_row["pulse_pressure_median"]
            ),
            "minimum_pp_across_clusters": float(pp_values.min()),
            "maximum_pp_across_clusters": float(pp_values.max()),
            "has_low_dbp_with_pp_above_minimum": has_low_dbp_preserved_pp,
            "lowest_dbp_cluster_percent": float(lowest_dbp_row["percent"]),
        }
    )

physiological_screen = pd.DataFrame(physiological_screen_rows)

physiological_screen = physiological_screen.merge(
    model_comparison[
        [
            "model_name",
            "bic",
            "delta_bic_vs_2_cluster_reference",
            "silhouette",
            "minimum_cluster_percent",
            "median_bootstrap_ari",
            "p05_bootstrap_ari",
            "uncertain_assignment_percent",
        ]
    ],
    on="model_name",
    how="left",
)

print("\n" + "=" * 80)
print("PHYSIOLOGICAL SCREENING OF 3-CLUSTER MODELS")
print("=" * 80)

display(
    physiological_screen.sort_values(
        [
            "median_bootstrap_ari",
            "bic",
        ],
        ascending=[
            False,
            True,
        ],
    )
)

# ------------------------------------------------------------
# 11. RANK 3-CLUSTER CANDIDATES
# ------------------------------------------------------------

# Prespecified requirements:
# - Minimum cluster >=10%
# - Median bootstrap ARI >=0.70
# - 5th percentile bootstrap ARI >=0.50
# - Median posterior probability preferably >=0.75
#
# Physiological plausibility is considered after statistical
# eligibility, not as a substitute for stability.

three_model_comparison = model_comparison.loc[
    model_comparison["n_clusters"].eq(3)
].copy()

three_model_comparison["statistically_eligible"] = (
    three_model_comparison["minimum_cluster_percent"].ge(10)
    & three_model_comparison["median_bootstrap_ari"].ge(0.70)
    & three_model_comparison["p05_bootstrap_ari"].ge(0.50)
)

three_model_ranking = three_model_comparison.merge(
    physiological_screen[
        [
            "model_name",
            "has_low_dbp_with_pp_above_minimum",
            "lowest_dbp_cluster_percent",
        ]
    ],
    on="model_name",
    how="left",
)

three_model_ranking = three_model_ranking.sort_values(
    [
        "statistically_eligible",
        "has_low_dbp_with_pp_above_minimum",
        "median_bootstrap_ari",
        "bic",
    ],
    ascending=[
        False,
        False,
        False,
        True,
    ],
).reset_index(drop=True)

print("\n" + "=" * 80)
print("RANKING OF 3-CLUSTER MODELS")
print("=" * 80)

display(three_model_ranking)

# ------------------------------------------------------------
# 12. SELECT BEST 3-CLUSTER CANDIDATE
# ------------------------------------------------------------

statistically_eligible = three_model_ranking.loc[
    three_model_ranking["statistically_eligible"]
]

if not statistically_eligible.empty:

    best_three_model_name = statistically_eligible.iloc[0]["model_name"]

    selection_status = "A statistically eligible 3-cluster " "solution was identified."

else:

    best_three_model_name = three_model_ranking.iloc[0]["model_name"]

    selection_status = (
        "No 3-cluster model fulfilled all "
        "prespecified stability requirements. "
        "The highest-ranked model is retained "
        "for exploratory inspection only."
    )

print(
    "\nBest 3-cluster candidate:",
    best_three_model_name,
)

print(
    "Selection status:",
    selection_status,
)

# ------------------------------------------------------------
# 13. ASSIGN BEST 3-CLUSTER MODEL
# ------------------------------------------------------------

best_three_model = fitted_models[best_three_model_name]["model"]

best_three_labels = fitted_models[best_three_model_name]["labels"]

best_three_probabilities = fitted_models[best_three_model_name][
    "posterior_probabilities"
]

best_three_data = phenotype_data[
    [
        "patientid",
        "dbp_baseline",
        "pulse_pressure_baseline",
        "heart_rate_baseline",
    ]
].copy()

best_three_data["cluster_raw"] = best_three_labels

best_three_data["cluster_probability"] = best_three_probabilities.max(axis=1)

best_three_data["uncertain_assignment"] = best_three_data["cluster_probability"] < 0.70

# ------------------------------------------------------------
# 14. ORDER THE THREE PHENOTYPES PHYSIOLOGICALLY
# ------------------------------------------------------------

best_profile = best_three_data.groupby(
    "cluster_raw",
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
    pulse_pressure_median=(
        "pulse_pressure_baseline",
        "median",
    ),
    heart_rate_median=(
        "heart_rate_baseline",
        "median",
    ),
    assignment_probability_median=(
        "cluster_probability",
        "median",
    ),
    uncertain_assignments=(
        "uncertain_assignment",
        "sum",
    ),
)

best_profile["percent"] = 100 * best_profile["n"] / len(best_three_data)

# Physiological ordering score:
# - lower DBP increases vasoplegia compatibility
# - higher PP increases vasoplegia compatibility
#
# This score is for ordering only.
# It does not create the clusters.

best_profile["dbp_z_between_clusters"] = (
    best_profile["dbp_median"] - best_profile["dbp_median"].mean()
) / best_profile["dbp_median"].std(ddof=0)

best_profile["pp_z_between_clusters"] = (
    best_profile["pulse_pressure_median"] - best_profile["pulse_pressure_median"].mean()
) / best_profile["pulse_pressure_median"].std(ddof=0)

best_profile["vasoplegia_compatibility_score"] = (
    -best_profile["dbp_z_between_clusters"] + best_profile["pp_z_between_clusters"]
)

best_profile = best_profile.sort_values(
    "vasoplegia_compatibility_score",
    ascending=False,
).reset_index(drop=True)

best_profile["phenotype"] = np.arange(
    1,
    len(best_profile) + 1,
)

raw_to_ordered = dict(
    zip(
        best_profile["cluster_raw"],
        best_profile["phenotype"],
    )
)

best_three_data["phenotype_3class"] = best_three_data["cluster_raw"].map(raw_to_ordered)

best_three_data["candidate_vasoplegic_phenotype_3class"] = best_three_data[
    "phenotype_3class"
].eq(1)

print("\n" + "=" * 80)
print("BEST 3-CLUSTER PHYSIOLOGICAL PROFILE")
print("=" * 80)

display(
    best_profile[
        [
            "phenotype",
            "cluster_raw",
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
# 15. MERGE BEST 3-CLASS PHENOTYPES
# ------------------------------------------------------------

analysis_three = analysis.merge(
    best_three_data[
        [
            "patientid",
            "phenotype_3class",
            "cluster_raw",
            "cluster_probability",
            "uncertain_assignment",
            "candidate_vasoplegic_phenotype_3class",
        ]
    ],
    on="patientid",
    how="left",
)

# ------------------------------------------------------------
# 16. LONGITUDINAL VALIDATION VARIABLES
# ------------------------------------------------------------

VALIDATION_VARIABLES = [
    "dbp_median_h1_h6_weighted_validation",
    "map_median_h1_h6_weighted_validation",
    "pulse_pressure_median_h1_h6_validation",
    "norepi_ug_kg_min_median_h1_h6",
    "norepi_ug_kg_min_max_h1_h6",
    "norepi_ug_kg_min_auc_h1_h6",
    "norepi_ug_kg_min_slope_h1_h6",
    "proportion_hours_dbp_lt_50_h1_h6_validation",
    "proportion_hours_dbp_lt_45_h1_h6_validation",
    "proportion_hours_map_lt_65_h1_h6_validation",
    "lactate_baseline",
    "norepi_ug_kg_min_baseline_winsor",
    "diastolic_shock_index",
    "shock_index",
    "modified_shock_index",
]

available_validation_variables = [
    variable for variable in VALIDATION_VARIABLES if variable in analysis_three.columns
]

# ------------------------------------------------------------
# 17. VALIDATION SUMMARY
# ------------------------------------------------------------

validation_rows = []

for variable in available_validation_variables:

    for phenotype, group in analysis_three.dropna(subset=["phenotype_3class"]).groupby(
        "phenotype_3class"
    ):

        values = pd.to_numeric(
            group[variable],
            errors="coerce",
        ).dropna()

        validation_rows.append(
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

validation_summary = pd.DataFrame(validation_rows)

# ------------------------------------------------------------
# 18. GLOBAL VALIDATION TESTS
# ------------------------------------------------------------

global_test_rows = []

for variable in available_validation_variables:

    groups = []

    for phenotype in sorted(analysis_three["phenotype_3class"].dropna().unique()):

        values = pd.to_numeric(
            analysis_three.loc[
                analysis_three["phenotype_3class"].eq(phenotype),
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

print("\n" + "=" * 80)
print("LONGITUDINAL VALIDATION OF BEST 3-CLUSTER MODEL")
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
# 19. PLOT EACH 3-CLUSTER SOLUTION
# ------------------------------------------------------------

for specification in MODEL_SPECIFICATIONS:

    if specification["n_clusters"] != 3:
        continue

    model_name = specification["model_name"]

    labels = fitted_models[model_name]["labels"]

    plot_data = phenotype_data[
        [
            "dbp_baseline",
            "pulse_pressure_baseline",
            "heart_rate_baseline",
        ]
    ].copy()

    plot_data["cluster"] = labels

    plt.figure(figsize=(9, 7))

    for cluster in sorted(plot_data["cluster"].unique()):

        subset = plot_data.loc[plot_data["cluster"].eq(cluster)]

        plt.scatter(
            subset["dbp_baseline"],
            subset["pulse_pressure_baseline"],
            s=24,
            alpha=0.45,
            label=(f"Cluster {cluster}"),
        )

    plt.xlabel("Baseline diastolic arterial pressure (mmHg)")

    plt.ylabel("Baseline pulse pressure (mmHg)")

    plt.title(f"Three-cluster macrohemodynamic model: {model_name}")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"step06_{model_name}_scatterplot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

# ------------------------------------------------------------
# 20. PLOT MODEL COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plot_comparison = model_comparison.sort_values("bic")

plt.bar(
    plot_comparison["model_name"],
    plot_comparison["bic"],
)

plt.xticks(
    rotation=30,
    ha="right",
)

plt.ylabel("Bayesian information criterion")

plt.title("Comparison of macrohemodynamic GMM specifications")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "step06_gmm_covariance_bic_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

plt.figure(figsize=(10, 6))

plot_stability = model_comparison.sort_values(
    "median_bootstrap_ari",
    ascending=False,
)

plt.bar(
    plot_stability["model_name"],
    plot_stability["median_bootstrap_ari"],
)

plt.xticks(
    rotation=30,
    ha="right",
)

plt.ylabel("Median bootstrap adjusted Rand index")

plt.title("Bootstrap stability of macrohemodynamic models")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "step06_gmm_covariance_stability_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# ------------------------------------------------------------
# 21. SAVE BEST MODEL AND TRANSPORT PARAMETERS
# ------------------------------------------------------------

joblib.dump(
    scaler,
    MODEL_DIR / "step06_macrohemodynamic_scaler.joblib",
)

joblib.dump(
    best_three_model,
    MODEL_DIR / "step06_best_three_cluster_gmm.joblib",
)

transport_parameters = {
    "model_name": best_three_model_name,
    "n_clusters": 3,
    "covariance_type": best_three_model.covariance_type,
    "phenotype_variables": PHENOTYPE_VARIABLES,
    "winsor_limits": winsor_limits,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "raw_to_ordered_mapping": {
        str(key): int(value) for key, value in raw_to_ordered.items()
    },
    "candidate_vasoplegic_phenotype": 1,
    "selection_status": selection_status,
    "norepinephrine_used_for_clustering": False,
    "lactate_used_for_clustering": False,
    "phenotype_window": "Median DBP, pulse pressure, and heart rate from 0 to 1 hour after time zero",
}

with open(
    MODEL_DIR / "step06_three_cluster_transport_parameters.json",
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
# 22. SAVE OUTPUTS
# ------------------------------------------------------------

model_comparison.to_csv(
    TABLE_DIR / "step06_covariance_model_comparison.csv",
    index=False,
)

all_profiles.to_csv(
    TABLE_DIR / "step06_all_covariance_cluster_profiles.csv",
    index=False,
)

physiological_screen.to_csv(
    TABLE_DIR / "step06_three_cluster_physiological_screen.csv",
    index=False,
)

three_model_ranking.to_csv(
    TABLE_DIR / "step06_three_cluster_model_ranking.csv",
    index=False,
)

best_profile.to_csv(
    TABLE_DIR / "step06_best_three_cluster_profile.csv",
    index=False,
)

validation_summary.to_csv(
    TABLE_DIR / "step06_best_three_cluster_longitudinal_validation.csv",
    index=False,
)

global_tests.to_csv(
    TABLE_DIR / "step06_best_three_cluster_global_tests.csv",
    index=False,
)

best_three_data.to_csv(
    TABLE_DIR / "step06_three_cluster_assignments.csv",
    index=False,
)

analysis_three.to_parquet(
    DERIVED_DIR
    / "step06_analysis_dataset_with_three_macrohemodynamic_phenotypes.parquet",
    index=False,
)

analysis_three.to_csv(
    TABLE_DIR / "step06_analysis_dataset_with_three_macrohemodynamic_phenotypes.csv",
    index=False,
)

run_summary = {
    "input_n": int(len(analysis)),
    "phenotype_derivation_n": int(len(phenotype_data)),
    "best_three_model_name": best_three_model_name,
    "best_three_covariance_type": best_three_model.covariance_type,
    "selection_status": selection_status,
    "statistically_eligible": bool(
        three_model_ranking.loc[
            three_model_ranking["model_name"].eq(best_three_model_name),
            "statistically_eligible",
        ].iloc[0]
    ),
    "phenotype_variables": PHENOTYPE_VARIABLES,
    "candidate_vasoplegic_phenotype": 1,
}

with open(
    LOG_DIR / "step06_three_cluster_model_summary.json",
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
# 23. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 06 COMPLETE")
print("=" * 80)

print(
    "\nBest 3-cluster model:",
    best_three_model_name,
)

print(
    "Covariance type:",
    best_three_model.covariance_type,
)

print(
    "Selection status:",
    selection_status,
)

print("\nNorepinephrine was NOT used for clustering.")

print("Lactate was NOT used for clustering.")

print(
    "\nMain output:\n"
    f"{DERIVED_DIR / 'step06_analysis_dataset_with_three_macrohemodynamic_phenotypes.parquet'}"
)
