"""Source: original notebook cell 60 (AUC and early longitudinal input only). Run through run.py."""

from common import artifact, display, phase_dir, run_stamp

# ============================================================
# CÓDIGO 13C
# AUC DA NEE EM 6 HORAS
#
# Análise fisiológica:
#
# 1. AUC absoluta da NEE entre 0 e 6 h
#       soma da NEE média em seis intervalos de uma hora
#       unidade: µg/kg/min × h
#
# 2. AUC ajustada ao basal
#       AUC observada - (NEE basal × 6)
#
#       Valor negativo:
#           carga vasopressora abaixo da carga esperada
#           caso a NEE basal permanecesse constante.
# Não relê inputevents ou chartevents.
# ============================================================

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ------------------------------------------------------------

PROJECT_DIR = phase_dir("mimic_iv")

COHORT_DIR = PROJECT_DIR / "cohorts"
TABLE_DIR = PROJECT_DIR / "tables"
FIGURE_DIR = PROJECT_DIR / "figures"
QC_DIR = PROJECT_DIR / "qc"
DOCUMENT_DIR = PROJECT_DIR / "documents"

for directory in [COHORT_DIR, TABLE_DIR, FIGURE_DIR, QC_DIR, DOCUMENT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. Localizar arquivos
# ------------------------------------------------------------


def latest_file(pattern):
    return artifact(PROJECT_DIR, pattern)


HOURLY_PATH = latest_file("methylene_blue_hourly_NEE_*.csv.gz")

AUDIT_PATH = latest_file("methylene_blue_followup_end_audit_*.csv")

print("=" * 100)
print("CÓDIGO 13C — AUC 6 H E BASE LONGITUDINAL")
print("=" * 100)

print("\nBase horária:")
print(HOURLY_PATH)

print("\nAuditoria de mortalidade:")
print(AUDIT_PATH)

# ------------------------------------------------------------
# 3. Carregar dados
# ------------------------------------------------------------

hourly = pd.read_csv(HOURLY_PATH, compression="gzip", low_memory=False)

audit = pd.read_csv(AUDIT_PATH, low_memory=False)

for dataframe in [hourly, audit]:
    for column in ["subject_id", "hadm_id", "stay_id"]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column], errors="coerce"
            ).astype("Int64")

PHENOTYPE_ORDER = [
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
    "Preserved diastolic pressure",
]

hourly = hourly[hourly["hemodynamic_phenotype"].isin(PHENOTYPE_ORDER)].copy()

hourly["hemodynamic_phenotype"] = pd.Categorical(
    hourly["hemodynamic_phenotype"], categories=PHENOTYPE_ORDER, ordered=True
)

audit["hours_mb_to_death"] = pd.to_numeric(audit["hours_mb_to_death"], errors="coerce")

audit["death_within_6h"] = audit["hours_mb_to_death"].notna() & audit[
    "hours_mb_to_death"
].between(0, 6, inclusive="both")

# ------------------------------------------------------------
# 4. Construir base individual para AUC
# ------------------------------------------------------------

baseline = hourly[hourly["hour_index"].eq(-1)][
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "hemodynamic_phenotype",
        "nee_median",
        "minutes_in_interval",
    ]
].rename(
    columns={"nee_median": "nee_baseline", "minutes_in_interval": "baseline_minutes"}
)

post_6h = hourly[hourly["hour_index"].between(0, 5, inclusive="both")].copy()

# Para a AUC, usamos a média minuto a minuto dentro de cada hora.
post_6h["hourly_auc_contribution"] = (
    post_6h["nee_mean"] * post_6h["minutes_in_interval"] / 60
)

auc_summary = post_6h.groupby("subject_id", as_index=False).agg(
    auc_nee_0_6h=("hourly_auc_contribution", "sum"),
    observed_post_mb_hours=("minutes_in_interval", lambda values: values.sum() / 60),
    n_post_mb_intervals=("hour_index", "nunique"),
    post_mb_minutes=("minutes_in_interval", "sum"),
    minutes_with_vasopressor_0_6h=("minutes_with_vasopressor", "sum"),
)

mortality = audit[
    ["subject_id", "death_within_6h", "hours_mb_to_death"]
].drop_duplicates(subset=["subject_id"])

auc_individual = baseline.merge(
    auc_summary, on="subject_id", how="left", validate="one_to_one"
).merge(mortality, on="subject_id", how="left", validate="one_to_one")

auc_individual["death_within_6h"] = (
    auc_individual["death_within_6h"].fillna(False).astype(bool)
)

auc_individual["alive_at_6h"] = ~auc_individual["death_within_6h"]

auc_individual["complete_baseline"] = auc_individual["baseline_minutes"] >= 60

auc_individual["complete_6h_followup"] = auc_individual["n_post_mb_intervals"].eq(
    6
) & auc_individual["post_mb_minutes"].ge(360)

auc_individual["eligible_auc_6h"] = (
    auc_individual["alive_at_6h"]
    & auc_individual["complete_baseline"]
    & auc_individual["complete_6h_followup"]
    & auc_individual["nee_baseline"].notna()
    & auc_individual["auc_nee_0_6h"].notna()
)

# Carga esperada se a dose basal permanecesse constante por 6 h
auc_individual["expected_auc_from_baseline"] = auc_individual["nee_baseline"] * 6

auc_individual["auc_adjusted_for_baseline"] = (
    auc_individual["auc_nee_0_6h"] - auc_individual["expected_auc_from_baseline"]
)

auc_individual["auc_ratio_to_baseline"] = (
    auc_individual["auc_nee_0_6h"] / auc_individual["expected_auc_from_baseline"]
)

auc_individual["mean_nee_0_6h"] = auc_individual["auc_nee_0_6h"] / 6

auc_analysis = auc_individual[auc_individual["eligible_auc_6h"]].copy()

# ------------------------------------------------------------
# 5. Resumo da AUC por fenótipo
# ------------------------------------------------------------


def q25(values):
    return values.quantile(0.25)


def q75(values):
    return values.quantile(0.75)


auc_by_phenotype = auc_analysis.groupby(
    "hemodynamic_phenotype", observed=False, as_index=False
).agg(
    n_eligible=("subject_id", "nunique"),
    auc_6h_median=("auc_nee_0_6h", "median"),
    auc_6h_p25=("auc_nee_0_6h", q25),
    auc_6h_p75=("auc_nee_0_6h", q75),
    adjusted_auc_median=("auc_adjusted_for_baseline", "median"),
    adjusted_auc_p25=("auc_adjusted_for_baseline", q25),
    adjusted_auc_p75=("auc_adjusted_for_baseline", q75),
    mean_nee_6h_median=("mean_nee_0_6h", "median"),
    mean_nee_6h_p25=("mean_nee_0_6h", q25),
    mean_nee_6h_p75=("mean_nee_0_6h", q75),
)

# ------------------------------------------------------------
# 6. Testes globais da AUC
# ------------------------------------------------------------


def kruskal_by_phenotype(dataframe, variable):

    groups = [
        dataframe.loc[
            dataframe["hemodynamic_phenotype"].eq(phenotype), variable
        ].dropna()
        for phenotype in PHENOTYPE_ORDER
    ]

    if not all(len(group) > 0 for group in groups):
        return np.nan

    return stats.kruskal(*groups).pvalue


auc_absolute_p = kruskal_by_phenotype(auc_analysis, "auc_nee_0_6h")

auc_adjusted_p = kruskal_by_phenotype(auc_analysis, "auc_adjusted_for_baseline")

auc_ratio_p = kruskal_by_phenotype(auc_analysis, "auc_ratio_to_baseline")

# ------------------------------------------------------------
# 7. Comparações par a par com correção de Holm
# ------------------------------------------------------------

pairwise_rows = []

phenotype_pairs = [
    ("Vasoplegia-compatible", "Mixed/low-pulsatility"),
    ("Vasoplegia-compatible", "Preserved diastolic pressure"),
    ("Mixed/low-pulsatility", "Preserved diastolic pressure"),
]

for phenotype_1, phenotype_2 in phenotype_pairs:

    values_1 = auc_analysis.loc[
        auc_analysis["hemodynamic_phenotype"].eq(phenotype_1),
        "auc_adjusted_for_baseline",
    ].dropna()

    values_2 = auc_analysis.loc[
        auc_analysis["hemodynamic_phenotype"].eq(phenotype_2),
        "auc_adjusted_for_baseline",
    ].dropna()

    statistic, raw_p = stats.mannwhitneyu(values_1, values_2, alternative="two-sided")

    pairwise_rows.append(
        {
            "phenotype_1": phenotype_1,
            "phenotype_2": phenotype_2,
            "n_1": len(values_1),
            "n_2": len(values_2),
            "mann_whitney_u": statistic,
            "raw_p_value": raw_p,
        }
    )

pairwise_tests = pd.DataFrame(pairwise_rows)

# Correção de Holm
pairwise_tests = pairwise_tests.sort_values("raw_p_value").reset_index(drop=True)

m = len(pairwise_tests)

holm_values = []

previous_adjusted = 0

for index, row in pairwise_tests.iterrows():

    adjusted = min(1, row["raw_p_value"] * (m - index))

    adjusted = max(adjusted, previous_adjusted)

    holm_values.append(adjusted)

    previous_adjusted = adjusted

pairwise_tests["holm_adjusted_p"] = holm_values

# ------------------------------------------------------------
# 8. Base para modelo longitudinal misto
#
# Baseline é tempo 0.
# Intervalo 0–1 h é representado como tempo 1.
# ...
# Intervalo 5–6 h é representado como tempo 6.
# ------------------------------------------------------------

longitudinal_model = hourly[
    hourly["hour_index"].between(-1, 5, inclusive="both")
].copy()

time_mapping = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

longitudinal_model["time_hours"] = (
    longitudinal_model["hour_index"].map(time_mapping).astype(float)
)

longitudinal_model["time_hours_squared"] = longitudinal_model["time_hours"] ** 2

longitudinal_model["log1p_nee"] = np.log1p(longitudinal_model["nee_median"])

longitudinal_model["phenotype_model"] = longitudinal_model[
    "hemodynamic_phenotype"
].astype(str)

# Referência: Vasoplegia-compatible
longitudinal_model["phenotype_model"] = pd.Categorical(
    longitudinal_model["phenotype_model"], categories=PHENOTYPE_ORDER, ordered=True
)

longitudinal_model = longitudinal_model.dropna(
    subset=["subject_id", "log1p_nee", "time_hours", "phenotype_model"]
).copy()
# ------------------------------------------------------------
# 13. Controle de qualidade
# ------------------------------------------------------------

qc_summary = pd.DataFrame(
    {
        "metric": [
            "Patients in cohort",
            "Deaths within 6 h",
            "Patients alive at 6 h",
            "Patients eligible for complete 6 h AUC",
            "Patients excluded from AUC because of death",
            "Patients excluded from AUC for incomplete observation",
            "Global p value for absolute AUC",
            "Global p value for baseline-adjusted AUC",
            "Global p value for AUC-to-baseline ratio",
            "Maximum hourly NEE in 0–6 h",
            "Patients with hourly NEE >2",
            "Patients with hourly NEE >3",
            "Patients with hourly NEE >4",
        ],
        "value": [
            len(auc_individual),
            int(auc_individual["death_within_6h"].sum()),
            int(auc_individual["alive_at_6h"].sum()),
            int(auc_individual["eligible_auc_6h"].sum()),
            int(auc_individual["death_within_6h"].sum()),
            int(
                (
                    auc_individual["alive_at_6h"]
                    & ~auc_individual["complete_6h_followup"]
                ).sum()
            ),
            auc_absolute_p,
            auc_adjusted_p,
            auc_ratio_p,
            post_6h["nee_max"].max(),
            post_6h.loc[post_6h["nee_max"].gt(2), "subject_id"].nunique(),
            post_6h.loc[post_6h["nee_max"].gt(3), "subject_id"].nunique(),
            post_6h.loc[post_6h["nee_max"].gt(4), "subject_id"].nunique(),
        ],
    }
)

timestamp = run_stamp()
COHORT_DIR.mkdir(parents=True, exist_ok=True)
auc_individual_path = (
    COHORT_DIR / f"methylene_blue_6h_NEE_AUC_individual_{timestamp}.csv.gz"
)

auc_summary_path = TABLE_DIR / f"methylene_blue_6h_NEE_AUC_by_phenotype_{timestamp}.csv"

pairwise_path = TABLE_DIR / f"methylene_blue_6h_NEE_AUC_pairwise_tests_{timestamp}.csv"

model_data_path = (
    COHORT_DIR / f"methylene_blue_6h_longitudinal_model_data_{timestamp}.csv.gz"
)

qc_path = QC_DIR / f"methylene_blue_code13C_AUC_mixed_model_qc_{timestamp}.csv"

auc_individual.to_csv(auc_individual_path, index=False, compression="gzip")

auc_by_phenotype.to_csv(auc_summary_path, index=False)

pairwise_tests.to_csv(pairwise_path, index=False)

longitudinal_model.to_csv(model_data_path, index=False, compression="gzip")

qc_summary.to_csv(qc_path, index=False)

display(auc_by_phenotype)
display(qc_summary)
