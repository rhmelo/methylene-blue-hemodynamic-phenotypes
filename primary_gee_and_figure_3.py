"""Source: revision notebook cells 81–83 and 85. Run through run.py."""

from common import artifact, display, phase_dir

import pandas as pd

ROOT = phase_dir("mimic_iv")
long = pd.read_csv(artifact(ROOT, "methylene_blue_6h_longitudinal_model_data_*.csv.gz"))
# ============================================================
# CÓDIGO 16B — CONTINUAÇÃO CORRIGIDA
# AUDITORIA DO SEGUIMENTO LONGITUDINAL
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# Coluna temporal correta
# ------------------------------------------------------------

TIME_COL = "interval_label"

print("\nColuna temporal utilizada:")
print(TIME_COL)

# Ordem clínica correta dos intervalos
TIME_ORDER = [
    "Baseline (-1 to 0 h)",
    "0 to <1 h",
    "1 to <2 h",
    "2 to <3 h",
    "3 to <4 h",
    "4 to <5 h",
    "5 to <6 h",
]

long[TIME_COL] = pd.Categorical(
    long[TIME_COL],
    categories=TIME_ORDER,
    ordered=True,
)

# ============================================================
# 1. PACIENTES CONTRIBUINDO EM CADA MOMENTO
# ============================================================

print("\n" + "=" * 70)
print("PATIENTS CONTRIBUTING AT EACH TIMEPOINT")
print("=" * 70)

patients_by_time = (
    long.groupby(
        [TIME_COL, "hemodynamic_phenotype"],
        observed=False,
    )["subject_id"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(TIME_ORDER)
)

patients_by_time["Overall"] = patients_by_time.sum(axis=1)

display(patients_by_time)

# ============================================================
# 2. NÚMERO DE MOMENTOS POR PACIENTE
# ============================================================

timepoints_per_patient = long.groupby("subject_id")[TIME_COL].nunique()

expected_timepoints = len(TIME_ORDER)

print("\nNúmero esperado de momentos:")
print(expected_timepoints)

print("\nPacientes com todos os momentos:")
print(int(timepoints_per_patient.eq(expected_timepoints).sum()))

print("\nPacientes com seguimento incompleto:")
print(int(timepoints_per_patient.lt(expected_timepoints).sum()))

print("\nDistribuição do número de momentos por paciente:")

timepoint_distribution = (
    timepoints_per_patient.value_counts()
    .sort_index()
    .rename_axis("number_of_timepoints")
    .reset_index(name="patients")
)

display(timepoint_distribution)

# ============================================================
# 3. PACIENTES COMPLETOS POR FENÓTIPO
# ============================================================

complete_ids = timepoints_per_patient[
    timepoints_per_patient.eq(expected_timepoints)
].index

complete_by_phenotype = (
    long[long["subject_id"].isin(complete_ids)]
    .drop_duplicates("subject_id")["hemodynamic_phenotype"]
    .value_counts()
    .rename_axis("hemodynamic_phenotype")
    .reset_index(name="complete_patients")
)

print("\nPacientes completos por fenótipo:")
display(complete_by_phenotype)

# ============================================================
# 4. AUDITORIA DE NEE = 0
# ============================================================

print("\n" + "=" * 70)
print("NEE ZERO AUDIT")
print("=" * 70)

zero_audit = pd.DataFrame(
    {
        "metric": [
            "Rows with NEE median = 0",
            "Patients with at least one NEE median = 0",
            "Rows flagged as NEE zero entire hour",
            "Patients with at least one full hour at NEE = 0",
            "Rows without any vasopressor in the hour",
            "Patients without any vasopressor in at least one hour",
        ],
        "value": [
            int(long["nee_median"].eq(0).sum()),
            int(long.loc[long["nee_median"].eq(0), "subject_id"].nunique()),
            int(long["nee_zero_entire_hour"].eq(True).sum()),
            int(
                long.loc[
                    long["nee_zero_entire_hour"].eq(True),
                    "subject_id",
                ].nunique()
            ),
            int(long["any_vasopressor_in_hour"].eq(False).sum()),
            int(
                long.loc[
                    long["any_vasopressor_in_hour"].eq(False),
                    "subject_id",
                ].nunique()
            ),
        ],
    }
)

display(zero_audit)

# ============================================================
# 5. VERIFICAR SE ZERO FOI MANTIDO NO log1p
# ============================================================

zero_log_check = long.loc[
    long["nee_median"].eq(0),
    [
        "subject_id",
        TIME_COL,
        "nee_median",
        "log1p_nee",
        "any_vasopressor_in_hour",
        "nee_zero_entire_hour",
    ],
].copy()

print("\nExemplos de horas com NEE = 0:")
display(zero_log_check.head(20))

# 7. SALVAR AUDITORIA
# ============================================================

OUT = ROOT / "revision_analyses"
OUT.mkdir(parents=True, exist_ok=True)

patients_by_time.to_csv(OUT / "audit_patients_by_timepoint.csv")

timepoint_distribution.to_csv(
    OUT / "audit_timepoints_per_patient.csv",
    index=False,
)

complete_by_phenotype.to_csv(
    OUT / "audit_complete_patients_by_phenotype.csv",
    index=False,
)

zero_audit.to_csv(
    OUT / "audit_NEE_zero_summary.csv",
    index=False,
)

zero_log_check.to_csv(
    OUT / "audit_NEE_zero_records.csv",
    index=False,
)

print("\n" + "=" * 70)
print("CÓDIGO 16B CONCLUÍDO")
print("=" * 70)

print("\nArquivos salvos em:")
print(OUT)
# ============================================================
# CÓDIGO 16B.2
# AUDITORIA DE OBSERVAÇÕES APÓS O ÓBITO
# ============================================================

import numpy as np
import pandas as pd

ROOT = phase_dir("mimic_iv")

OUT = ROOT / "revision_analyses"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Carregar os pacientes com informação de óbito
# ------------------------------------------------------------

SHORT_PATH = artifact(
    phase_dir("mimic_iv"), "methylene_blue_short_followup_patients_*.csv"
)

short = pd.read_csv(SHORT_PATH)

for col in ["mb_time_preliminary", "deathtime"]:
    short[col] = pd.to_datetime(short[col], errors="coerce")

death_info = short.loc[
    short["deathtime"].notna(), ["subject_id", "deathtime", "hours_mb_to_death"]
].drop_duplicates("subject_id")

# ------------------------------------------------------------
# 2. Preparar os limites de cada intervalo
# ------------------------------------------------------------

interval_bounds = {
    "Baseline (-1 to 0 h)": (-1.0, 0.0),
    "0 to <1 h": (0.0, 1.0),
    "1 to <2 h": (1.0, 2.0),
    "2 to <3 h": (2.0, 3.0),
    "3 to <4 h": (3.0, 4.0),
    "4 to <5 h": (4.0, 5.0),
    "5 to <6 h": (5.0, 6.0),
}

long_audit = long.merge(
    death_info,
    on="subject_id",
    how="left",
    validate="many_to_one",
)

long_audit["interval_start_h"] = (
    long_audit["interval_label"].astype(str).map(lambda x: interval_bounds[x][0])
)

long_audit["interval_end_h"] = (
    long_audit["interval_label"].astype(str).map(lambda x: interval_bounds[x][1])
)

# Intervalo inteiro posterior ao óbito
long_audit["entire_interval_after_death"] = long_audit["hours_mb_to_death"].notna() & (
    long_audit["interval_start_h"] >= long_audit["hours_mb_to_death"]
)

# Óbito ocorreu durante o intervalo
long_audit["death_during_interval"] = (
    long_audit["hours_mb_to_death"].notna()
    & (long_audit["hours_mb_to_death"] > long_audit["interval_start_h"])
    & (long_audit["hours_mb_to_death"] <= long_audit["interval_end_h"])
)

# ------------------------------------------------------------
# 3. Resumo das observações pós-óbito
# ------------------------------------------------------------

post_death_rows = long_audit[long_audit["entire_interval_after_death"]].copy()

audit_summary = pd.DataFrame(
    {
        "metric": [
            "Patients with death information",
            "Patients dying within 6 h",
            "Rows entirely after death",
            "Patients contributing rows entirely after death",
            "Post-death rows with NEE = 0",
            "Post-death rows flagged zero entire hour",
            "Post-death rows with NEE > 0",
        ],
        "value": [
            death_info["subject_id"].nunique(),
            death_info.loc[
                death_info["hours_mb_to_death"].le(6), "subject_id"
            ].nunique(),
            len(post_death_rows),
            post_death_rows["subject_id"].nunique(),
            post_death_rows["nee_median"].eq(0).sum(),
            post_death_rows["nee_zero_entire_hour"].eq(True).sum(),
            post_death_rows["nee_median"].gt(0).sum(),
        ],
    }
)

print("=" * 70)
print("POST-DEATH OBSERVATION AUDIT")
print("=" * 70)
display(audit_summary)

print("\nObservações inteiramente posteriores ao óbito:")
display(
    post_death_rows[
        [
            "subject_id",
            "hemodynamic_phenotype",
            "interval_label",
            "hours_mb_to_death",
            "nee_median",
            "log1p_nee",
            "any_vasopressor_in_hour",
            "nee_zero_entire_hour",
        ]
    ].sort_values(["subject_id", "interval_label"])
)

# ------------------------------------------------------------
# 4. Distribuição por fenótipo
# ------------------------------------------------------------

post_death_by_phenotype = (
    post_death_rows.groupby("hemodynamic_phenotype", observed=True)
    .agg(
        patients=("subject_id", "nunique"),
        rows=("subject_id", "size"),
        zero_rows=("nee_median", lambda x: x.eq(0).sum()),
    )
    .reset_index()
)

print("\nObservações pós-óbito por fenótipo:")
display(post_death_by_phenotype)

# ------------------------------------------------------------
# 5. Criar dataset corrigido
# ------------------------------------------------------------

long_no_post_death = long_audit.loc[~long_audit["entire_interval_after_death"]].copy()

# Remover colunas auxiliares antes de modelar posteriormente
helper_columns = [
    "deathtime",
    "hours_mb_to_death",
    "interval_start_h",
    "interval_end_h",
    "entire_interval_after_death",
    "death_during_interval",
]

long_no_post_death_model = long_no_post_death.drop(
    columns=helper_columns,
    errors="ignore",
)

# ------------------------------------------------------------
# 6. Número de pacientes por horário após correção
# ------------------------------------------------------------

corrected_patients_by_time = (
    long_no_post_death_model.groupby(
        ["interval_label", "hemodynamic_phenotype"],
        observed=True,
    )["subject_id"]
    .nunique()
    .unstack(fill_value=0)
)

corrected_patients_by_time["Overall"] = corrected_patients_by_time.sum(axis=1)

print("\nPacientes por intervalo após excluir observações pós-óbito:")
display(corrected_patients_by_time)

# ------------------------------------------------------------
# 7. Salvar resultados
# ------------------------------------------------------------

audit_summary.to_csv(
    OUT / "audit_post_death_observations_summary.csv",
    index=False,
)

post_death_rows.to_csv(
    OUT / "audit_post_death_observation_records.csv",
    index=False,
)

post_death_by_phenotype.to_csv(
    OUT / "audit_post_death_by_phenotype.csv",
    index=False,
)

corrected_patients_by_time.to_csv(OUT / "audit_patients_by_time_without_post_death.csv")

long_no_post_death_model.to_csv(
    OUT / "methylene_blue_longitudinal_data_without_post_death.csv.gz",
    index=False,
    compression="gzip",
)

print("\nCódigo 16B.2 concluído.")
print("Outputs salvos em:")
print(OUT)
# ============================================================
# CÓDIGO 16C
# GEE ORIGINAL VERSUS GEE SEM OBSERVAÇÕES PÓS-ÓBITO
# ============================================================

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

ROOT = phase_dir("mimic_iv")

OUT = ROOT / "revision_analyses"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Verificações
# ------------------------------------------------------------

required_objects = [
    "long",
    "long_audit",
]

for obj in required_objects:
    if obj not in globals():
        raise NameError(
            f"O objeto '{obj}' não foi encontrado. "
            "Rode primeiro os códigos 16B e 16B.2."
        )

required_columns = [
    "subject_id",
    "hemodynamic_phenotype",
    "hour_index",
    "nee_median",
    "log1p_nee",
]

missing_cols = [col for col in required_columns if col not in long.columns]

if missing_cols:
    raise KeyError(f"Colunas ausentes no dataset longitudinal: {missing_cols}")

# ------------------------------------------------------------
# 2. Padronizar os fenótipos
# ------------------------------------------------------------

PHENOTYPE_ORDER = [
    "Preserved diastolic pressure",
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
]


def prepare_model_data(df):
    result = df.copy()

    result["hemodynamic_phenotype"] = pd.Categorical(
        result["hemodynamic_phenotype"],
        categories=PHENOTYPE_ORDER,
        ordered=False,
    )

    # Tempo categórico:
    # -1 = baseline
    #  0 = 0 a <1 h
    # ...
    #  5 = 5 a <6 h
    result["hour_index"] = pd.Categorical(
        result["hour_index"],
        categories=[-1, 0, 1, 2, 3, 4, 5],
        ordered=True,
    )

    # Recalcular explicitamente log(1 + NEE)
    result["log1p_nee"] = np.log1p(
        pd.to_numeric(
            result["nee_median"],
            errors="coerce",
        )
    )

    result = result.dropna(
        subset=[
            "subject_id",
            "hemodynamic_phenotype",
            "hour_index",
            "log1p_nee",
        ]
    ).copy()

    return result


original_data = prepare_model_data(long)

corrected_data = prepare_model_data(
    long_audit.loc[~long_audit["entire_interval_after_death"]].copy()
)

print("=" * 72)
print("MODEL DATASETS")
print("=" * 72)

dataset_summary = pd.DataFrame(
    {
        "dataset": [
            "Original",
            "Without intervals entirely after death",
        ],
        "patients": [
            original_data["subject_id"].nunique(),
            corrected_data["subject_id"].nunique(),
        ],
        "observations": [
            len(original_data),
            len(corrected_data),
        ],
        "zero_NEE_observations": [
            original_data["nee_median"].eq(0).sum(),
            corrected_data["nee_median"].eq(0).sum(),
        ],
    }
)

display(dataset_summary)

# ------------------------------------------------------------
# 3. Ajustar o mesmo GEE nos dois datasets
# ------------------------------------------------------------

FORMULA = (
    "log1p_nee ~ "
    "C(hour_index, Treatment(reference=-1)) * "
    "C(hemodynamic_phenotype, "
    "Treatment(reference='Preserved diastolic pressure'))"
)


def fit_gee(data):
    model = smf.gee(
        formula=FORMULA,
        groups="subject_id",
        data=data,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    )

    return model.fit(
        maxiter=200,
        cov_type="robust",
    )


gee_original = fit_gee(original_data)
gee_corrected = fit_gee(corrected_data)

print("\n" + "=" * 72)
print("ORIGINAL GEE")
print("=" * 72)
print(gee_original.summary())

print("\n" + "=" * 72)
print("CORRECTED GEE")
print("=" * 72)
print(gee_corrected.summary())

# ------------------------------------------------------------
# 4. Teste global da interação fenótipo × tempo
# ------------------------------------------------------------


def global_interaction_test(result):
    parameter_names = list(result.params.index)

    interaction_idx = [
        i
        for i, name in enumerate(parameter_names)
        if ":" in name and "hour_index" in name and "hemodynamic_phenotype" in name
    ]

    if not interaction_idx:
        raise ValueError("Nenhum coeficiente de interação foi encontrado.")

    restriction = np.zeros((len(interaction_idx), len(parameter_names)))

    for row, col in enumerate(interaction_idx):
        restriction[row, col] = 1

    test = result.wald_test(
        restriction,
        scalar=True,
    )

    return {
        "n_interaction_parameters": len(interaction_idx),
        "wald_statistic": float(test.statistic),
        "df": len(interaction_idx),
        "p_value": float(test.pvalue),
    }


interaction_original = global_interaction_test(gee_original)

interaction_corrected = global_interaction_test(gee_corrected)

interaction_comparison = pd.DataFrame(
    [
        {
            "model": "Original",
            **interaction_original,
        },
        {
            "model": "Without intervals entirely after death",
            **interaction_corrected,
        },
    ]
)

print("\n" + "=" * 72)
print("GLOBAL PHENOTYPE-BY-TIME INTERACTION")
print("=" * 72)

display(interaction_comparison)

# ------------------------------------------------------------
# 5. Comparação dos coeficientes
# ------------------------------------------------------------

coef_original = pd.DataFrame(
    {
        "term": gee_original.params.index,
        "estimate_original": gee_original.params.values,
        "se_original": gee_original.bse.values,
        "p_original": gee_original.pvalues.values,
    }
)

coef_corrected = pd.DataFrame(
    {
        "term": gee_corrected.params.index,
        "estimate_corrected": gee_corrected.params.values,
        "se_corrected": gee_corrected.bse.values,
        "p_corrected": gee_corrected.pvalues.values,
    }
)

coefficient_comparison = coef_original.merge(
    coef_corrected,
    on="term",
    how="outer",
)

coefficient_comparison["absolute_change"] = (
    coefficient_comparison["estimate_corrected"]
    - coefficient_comparison["estimate_original"]
)

print("\n" + "=" * 72)
print("COEFFICIENT COMPARISON")
print("=" * 72)

display(coefficient_comparison)

# ------------------------------------------------------------
# 6. Pacientes por hora após a correção
# ------------------------------------------------------------

TIME_LABELS = {
    -1: "Baseline (-1 to 0 h)",
    0: "0 to <1 h",
    1: "1 to <2 h",
    2: "2 to <3 h",
    3: "3 to <4 h",
    4: "4 to <5 h",
    5: "5 to <6 h",
}

corrected_for_count = corrected_data.copy()

corrected_for_count["interval_label"] = (
    corrected_for_count["hour_index"].astype(int).map(TIME_LABELS)
)

patients_by_time_corrected = (
    corrected_for_count.groupby(
        [
            "interval_label",
            "hemodynamic_phenotype",
        ],
        observed=True,
    )["subject_id"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(list(TIME_LABELS.values()))
)

patients_by_time_corrected["Overall"] = patients_by_time_corrected.sum(axis=1)

print("\n" + "=" * 72)
print("PATIENTS CONTRIBUTING AFTER POST-DEATH EXCLUSION")
print("=" * 72)

display(patients_by_time_corrected)

# ------------------------------------------------------------
# 7. Salvar outputs
# ------------------------------------------------------------

dataset_summary.to_csv(
    OUT / "GEE_dataset_comparison_original_vs_no_post_death.csv",
    index=False,
)

interaction_comparison.to_csv(
    OUT / "GEE_global_interaction_original_vs_no_post_death.csv",
    index=False,
)

coefficient_comparison.to_csv(
    OUT / "GEE_coefficients_original_vs_no_post_death.csv",
    index=False,
)

patients_by_time_corrected.to_csv(OUT / "GEE_patients_by_time_without_post_death.csv")

with open(
    OUT / "GEE_original_model_summary.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(gee_original.summary().as_text())

with open(
    OUT / "GEE_without_post_death_model_summary.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(gee_corrected.summary().as_text())

corrected_data.to_csv(
    OUT / "GEE_model_data_without_post_death.csv.gz",
    index=False,
    compression="gzip",
)

print("\n" + "=" * 72)
print("CÓDIGO 16C CONCLUÍDO")
print("=" * 72)

print("\nOutputs salvos em:")
print(OUT)
# ============================================================
# 16E
# CONTRASTES CORRETOS + FIGURA 3 FINAL
# ============================================================

from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt

from patsy import build_design_matrices
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# ------------------------------------------------------------
# Objetos necessários
# ------------------------------------------------------------

result = gee_corrected

TIME_ORDER = [-1, 0, 1, 2, 3, 4, 5]

TIME_LABELS = {
    -1: "Baseline",
    0: "0–1",
    1: "1–2",
    2: "2–3",
    3: "3–4",
    4: "4–5",
    5: "5–6",
}

PHENOTYPES = [
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
    "Preserved diastolic pressure",
]

N_BY_TIME = (
    corrected_data.groupby("hour_index", observed=True)["subject_id"]
    .nunique()
    .reindex(TIME_ORDER, fill_value=0)
    .tolist()
)

OUT = ROOT / "revision_analyses"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Grade de predição
# ------------------------------------------------------------

newdata = pd.DataFrame(
    [
        {
            "hour_index": hour,
            "hemodynamic_phenotype": phenotype,
        }
        for phenotype in PHENOTYPES
        for hour in TIME_ORDER
    ]
)

# Garantir categorias idênticas às usadas no modelo
newdata["hour_index"] = pd.Categorical(
    newdata["hour_index"],
    categories=TIME_ORDER,
    ordered=True,
)

newdata["hemodynamic_phenotype"] = pd.Categorical(
    newdata["hemodynamic_phenotype"],
    categories=[
        "Preserved diastolic pressure",
        "Vasoplegia-compatible",
        "Mixed/low-pulsatility",
    ],
)

# Matriz de desenho correspondente ao modelo ajustado
design_info = result.model.data.design_info

X = np.asarray(
    build_design_matrices(
        [design_info],
        newdata,
        return_type="dataframe",
    )[0]
)

beta = np.asarray(result.params)
cov_beta = np.asarray(result.cov_params())

# ------------------------------------------------------------
# 2. EMMs na escala log(1 + NEE)
# ------------------------------------------------------------

eta = X @ beta

eta_var = np.einsum(
    "ij,jk,ik->i",
    X,
    cov_beta,
    X,
)

eta_se = np.sqrt(np.maximum(eta_var, 0))

z975 = norm.ppf(0.975)

newdata["estimate_log1p"] = eta
newdata["se_log1p"] = eta_se
newdata["lower_log1p"] = eta - z975 * eta_se
newdata["upper_log1p"] = eta + z975 * eta_se

# Voltar para a escala original
newdata["estimated_NEE"] = np.expm1(newdata["estimate_log1p"])

newdata["lower_NEE"] = np.maximum(
    0,
    np.expm1(newdata["lower_log1p"]),
)

newdata["upper_NEE"] = np.expm1(newdata["upper_log1p"])

emm_corrected = newdata.copy()

print("=" * 72)
print("CORRECTED ESTIMATED MARGINAL MEANS")
print("=" * 72)

display(
    emm_corrected[
        [
            "hour_index",
            "hemodynamic_phenotype",
            "estimate_log1p",
            "se_log1p",
            "estimated_NEE",
            "lower_NEE",
            "upper_NEE",
        ]
    ]
)

# ------------------------------------------------------------
# 3. Contrastes corretos usando Cov(beta)
# ------------------------------------------------------------

contrast_rows = []

for hour in TIME_ORDER:

    hour_rows = newdata.index[newdata["hour_index"].astype(int).eq(hour)].tolist()

    for i, j in combinations(hour_rows, 2):

        phenotype_i = str(newdata.loc[i, "hemodynamic_phenotype"])

        phenotype_j = str(newdata.loc[j, "hemodynamic_phenotype"])

        contrast_vector = X[i, :] - X[j, :]

        estimate = float(contrast_vector @ beta)

        variance = float(contrast_vector @ cov_beta @ contrast_vector.T)

        se = np.sqrt(max(variance, 0))

        z_value = estimate / se
        p_value = 2 * norm.sf(abs(z_value))

        lower = estimate - z975 * se
        upper = estimate + z975 * se

        contrast_rows.append(
            {
                "hour_index": hour,
                "interval": TIME_LABELS[hour],
                "comparison": (f"{phenotype_i} vs {phenotype_j}"),
                "difference_log1p": estimate,
                "SE": se,
                "CI_low_log1p": lower,
                "CI_high_log1p": upper,
                "z": z_value,
                "P_raw": p_value,
            }
        )

contrasts_corrected = pd.DataFrame(contrast_rows)

# Holm dentro de cada intervalo:
# três comparações por horário
contrasts_corrected["P_Holm_within_time"] = np.nan

for hour in TIME_ORDER:

    mask = contrasts_corrected["hour_index"].eq(hour)

    contrasts_corrected.loc[
        mask,
        "P_Holm_within_time",
    ] = multipletests(
        contrasts_corrected.loc[
            mask,
            "P_raw",
        ],
        method="holm",
    )[1]

print("\n" + "=" * 72)
print("CORRECTED PAIRWISE CONTRASTS")
print("=" * 72)

display(contrasts_corrected)

# ------------------------------------------------------------

emm_corrected.to_csv(OUT / "GEE_estimated_marginal_means.csv", index=False)
contrasts_corrected.to_csv(
    OUT / "GEE_pairwise_contrasts_within_hour_Holm.csv", index=False
)
AUC_PATH = artifact(ROOT, "methylene_blue_6h_NEE_AUC_individual_*.csv.gz")
print("\nArquivo de AUC utilizado:")
print(AUC_PATH)

auc_plot_data = pd.read_csv(AUC_PATH)

print("\nColunas disponíveis:")
print(auc_plot_data.columns.tolist())

display(auc_plot_data.head())

# ------------------------------------------------------------
# 5. Identificar as colunas
# ------------------------------------------------------------

phenotype_candidates = [
    "hemodynamic_phenotype",
    "phenotype",
    "phenotype_model",
]

auc_value_candidates = [
    "auc_adjusted_for_baseline",
    "baseline_adjusted_nee_auc",
    "baseline_adjusted_auc",
    "nee_auc_adjusted",
    "adjusted_nee_auc",
    "auc_adjusted",
    "baseline_adjusted_nee_auc_6h",
    "nee_auc_6h_baseline_adjusted",
]

auc_col = next(
    (col for col in auc_value_candidates if col in auc_plot_data.columns),
    None,
)

if auc_col is None:
    raise KeyError(
        "Não identifiquei a coluna da AUC ajustada. "
        f"Colunas disponíveis: {auc_plot_data.columns.tolist()}"
    )

print("\nColuna de AUC utilizada:")
print(auc_col)

phenotype_col = next(
    (col for col in phenotype_candidates if col in auc_plot_data.columns),
    None,
)

auc_col = next(
    (col for col in auc_value_candidates if col in auc_plot_data.columns),
    None,
)

# Busca alternativa por partes do nome
if auc_col is None:
    auc_matching = [
        col
        for col in auc_plot_data.columns
        if "auc" in col.lower()
        and ("adjust" in col.lower() or "baseline" in col.lower())
        and pd.api.types.is_numeric_dtype(auc_plot_data[col])
    ]

    if len(auc_matching) == 1:
        auc_col = auc_matching[0]

if phenotype_col is None:
    raise KeyError(
        "Não identifiquei a coluna de fenótipo. "
        f"Colunas: {auc_plot_data.columns.tolist()}"
    )

if auc_col is None:
    raise KeyError(
        "Não identifiquei automaticamente a coluna da AUC ajustada. "
        f"Colunas: {auc_plot_data.columns.tolist()}"
    )

print("\nColuna de fenótipo:")
print(phenotype_col)

print("\nColuna de AUC:")
print(auc_col)

# ------------------------------------------------------------
# 6. Padronizar nomes dos fenótipos
# ------------------------------------------------------------

phenotype_map = {
    "Preserved DBP": "Preserved diastolic pressure",
    "Preserved diastolic pressure": "Preserved diastolic pressure",
    "Vasoplegia-compatible": "Vasoplegia-compatible",
    "Mixed": "Mixed/low-pulsatility",
    "Mixed/low-pulsatility": "Mixed/low-pulsatility",
}

auc_plot_data[phenotype_col] = auc_plot_data[phenotype_col].replace(phenotype_map)

# ------------------------------------------------------------
# 7. Construir a Figura 3
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6.5),
    gridspec_kw={
        "width_ratios": [1.65, 1.0],
        "wspace": 0.30,
    },
)

ax1, ax2 = axes

TIME_ORDER = [-1, 0, 1, 2, 3, 4, 5]

TIME_LABELS = {
    -1: "Baseline",
    0: "0–1",
    1: "1–2",
    2: "2–3",
    3: "3–4",
    4: "4–5",
    5: "5–6",
}

PLOT_ORDER = [
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
    "Preserved diastolic pressure",
]

N_BY_TIME = (
    corrected_data.groupby("hour_index", observed=True)["subject_id"]
    .nunique()
    .reindex(TIME_ORDER, fill_value=0)
    .tolist()
)

x = np.arange(len(TIME_ORDER))

# ------------------------------------------------------------
# Painel A: estimativas marginais do GEE
# ------------------------------------------------------------

for phenotype in PLOT_ORDER:

    tmp = emm_corrected.loc[
        emm_corrected["hemodynamic_phenotype"].astype(str).eq(phenotype)
    ].copy()

    tmp["hour_numeric"] = tmp["hour_index"].astype(int)

    tmp = tmp.sort_values("hour_numeric")

    ax1.plot(
        x,
        tmp["estimated_NEE"].to_numpy(),
        marker="o",
        linewidth=2.2,
        markersize=5.5,
        label=phenotype,
    )

    ax1.fill_between(
        x,
        tmp["lower_NEE"].to_numpy(),
        tmp["upper_NEE"].to_numpy(),
        alpha=0.15,
    )

ax1.set_title(
    "A. Early vasopressor trajectory",
    loc="left",
    fontweight="bold",
)

ax1.set_ylabel("Estimated norepinephrine-equivalent dose\n" "(µg/kg/min)")

ax1.set_xlabel("Time relative to MB administration (hours)")

ax1.set_xticks(x)

ax1.set_xticklabels([TIME_LABELS[t] for t in TIME_ORDER])

ax1.grid(
    axis="y",
    alpha=0.25,
)

ax1.text(
    0.98,
    0.98,
    "Global phenotype × time interaction\n"
    f"GEE: P = {interaction_corrected['p_value']:.3f}",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=10.5,
)

# Número de pacientes em cada intervalo
for xpos, n_value in zip(
    x,
    N_BY_TIME,
):
    ax1.text(
        xpos,
        -0.15,
        f"n={n_value}",
        transform=ax1.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.5,
    )

ax1.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
    frameon=False,
)

# ------------------------------------------------------------
# Painel B: AUC original
# ------------------------------------------------------------

box_order = [
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
    "Preserved diastolic pressure",
]

box_data = [
    auc_plot_data.loc[
        auc_plot_data[phenotype_col].eq(group),
        auc_col,
    ]
    .dropna()
    .to_numpy()
    for group in box_order
]

ax2.boxplot(
    box_data,
    widths=0.55,
    showfliers=False,
)

rng = np.random.default_rng(2026)

for position, values in enumerate(
    box_data,
    start=1,
):
    jitter = rng.normal(
        position,
        0.045,
        len(values),
    )

    ax2.scatter(
        jitter,
        values,
        s=28,
        alpha=0.72,
    )

ax2.axhline(
    0,
    linestyle="--",
    linewidth=1,
)

ax2.set_title(
    "B. Six-hour cumulative vasopressor burden",
    loc="left",
    fontweight="bold",
)

ax2.set_ylabel("Baseline-adjusted NEE AUC\n" "(µg/kg/min × h)")

ax2.set_xticks([1, 2, 3])

ax2.set_xticklabels(
    [
        f"Vasoplegia-\ncompatible\n(n={len(box_data[0])})",
        f"Mixed/low-\npulsatility\n(n={len(box_data[1])})",
        f"Preserved\nDBP\n(n={len(box_data[2])})",
    ]
)

ax2.grid(
    axis="y",
    alpha=0.0,
)

# Preserve the archived panel B plotting population. Its annotation comes
# from the eligible complete-case AUC test, as in the submitted figure.
auc_qc = pd.read_csv(artifact(ROOT, "methylene_blue_code13C_AUC_mixed_model_qc_*.csv"))
eligible_auc_p = auc_qc.loc[
    auc_qc["metric"].eq("Global p value for baseline-adjusted AUC"), "value"
].item()

ax2.text(
    0.98,
    0.98,
    f"Global Kruskal–Wallis test\nP = {eligible_auc_p:.3f}",
    transform=ax2.transAxes,
    ha="right",
    va="top",
    fontsize=10.5,
)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.subplots_adjust(
    bottom=0.28,
    top=0.90,
    left=0.08,
    right=0.98,
)

# ------------------------------------------------------------
# 8. Salvar
# ------------------------------------------------------------

FIGURE_OUT = OUT / "Figure3_final_GEE_corrected"

fig.savefig(
    str(FIGURE_OUT) + ".png",
    dpi=400,
    bbox_inches="tight",
)

fig.savefig(
    str(FIGURE_OUT) + ".pdf",
    bbox_inches="tight",
)

fig.savefig(
    str(FIGURE_OUT) + ".tiff",
    dpi=400,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"},
)

plt.show()

print("\nFigura salva em:")
print(str(FIGURE_OUT) + ".png")
print(str(FIGURE_OUT) + ".pdf")
print(str(FIGURE_OUT) + ".tiff")
