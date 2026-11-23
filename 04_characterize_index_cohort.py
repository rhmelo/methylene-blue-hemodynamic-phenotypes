"""Source: original notebook cell 27. Run through run.py."""

from common import artifact, display, mimic_source, phase_dir, run_stamp, source_file

# ============================================================
# CÓDIGO 7B — CORREÇÃO DA CARACTERIZAÇÃO DA COORTE
#
# Corrige:
# - colunas duplicadas de icustays
# - unidade de terapia intensiva ausente
# - hours_icu_to_mb ausente
# - mortalidade na UTI indisponível
#
# Fonte:

#
# Output:
#   HIRID/.../MIMIC_Phase
# ============================================================

import pandas as pd
import numpy as np
import gc

# ------------------------------------------------------------
# 1. Caminhos
# ------------------------------------------------------------

SOURCE_DIR = mimic_source()

PROJECT_DIR = phase_dir("mimic_iv")

COHORT_DIR = PROJECT_DIR / "cohorts"
QC_DIR = PROJECT_DIR / "qc"
TABLE_DIR = PROJECT_DIR / "tables"

for folder in [COHORT_DIR, QC_DIR, TABLE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("CÓDIGO 7B — CORREÇÃO DA CARACTERIZAÇÃO")
print("=" * 90)

# ------------------------------------------------------------
# 2. Função para localizar arquivos-fonte
# ------------------------------------------------------------


def locate_file(filename):
    return source_file(SOURCE_DIR, filename)


PATIENTS_PATH = locate_file("patients.csv.gz")
ADMISSIONS_PATH = locate_file("admissions.csv.gz")
ICUSTAYS_PATH = locate_file("icustays.csv.gz")

# ------------------------------------------------------------
# 3. Localizar a exposição índice preliminar mais recente
# ------------------------------------------------------------

INDEX_PATH = artifact(COHORT_DIR, "methylene_blue_preliminary_index_exposure_*.csv.gz")

print("\nExposição índice utilizada:")
print(INDEX_PATH)

# ------------------------------------------------------------
# 4. Carregar a exposição índice
# ------------------------------------------------------------

index_raw = pd.read_csv(INDEX_PATH, compression="gzip", low_memory=False)

index_raw["mb_time_preliminary"] = pd.to_datetime(
    index_raw["mb_time_preliminary"], errors="coerce"
)

index_raw = (
    index_raw.sort_values(["subject_id", "mb_time_preliminary"])
    .drop_duplicates(subset=["subject_id"], keep="first")
    .copy()
)

print("\nExposição índice carregada:")
print("Pacientes:", index_raw["subject_id"].nunique())
print("Internações:", index_raw["hadm_id"].nunique())
print("UTIs:", index_raw["stay_id"].nunique())

# ------------------------------------------------------------
# 5. Remover colunas antigas de patients/admissions/icustays
# ------------------------------------------------------------

# Mantemos somente identificadores e dados específicos da exposição.
# Isso impede conflitos e criação de sufixos nos novos merges.

columns_to_remove = [
    # patients
    "gender",
    "anchor_age",
    "anchor_year",
    "anchor_year_group",
    "dod",
    # admissions
    "admittime",
    "dischtime",
    "deathtime",
    "admission_type",
    "admit_provider_id",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "marital_status",
    "race",
    "edregtime",
    "edouttime",
    "hospital_expire_flag",
    # icustays
    "first_careunit",
    "last_careunit",
    "intime",
    "outtime",
    "los",
    # possíveis versões sufixadas
    "first_careunit_icu",
    "last_careunit_icu",
    "intime_icu",
    "outtime_icu",
    "los_icu",
    # variáveis derivadas antigas
    "age_at_admission",
    "hours_hospital_to_mb",
    "hours_icu_to_mb",
    "hours_mb_to_icu_discharge",
    "hospital_los_days",
    "icu_los_days_calculated",
    "hospital_mortality",
    "icu_mortality",
    "death_after_mb",
    "days_mb_to_death",
    "sex",
    "careunit_group",
]

index_clean = index_raw.drop(
    columns=[col for col in columns_to_remove if col in index_raw.columns],
    errors="ignore",
).copy()

print("\nColunas removidas para evitar conflitos:")
print([col for col in columns_to_remove if col in index_raw.columns])

# ------------------------------------------------------------
# 6. Carregar patients
# ------------------------------------------------------------

patients_header = pd.read_csv(PATIENTS_PATH, compression="gzip", nrows=0)

patient_columns = [
    col
    for col in [
        "subject_id",
        "gender",
        "anchor_age",
        "anchor_year",
        "anchor_year_group",
        "dod",
    ]
    if col in patients_header.columns
]

patients = pd.read_csv(
    PATIENTS_PATH, compression="gzip", usecols=patient_columns, low_memory=False
)

patients = patients[patients["subject_id"].isin(index_clean["subject_id"])].copy()

if "dod" in patients.columns:
    patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")

# ------------------------------------------------------------
# 7. Carregar admissions
# ------------------------------------------------------------

admissions_header = pd.read_csv(ADMISSIONS_PATH, compression="gzip", nrows=0)

admission_columns = [
    col
    for col in [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "edregtime",
        "edouttime",
        "hospital_expire_flag",
    ]
    if col in admissions_header.columns
]

admissions = pd.read_csv(
    ADMISSIONS_PATH, compression="gzip", usecols=admission_columns, low_memory=False
)

admissions = admissions[admissions["hadm_id"].isin(index_clean["hadm_id"])].copy()

for column in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
    if column in admissions.columns:
        admissions[column] = pd.to_datetime(admissions[column], errors="coerce")

# ------------------------------------------------------------
# 8. Carregar icustays
# ------------------------------------------------------------

icu_header = pd.read_csv(ICUSTAYS_PATH, compression="gzip", nrows=0)

icu_columns = [
    col
    for col in [
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
        "los",
    ]
    if col in icu_header.columns
]

icustays = pd.read_csv(
    ICUSTAYS_PATH, compression="gzip", usecols=icu_columns, low_memory=False
)

icustays = icustays[icustays["stay_id"].isin(index_clean["stay_id"])].copy()

icustays["intime"] = pd.to_datetime(icustays["intime"], errors="coerce")

icustays["outtime"] = pd.to_datetime(icustays["outtime"], errors="coerce")

# ------------------------------------------------------------
# 9. Conferir unicidade antes dos merges
# ------------------------------------------------------------

if patients["subject_id"].duplicated().any():
    raise RuntimeError("Há subject_id duplicado em patients após o filtro.")

if admissions["hadm_id"].duplicated().any():
    raise RuntimeError("Há hadm_id duplicado em admissions após o filtro.")

if icustays["stay_id"].duplicated().any():
    raise RuntimeError("Há stay_id duplicado em icustays após o filtro.")

# ------------------------------------------------------------
# 10. Realizar merges sem colisões
# ------------------------------------------------------------

cohort = (
    index_clean.merge(patients, on="subject_id", how="left", validate="one_to_one")
    .merge(admissions, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
    .merge(
        icustays,
        on=["subject_id", "hadm_id", "stay_id"],
        how="left",
        validate="one_to_one",
    )
)

print("\nLinhas após merges:", len(cohort))

# ------------------------------------------------------------
# 11. Verificar disponibilidade das colunas da UTI
# ------------------------------------------------------------

icu_merge_qc = pd.DataFrame(
    {
        "variable": ["intime", "outtime", "first_careunit", "last_careunit"],
        "n_available": [
            cohort["intime"].notna().sum(),
            cohort["outtime"].notna().sum(),
            cohort["first_careunit"].notna().sum(),
            cohort["last_careunit"].notna().sum(),
        ],
        "n_total": len(cohort),
    }
)

print("\n" + "=" * 90)
print("CONTROLE DO MERGE COM ICUSTAYS")
print("=" * 90)

display(icu_merge_qc)

# ------------------------------------------------------------
# 12. Calcular idade
# ------------------------------------------------------------

cohort["age_at_admission"] = cohort["anchor_age"] + (
    cohort["admittime"].dt.year - cohort["anchor_year"]
)

# ------------------------------------------------------------
# 13. Calcular intervalos
# ------------------------------------------------------------

cohort["hours_hospital_to_mb"] = (
    cohort["mb_time_preliminary"] - cohort["admittime"]
).dt.total_seconds() / 3600

cohort["hours_icu_to_mb"] = (
    cohort["mb_time_preliminary"] - cohort["intime"]
).dt.total_seconds() / 3600

cohort["hours_mb_to_icu_discharge"] = (
    cohort["outtime"] - cohort["mb_time_preliminary"]
).dt.total_seconds() / 3600

cohort["hospital_los_days"] = (
    cohort["dischtime"] - cohort["admittime"]
).dt.total_seconds() / 86400

cohort["icu_los_days_calculated"] = (
    cohort["outtime"] - cohort["intime"]
).dt.total_seconds() / 86400

# ------------------------------------------------------------
# 14. Mortalidade hospitalar
# ------------------------------------------------------------

if "hospital_expire_flag" in cohort.columns:

    cohort["hospital_mortality"] = pd.to_numeric(
        cohort["hospital_expire_flag"], errors="coerce"
    )

else:

    cohort["hospital_mortality"] = (cohort["deathtime"].notna()).astype(int)

# ------------------------------------------------------------
# 15. Mortalidade durante a estadia de UTI
# ------------------------------------------------------------

cohort["icu_mortality"] = (
    cohort["deathtime"].notna()
    & cohort["intime"].notna()
    & cohort["outtime"].notna()
    & (cohort["deathtime"] >= cohort["intime"])
    & (cohort["deathtime"] <= cohort["outtime"])
).astype(int)

# ------------------------------------------------------------
# 16. Morte após o azul
# ------------------------------------------------------------

cohort["death_after_mb"] = (
    cohort["deathtime"].notna() & (cohort["deathtime"] >= cohort["mb_time_preliminary"])
).astype(int)

cohort["days_mb_to_death"] = (
    cohort["deathtime"] - cohort["mb_time_preliminary"]
).dt.total_seconds() / 86400

cohort.loc[cohort["days_mb_to_death"] < 0, "days_mb_to_death"] = np.nan

# ------------------------------------------------------------
# 17. Variáveis categóricas
# ------------------------------------------------------------

cohort["sex"] = cohort["gender"]

cohort["careunit_group"] = cohort["first_careunit"].fillna("Missing").astype(str)

# ------------------------------------------------------------
# 18. Funções de resumo
# ------------------------------------------------------------


def continuous_row(variable, label):

    x = pd.to_numeric(cohort[variable], errors="coerce")

    return {
        "variable": label,
        "n": int(x.notna().sum()),
        "mean": x.mean(),
        "sd": x.std(),
        "median": x.median(),
        "p25": x.quantile(0.25),
        "p75": x.quantile(0.75),
        "min": x.min(),
        "max": x.max(),
    }


def binary_row(variable, label):

    x = pd.to_numeric(cohort[variable], errors="coerce")

    n_available = int(x.notna().sum())
    n_positive = int((x == 1).sum())

    return {
        "variable": label,
        "n_positive": n_positive,
        "n_available": n_available,
        "percent": (100 * n_positive / n_available if n_available > 0 else np.nan),
    }


# ------------------------------------------------------------
# 19. Resumo contínuo corrigido
# ------------------------------------------------------------

continuous_summary = pd.DataFrame(
    [
        continuous_row("age_at_admission", "Age at hospital admission, years"),
        continuous_row("dose_mg_clean", "Index methylene blue dose, mg"),
        continuous_row(
            "hours_hospital_to_mb", "Time from hospital admission to MB, hours"
        ),
        continuous_row("hours_icu_to_mb", "Time from ICU admission to MB, hours"),
        continuous_row(
            "hours_mb_to_icu_discharge", "Time from MB to ICU discharge, hours"
        ),
        continuous_row("icu_los_days_calculated", "ICU length of stay, days"),
        continuous_row("hospital_los_days", "Hospital length of stay, days"),
    ]
)

# ------------------------------------------------------------
# 20. Resumo binário corrigido
# ------------------------------------------------------------

binary_summary = pd.DataFrame(
    [
        binary_row("hospital_mortality", "Hospital mortality"),
        binary_row("icu_mortality", "ICU mortality"),
        binary_row("death_after_mb", "Death after preliminary MB time"),
    ]
)

# ------------------------------------------------------------
# 21. Resumo por unidade
# ------------------------------------------------------------

careunit_summary = (
    cohort["careunit_group"]
    .value_counts(dropna=False)
    .rename_axis("careunit")
    .reset_index(name="n")
)

careunit_summary["percent"] = 100 * careunit_summary["n"] / len(cohort)

# ------------------------------------------------------------
# 22. Resumo por tipo de admissão
# ------------------------------------------------------------

admission_type_summary = (
    cohort["admission_type"]
    .fillna("Missing")
    .value_counts(dropna=False)
    .rename_axis("admission_type")
    .reset_index(name="n")
)

admission_type_summary["percent"] = 100 * admission_type_summary["n"] / len(cohort)

# ------------------------------------------------------------
# 23. Mostrar resultados corrigidos
# ------------------------------------------------------------

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 220)

print("\n" + "=" * 90)
print("RESUMO CONTÍNUO CORRIGIDO")
print("=" * 90)

display(continuous_summary)

print("\n" + "=" * 90)
print("RESUMO DE MORTALIDADE CORRIGIDO")
print("=" * 90)

display(binary_summary)

print("\n" + "=" * 90)
print("UNIDADES DE TERAPIA INTENSIVA")
print("=" * 90)

display(careunit_summary)

print("\n" + "=" * 90)
print("TIPO DE ADMISSÃO")
print("=" * 90)

display(admission_type_summary)

# ------------------------------------------------------------
# 24. Controle de coerência temporal
# ------------------------------------------------------------

temporal_qc = pd.DataFrame(
    {
        "metric": [
            "MB before hospital admission",
            "MB before ICU admission",
            "MB after ICU discharge",
            "Missing ICU admission time",
            "Missing ICU discharge time",
            "Missing care unit",
        ],
        "value": [
            int((cohort["hours_hospital_to_mb"] < 0).sum()),
            int((cohort["hours_icu_to_mb"] < 0).sum()),
            int((cohort["hours_mb_to_icu_discharge"] < 0).sum()),
            int(cohort["intime"].isna().sum()),
            int(cohort["outtime"].isna().sum()),
            int(cohort["first_careunit"].isna().sum()),
        ],
    }
)

print("\n" + "=" * 90)
print("CONTROLE DE COERÊNCIA TEMPORAL")
print("=" * 90)

display(temporal_qc)

# ------------------------------------------------------------
# 25. Salvar outputs
# ------------------------------------------------------------

timestamp = run_stamp()

cohort_path = (
    COHORT_DIR / f"methylene_blue_characterized_cohort_corrected_{timestamp}.csv.gz"
)

continuous_path = (
    TABLE_DIR / f"methylene_blue_continuous_summary_corrected_{timestamp}.csv"
)

binary_path = TABLE_DIR / f"methylene_blue_mortality_summary_corrected_{timestamp}.csv"

careunit_path = TABLE_DIR / f"methylene_blue_careunit_summary_corrected_{timestamp}.csv"

admission_path = TABLE_DIR / f"methylene_blue_admission_type_corrected_{timestamp}.csv"

merge_qc_path = QC_DIR / f"methylene_blue_icustays_merge_qc_{timestamp}.csv"

temporal_qc_path = QC_DIR / f"methylene_blue_temporal_qc_{timestamp}.csv"

report_path = QC_DIR / f"methylene_blue_code7b_report_{timestamp}.txt"

cohort.to_csv(cohort_path, index=False, compression="gzip")

continuous_summary.to_csv(continuous_path, index=False)

binary_summary.to_csv(binary_path, index=False)

careunit_summary.to_csv(careunit_path, index=False)

admission_type_summary.to_csv(admission_path, index=False)

icu_merge_qc.to_csv(merge_qc_path, index=False)

temporal_qc.to_csv(temporal_qc_path, index=False)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("METHYLENE BLUE — CORRECTED COHORT CHARACTERIZATION\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"Patients: {len(cohort)}\n")
    file.write(f"ICU time available: " f"{cohort['intime'].notna().sum()}\n")
    file.write(f"Care unit available: " f"{cohort['first_careunit'].notna().sum()}\n")
    file.write(
        f"Median hours from ICU admission to MB: "
        f"{cohort['hours_icu_to_mb'].median()}\n"
    )
    file.write(
        f"Hospital mortality: " f"{100 * cohort['hospital_mortality'].mean():.1f}%\n"
    )
    file.write(f"ICU mortality: " f"{100 * cohort['icu_mortality'].mean():.1f}%\n")

print("\n" + "=" * 90)
print("OUTPUTS SALVOS")
print("=" * 90)

print(cohort_path)
print(continuous_path)
print(binary_path)
print(careunit_path)
print(admission_path)
print(merge_qc_path)
print(temporal_qc_path)
print(report_path)

gc.collect()

print("\nCÓDIGO 7B CONCLUÍDO COM SUCESSO.")
