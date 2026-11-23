"""Source: original notebook cell 56. Run through run.py."""

from common import artifact, display, mimic_source, phase_dir, run_stamp, source_file

# ============================================================
# CÓDIGO 13A-QC — VERSÃO CORRIGIDA
# AUDITORIA DO SEGUIMENTO: ÓBITO, ALTA E NOVA ESTADIA DE UTI
#
# Corrige duplicação de colunas após merge com admissions.
#
# Não relê inputevents ou chartevents.
# ============================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------

# ------------------------------------------------------------

SOURCE_DIR = mimic_source()

PROJECT_DIR = phase_dir("mimic_iv")

TABLE_DIR = PROJECT_DIR / "tables"
QC_DIR = PROJECT_DIR / "qc"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. Funções auxiliares
# ------------------------------------------------------------


def latest_project_file(patterns):
    return artifact(PROJECT_DIR, patterns)


def locate_source_file(filename):
    return source_file(SOURCE_DIR, filename)


# ------------------------------------------------------------
# 3. Localizar os arquivos
# ------------------------------------------------------------

COHORT_PATH = artifact(
    PROJECT_DIR, "METHYLENE_BLUE_ANALYTIC_COHORT_FROZEN_FINAL_*.csv.gz"
)

ICUSTAYS_PATH = locate_source_file("icustays.csv.gz")

ADMISSIONS_PATH = locate_source_file("admissions.csv.gz")

print("=" * 100)
print("AUDITORIA DO ENCERRAMENTO DO SEGUIMENTO — CORRIGIDA")
print("=" * 100)

print("\nCoorte:")
print(COHORT_PATH)

print("\nICU stays:")
print(ICUSTAYS_PATH)

print("\nAdmissions:")
print(ADMISSIONS_PATH)

# ------------------------------------------------------------
# 4. Carregar a coorte
# ------------------------------------------------------------

cohort = pd.read_csv(COHORT_PATH, compression="gzip", low_memory=False)

cohort = (
    cohort.sort_values(["subject_id", "mb_time_preliminary"])
    .drop_duplicates(subset=["subject_id"], keep="first")
    .copy()
)

for column in ["subject_id", "hadm_id", "stay_id"]:
    cohort[column] = pd.to_numeric(cohort[column], errors="coerce").astype("Int64")

cohort["mb_time_preliminary"] = pd.to_datetime(
    cohort["mb_time_preliminary"], errors="coerce"
)

# Evitar duplicação no merge.
# Esses dados serão obtidos novamente das tabelas oficiais.
columns_to_refresh = [
    "intime",
    "outtime",
    "first_careunit",
    "last_careunit",
    "los",
    "admittime",
    "dischtime",
    "deathtime",
    "hospital_expire_flag",
    "admission_type",
    "discharge_location",
]

cohort = cohort.drop(
    columns=[column for column in columns_to_refresh if column in cohort.columns],
    errors="ignore",
)

print("\nPacientes na coorte:", len(cohort))

# ------------------------------------------------------------
# 5. Carregar icustays
# ------------------------------------------------------------

icu_header = pd.read_csv(ICUSTAYS_PATH, compression="gzip", nrows=0)

icu_usecols = [
    column
    for column in [
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
        "los",
    ]
    if column in icu_header.columns
]

icu = pd.read_csv(
    ICUSTAYS_PATH, compression="gzip", usecols=icu_usecols, low_memory=False
)

for column in ["subject_id", "hadm_id", "stay_id"]:
    icu[column] = pd.to_numeric(icu[column], errors="coerce").astype("Int64")

for column in ["intime", "outtime"]:
    if column in icu.columns:
        icu[column] = pd.to_datetime(icu[column], errors="coerce")

# ------------------------------------------------------------
# 6. Carregar admissions
# ------------------------------------------------------------

admissions_header = pd.read_csv(ADMISSIONS_PATH, compression="gzip", nrows=0)

admission_usecols = [
    column
    for column in [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "hospital_expire_flag",
        "admission_type",
        "discharge_location",
    ]
    if column in admissions_header.columns
]

admissions = pd.read_csv(
    ADMISSIONS_PATH, compression="gzip", usecols=admission_usecols, low_memory=False
)

for column in ["subject_id", "hadm_id"]:
    admissions[column] = pd.to_numeric(admissions[column], errors="coerce").astype(
        "Int64"
    )

for column in ["admittime", "dischtime", "deathtime"]:
    if column in admissions.columns:
        admissions[column] = pd.to_datetime(admissions[column], errors="coerce")

# Uma linha por internação
admissions = admissions.drop_duplicates(subset=["subject_id", "hadm_id"], keep="first")

# ------------------------------------------------------------
# 7. Integrar a estadia índice e a internação hospitalar
# ------------------------------------------------------------

index_icu = icu.rename(
    columns={
        "intime": "index_icu_intime",
        "outtime": "index_icu_outtime",
        "los": "index_icu_los_days",
        "first_careunit": "index_first_careunit",
        "last_careunit": "index_last_careunit",
    }
)

audit = cohort.merge(
    index_icu,
    on=["subject_id", "hadm_id", "stay_id"],
    how="left",
    validate="one_to_one",
).merge(admissions, on=["subject_id", "hadm_id"], how="left", validate="many_to_one")

required_after_merge = ["index_icu_outtime", "dischtime", "deathtime"]

print("\nDisponibilidade após os merges:")

for column in required_after_merge:
    if column in audit.columns:
        print(f"{column:<30}", f"{audit[column].notna().sum()}/{len(audit)}")
    else:
        print(f"{column:<30}", "COLUNA AUSENTE")

# ------------------------------------------------------------
# 8. Procurar outras estadias na mesma internação
# ------------------------------------------------------------

index_identifiers = cohort[
    ["subject_id", "hadm_id", "stay_id", "mb_time_preliminary"]
].rename(columns={"stay_id": "index_stay_id"})

same_admission_stays = index_identifiers.merge(
    icu, on=["subject_id", "hadm_id"], how="left"
)

subsequent_stays = same_admission_stays[
    same_admission_stays["stay_id"].notna()
    & same_admission_stays["index_stay_id"].notna()
    & (same_admission_stays["stay_id"] != same_admission_stays["index_stay_id"])
    & same_admission_stays["intime"].notna()
    & (same_admission_stays["intime"] >= same_admission_stays["mb_time_preliminary"])
].copy()

subsequent_stays["hours_mb_to_next_icu_intime"] = (
    subsequent_stays["intime"] - subsequent_stays["mb_time_preliminary"]
).dt.total_seconds() / 3600

next_stay_summary = (
    subsequent_stays.sort_values(["subject_id", "intime"])
    .groupby(["subject_id", "hadm_id", "index_stay_id"], as_index=False)
    .agg(
        n_subsequent_icu_stays=("stay_id", "nunique"),
        next_stay_id=("stay_id", "first"),
        next_icu_intime=("intime", "first"),
        next_icu_outtime=("outtime", "first"),
        next_first_careunit=("first_careunit", "first"),
        hours_mb_to_next_icu_intime=("hours_mb_to_next_icu_intime", "first"),
    )
    .rename(columns={"index_stay_id": "stay_id"})
)

audit = audit.merge(
    next_stay_summary,
    on=["subject_id", "hadm_id", "stay_id"],
    how="left",
    validate="one_to_one",
)

audit["n_subsequent_icu_stays"] = audit["n_subsequent_icu_stays"].fillna(0).astype(int)

audit["has_subsequent_icu_stay"] = audit["n_subsequent_icu_stays"] > 0

# ------------------------------------------------------------
# 9. Calcular tempos após o azul
# ------------------------------------------------------------

audit["hours_mb_to_index_icu_out"] = (
    audit["index_icu_outtime"] - audit["mb_time_preliminary"]
).dt.total_seconds() / 3600

audit["hours_mb_to_hospital_discharge"] = (
    audit["dischtime"] - audit["mb_time_preliminary"]
).dt.total_seconds() / 3600

audit["hours_mb_to_death"] = (
    audit["deathtime"] - audit["mb_time_preliminary"]
).dt.total_seconds() / 3600

audit["death_after_mb"] = audit["hours_mb_to_death"].notna() & audit[
    "hours_mb_to_death"
].ge(0)

for threshold in [6, 24, 48, 72]:

    audit[f"death_within_{threshold}h"] = audit["death_after_mb"] & audit[
        "hours_mb_to_death"
    ].le(threshold)

    audit[f"index_icu_out_within_{threshold}h"] = audit[
        "hours_mb_to_index_icu_out"
    ].notna() & audit["hours_mb_to_index_icu_out"].between(
        0, threshold, inclusive="both"
    )

# ------------------------------------------------------------
# 10. Relação temporal entre óbito e saída da UTI
# ------------------------------------------------------------

audit["hours_death_minus_icu_out"] = (
    audit["deathtime"] - audit["index_icu_outtime"]
).dt.total_seconds() / 3600

audit["death_near_index_icu_out"] = audit["hours_death_minus_icu_out"].notna() & audit[
    "hours_death_minus_icu_out"
].abs().le(6)

audit["next_stay_gap_hours"] = (
    audit["next_icu_intime"] - audit["index_icu_outtime"]
).dt.total_seconds() / 3600

audit["next_stay_shortly_after_index_out"] = audit[
    "next_stay_gap_hours"
].notna() & audit["next_stay_gap_hours"].between(-1, 24, inclusive="both")

audit["hospital_discharge_gap_hours"] = (
    audit["dischtime"] - audit["index_icu_outtime"]
).dt.total_seconds() / 3600

audit["hospital_discharge_near_icu_out"] = audit[
    "hospital_discharge_gap_hours"
].notna() & audit["hospital_discharge_gap_hours"].abs().le(6)

# ------------------------------------------------------------
# 11. Motivo provável do encerramento do stay índice
# ------------------------------------------------------------

audit["followup_end_reason"] = np.select(
    [
        audit["death_near_index_icu_out"],
        audit["next_stay_shortly_after_index_out"],
        audit["hospital_discharge_near_icu_out"],
        audit["index_icu_outtime"].notna(),
    ],
    [
        "Death around index ICU outtime",
        "Subsequent ICU stay in same admission",
        "Hospital discharge around ICU outtime",
        "Left index ICU alive or reason uncertain",
    ],
    default="Missing index ICU outtime",
)

# ------------------------------------------------------------
# 12. Resumo de mortalidade precoce
# ------------------------------------------------------------

death_summary = pd.DataFrame(
    {
        "time_window": [
            "Death within 6 h",
            "Death within 24 h",
            "Death within 48 h",
            "Death within 72 h",
            "Any hospital death after MB",
        ],
        "n": [
            int(audit["death_within_6h"].sum()),
            int(audit["death_within_24h"].sum()),
            int(audit["death_within_48h"].sum()),
            int(audit["death_within_72h"].sum()),
            int(audit["death_after_mb"].sum()),
        ],
    }
)

death_summary["total"] = len(audit)

death_summary["percent"] = 100 * death_summary["n"] / death_summary["total"]

# ------------------------------------------------------------
# 13. Resumo dos motivos de encerramento
# ------------------------------------------------------------

end_reason_summary = (
    audit["followup_end_reason"]
    .value_counts(dropna=False)
    .rename_axis("followup_end_reason")
    .reset_index(name="n")
)

end_reason_summary["percent"] = 100 * end_reason_summary["n"] / len(audit)

# ------------------------------------------------------------
# 14. Resumo de múltiplas estadias
# ------------------------------------------------------------

multi_stay_summary = pd.DataFrame(
    {
        "metric": [
            "Patients in cohort",
            "Patients with index ICU outtime available",
            "Patients with subsequent ICU stay in same admission",
            "Subsequent ICU stay within 24 h after index ICU outtime",
            "Deaths after methylene blue",
            "Deaths within 24 h",
            "Deaths within 48 h",
            "Deaths within 72 h",
            "Deaths around index ICU outtime",
            "Index ICU outtime within 24 h",
            "Index ICU outtime within 48 h",
            "Index ICU outtime within 72 h",
        ],
        "value": [
            len(audit),
            int(audit["index_icu_outtime"].notna().sum()),
            int(audit["has_subsequent_icu_stay"].sum()),
            int(audit["next_stay_shortly_after_index_out"].sum()),
            int(audit["death_after_mb"].sum()),
            int(audit["death_within_24h"].sum()),
            int(audit["death_within_48h"].sum()),
            int(audit["death_within_72h"].sum()),
            int(audit["death_near_index_icu_out"].sum()),
            int(audit["index_icu_out_within_24h"].sum()),
            int(audit["index_icu_out_within_48h"].sum()),
            int(audit["index_icu_out_within_72h"].sum()),
        ],
    }
)

# ------------------------------------------------------------
# 15. Pacientes com menos de 72 h no stay índice
# ------------------------------------------------------------

short_followup = audit[
    audit["hours_mb_to_index_icu_out"].notna()
    & audit["hours_mb_to_index_icu_out"].lt(72)
].copy()

short_followup_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "hemodynamic_phenotype",
    "mb_time_preliminary",
    "index_icu_intime",
    "index_icu_outtime",
    "hours_mb_to_index_icu_out",
    "deathtime",
    "hours_mb_to_death",
    "hours_death_minus_icu_out",
    "hospital_expire_flag",
    "dischtime",
    "hours_mb_to_hospital_discharge",
    "has_subsequent_icu_stay",
    "n_subsequent_icu_stays",
    "next_stay_id",
    "next_icu_intime",
    "next_stay_gap_hours",
    "followup_end_reason",
]

short_followup_columns = [
    column for column in short_followup_columns if column in short_followup.columns
]

short_followup = short_followup[short_followup_columns].sort_values(
    "hours_mb_to_index_icu_out"
)

# ------------------------------------------------------------
# 16. Mostrar os resultados
# ------------------------------------------------------------

pd.set_option("display.max_rows", 100)

pd.set_option("display.max_columns", 50)

pd.set_option("display.width", 350)

print("\n" + "=" * 100)
print("ÓBITOS APÓS O AZUL")
print("=" * 100)

display(death_summary)

print("\n" + "=" * 100)
print("MOTIVO PROVÁVEL DO ENCERRAMENTO DO STAY ÍNDICE")
print("=" * 100)

display(end_reason_summary)

print("\n" + "=" * 100)
print("AUDITORIA DE MÚLTIPLAS ESTADIAS E ÓBITO")
print("=" * 100)

display(multi_stay_summary)

print("\n" + "=" * 100)
print("PACIENTES COM MENOS DE 72 HORAS NO STAY ÍNDICE")
print("=" * 100)

display(short_followup)

# ------------------------------------------------------------
# 17. Salvar resultados
# ------------------------------------------------------------

timestamp = run_stamp()

audit_path = QC_DIR / f"methylene_blue_followup_end_audit_{timestamp}.csv"

short_followup_path = QC_DIR / f"methylene_blue_short_followup_patients_{timestamp}.csv"

death_summary_path = TABLE_DIR / f"methylene_blue_early_death_summary_{timestamp}.csv"

end_reason_path = TABLE_DIR / f"methylene_blue_followup_end_reasons_{timestamp}.csv"

multi_stay_path = QC_DIR / f"methylene_blue_multiple_icu_stays_qc_{timestamp}.csv"

audit.to_csv(audit_path, index=False)

short_followup.to_csv(short_followup_path, index=False)

death_summary.to_csv(death_summary_path, index=False)

end_reason_summary.to_csv(end_reason_path, index=False)

multi_stay_summary.to_csv(multi_stay_path, index=False)

print("\nArquivos salvos:")
print(audit_path)
print(short_followup_path)
print(death_summary_path)
print(end_reason_path)
print(multi_stay_path)

print("\nAUDITORIA CONCLUÍDA COM SUCESSO.")
