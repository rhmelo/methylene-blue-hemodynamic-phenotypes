"""Source: original notebook cell 28. Run through run.py."""

from common import artifact, display, mimic_source, phase_dir, run_stamp, source_file

# ============================================================
# CÓDIGO 8 — COORTE PRINCIPAL:
# SEPSE + NOREPINEFRINA ATIVA + AZUL DE METILENO IV
#
# População inicial:
#   Pacientes adultos com prescrição IV plausível de azul de
#   metileno durante uma estadia de UTI.
#
# Critérios para a coorte principal:
#   1. Diagnóstico de sepse na internação
#   2. Norepinefrina ativa no momento do azul de metileno
#   3. Exclusão de metemoglobinemia
#
# Não haverá subdivisão administrativa entre:
#   - sepse
#   - sepse grave
#   - choque séptico
#
# Output principal:
#   methylene_blue_primary_sepsis_norepinephrine_cohort
#
# Esse arquivo será a BASE DA TABELA 1.
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
METADATA_DIR = PROJECT_DIR / "metadata"

for folder in [COHORT_DIR, QC_DIR, TABLE_DIR, METADATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("CÓDIGO 8 — SEPSE + NOREPINEFRINA ATIVA + AZUL DE METILENO")
print("=" * 90)

print("\nFonte MIMIC-IV:")
print(SOURCE_DIR)

print("\nOutputs:")
print(PROJECT_DIR)

# ------------------------------------------------------------
# 2. Funções auxiliares
# ------------------------------------------------------------


def locate_source_file(filename):
    return source_file(SOURCE_DIR, filename)


def latest_file(directory, pattern):
    return artifact(directory, pattern)


# ------------------------------------------------------------
# 3. Localizar arquivos
# ------------------------------------------------------------

COHORT_PATH = latest_file(
    COHORT_DIR, "methylene_blue_characterized_cohort_corrected_*.csv.gz"
)

DIAGNOSES_PATH = latest_file(COHORT_DIR, "methylene_blue_cohort_diagnoses_*.csv.gz")

D_ITEMS_PATH = locate_source_file("d_items.csv.gz")

INPUTEVENTS_PATH = locate_source_file("inputevents.csv.gz")

print("\nCoorte ampla:")
print(COHORT_PATH)

print("\nDiagnósticos:")
print(DIAGNOSES_PATH)

print("\nDicionário de itens:")
print(D_ITEMS_PATH)

print("\nInputevents:")
print(INPUTEVENTS_PATH)

# ------------------------------------------------------------
# 4. Carregar a coorte ampla de 190 pacientes
# ------------------------------------------------------------

cohort = pd.read_csv(COHORT_PATH, compression="gzip", low_memory=False)

if cohort.empty:
    raise RuntimeError("A coorte caracterizada está vazia.")

required_cohort_columns = ["subject_id", "hadm_id", "stay_id", "mb_time_preliminary"]

missing_cohort_columns = [
    column for column in required_cohort_columns if column not in cohort.columns
]

if missing_cohort_columns:
    raise RuntimeError(
        "Colunas obrigatórias ausentes na coorte: " f"{missing_cohort_columns}"
    )

cohort["mb_time_preliminary"] = pd.to_datetime(
    cohort["mb_time_preliminary"], errors="coerce"
)

cohort = (
    cohort.sort_values(["subject_id", "mb_time_preliminary"])
    .drop_duplicates(subset=["subject_id"], keep="first")
    .copy()
)

print("\nCoorte ampla carregada:")
print("Pacientes:", cohort["subject_id"].nunique())
print("Internações:", cohort["hadm_id"].nunique())
print("UTIs:", cohort["stay_id"].nunique())

# ------------------------------------------------------------
# 5. Carregar diagnósticos da coorte
# ------------------------------------------------------------

diagnoses = pd.read_csv(DIAGNOSES_PATH, compression="gzip", low_memory=False)

required_diagnosis_columns = ["subject_id", "hadm_id", "icd_code", "icd_version"]

missing_diagnosis_columns = [
    column for column in required_diagnosis_columns if column not in diagnoses.columns
]

if missing_diagnosis_columns:
    raise RuntimeError(
        "Colunas obrigatórias ausentes em diagnoses: " f"{missing_diagnosis_columns}"
    )

print("\nDiagnósticos carregados:", f"{len(diagnoses):,}")

# ------------------------------------------------------------
# 6. Normalizar códigos diagnósticos
# ------------------------------------------------------------

diagnoses["icd_code_normalized"] = (
    diagnoses["icd_code"]
    .fillna("")
    .astype(str)
    .str.upper()
    .str.replace(r"[^A-Z0-9]", "", regex=True)
)

diagnoses["icd_version_numeric"] = pd.to_numeric(
    diagnoses["icd_version"], errors="coerce"
).astype("Int64")

if "long_title" in diagnoses.columns:

    diagnoses["diagnosis_text_normalized"] = (
        diagnoses["long_title"].fillna("").astype(str).str.lower()
    )

else:

    diagnoses["diagnosis_text_normalized"] = ""

# ------------------------------------------------------------
# 7. Definição simples de sepse: SIM/NÃO
# ------------------------------------------------------------

is_icd9 = diagnoses["icd_version_numeric"].eq(9)

is_icd10 = diagnoses["icd_version_numeric"].eq(10)

# ICD-9:
# 038*   septicemia
# 99591  sepsis
# 99592  severe sepsis
# 78552  septic shock

sepsis_icd9 = is_icd9 & (
    diagnoses["icd_code_normalized"].str.startswith("038", na=False)
    | diagnoses["icd_code_normalized"].isin(["99591", "99592", "78552"])
)

# ICD-10:
# A40*   streptococcal sepsis
# A41*   other sepsis
# R6520  severe sepsis without septic shock
# R6521  severe sepsis with septic shock
# B377   candidal sepsis

sepsis_icd10 = is_icd10 & (
    diagnoses["icd_code_normalized"].str.startswith(("A40", "A41"), na=False)
    | diagnoses["icd_code_normalized"].isin(["R6520", "R6521", "B377"])
)

diagnoses["sepsis_diagnosis"] = sepsis_icd9 | sepsis_icd10

# ------------------------------------------------------------
# 8. Diagnóstico de metemoglobinemia
# ------------------------------------------------------------

diagnoses["methemoglobinemia_diagnosis"] = diagnoses[
    "diagnosis_text_normalized"
].str.contains(r"methemoglobin|methaemoglobin", regex=True, na=False)

# ------------------------------------------------------------
# 9. Agregar sepse e metemoglobinemia por internação
# ------------------------------------------------------------

diagnosis_flags = diagnoses.groupby("hadm_id", as_index=False).agg(
    sepsis_diagnosis=("sepsis_diagnosis", "max"),
    methemoglobinemia_diagnosis=("methemoglobinemia_diagnosis", "max"),
)

# Remover flags antigas, se existirem
for column in [
    "sepsis_diagnosis",
    "methemoglobinemia_diagnosis",
    "sepsis_or_septic_shock",
    "methemoglobinemia",
]:
    if column in cohort.columns:
        cohort = cohort.drop(columns=column)

cohort = cohort.merge(diagnosis_flags, on="hadm_id", how="left", validate="one_to_one")

cohort["sepsis_diagnosis"] = cohort["sepsis_diagnosis"].fillna(False).astype(bool)

cohort["methemoglobinemia_diagnosis"] = (
    cohort["methemoglobinemia_diagnosis"].fillna(False).astype(bool)
)

print("\nDiagnóstico de sepse:")
print(cohort["sepsis_diagnosis"].value_counts(dropna=False))

print("\nDiagnóstico de metemoglobinemia:")
print(cohort["methemoglobinemia_diagnosis"].value_counts(dropna=False))

# ------------------------------------------------------------
# 10. Mostrar os códigos de sepse encontrados
# ------------------------------------------------------------

sepsis_codes = (
    diagnoses[diagnoses["sepsis_diagnosis"]]
    .groupby(["icd_code_normalized", "icd_version_numeric", "long_title"], dropna=False)
    .agg(
        n_records=("hadm_id", "size"),
        n_admissions=("hadm_id", "nunique"),
        n_patients=("subject_id", "nunique"),
    )
    .reset_index()
    .sort_values(["n_patients", "n_records"], ascending=[False, False])
    .reset_index(drop=True)
)

print("\n" + "=" * 90)
print("CÓDIGOS DE SEPSE ENCONTRADOS")
print("=" * 90)

display(sepsis_codes)

# ------------------------------------------------------------
# 11. Identificar itemids de norepinefrina
# ------------------------------------------------------------

d_items = pd.read_csv(D_ITEMS_PATH, compression="gzip", low_memory=False)

text_columns = [
    column
    for column in ["label", "abbreviation", "category", "unitname", "linksto"]
    if column in d_items.columns
]

d_items["_search_text"] = (
    d_items[text_columns].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
)

norepinephrine_mask = d_items["_search_text"].str.contains(
    r"\bnorepinephrine\b|\bnoradrenaline\b", regex=True, na=False
) & d_items["_search_text"].str.contains(r"inputevents", regex=True, na=False)

norepinephrine_items = d_items.loc[norepinephrine_mask].copy()

item_display_columns = [
    column
    for column in [
        "itemid",
        "label",
        "abbreviation",
        "category",
        "unitname",
        "linksto",
        "param_type",
    ]
    if column in norepinephrine_items.columns
]

norepinephrine_items = (
    norepinephrine_items[item_display_columns]
    .drop_duplicates()
    .sort_values("itemid")
    .reset_index(drop=True)
)

if norepinephrine_items.empty:
    raise RuntimeError("Nenhum itemid de norepinefrina foi identificado.")

norepinephrine_itemids = set(
    pd.to_numeric(norepinephrine_items["itemid"], errors="coerce")
    .dropna()
    .astype(int)
    .tolist()
)

print("\n" + "=" * 90)
print("ITEMIDS DE NOREPINEFRINA")
print("=" * 90)

display(norepinephrine_items)

print("\nItemids usados:", sorted(norepinephrine_itemids))

# ------------------------------------------------------------
# 12. Preparar os pacientes-alvo
# ------------------------------------------------------------

target_stay_ids = set(cohort["stay_id"].dropna().astype(int).tolist())

target_subject_ids = set(cohort["subject_id"].dropna().astype(int).tolist())

target_hadm_ids = set(cohort["hadm_id"].dropna().astype(int).tolist())

# ------------------------------------------------------------
# 13. Inspecionar cabeçalho do inputevents
# ------------------------------------------------------------

input_header = pd.read_csv(INPUTEVENTS_PATH, compression="gzip", nrows=0)

input_columns = input_header.columns.tolist()

preferred_input_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "storetime",
    "itemid",
    "amount",
    "amountuom",
    "rate",
    "rateuom",
    "orderid",
    "linkorderid",
    "ordercategoryname",
    "secondaryordercategoryname",
    "ordercomponenttypedescription",
    "ordercategorydescription",
    "patientweight",
    "statusdescription",
    "originalamount",
    "originalrate",
]

input_usecols = [
    column for column in preferred_input_columns if column in input_columns
]

required_input_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "itemid",
]

missing_input_columns = [
    column for column in required_input_columns if column not in input_usecols
]

if missing_input_columns:
    raise RuntimeError(
        "Colunas obrigatórias ausentes em inputevents: " f"{missing_input_columns}"
    )

print("\nColunas lidas de inputevents:")
print(input_usecols)

# ------------------------------------------------------------
# 14. Ler inputevents em chunks
# ------------------------------------------------------------

chunk_size = 300_000

norepinephrine_chunks = []

total_rows_read = 0
total_target_records = 0

print("\nIniciando leitura de inputevents em chunks...")
print(f"Chunk size: {chunk_size:,}")

reader = pd.read_csv(
    INPUTEVENTS_PATH,
    compression="gzip",
    usecols=input_usecols,
    chunksize=chunk_size,
    low_memory=False,
)

for chunk_number, chunk in enumerate(reader, start=1):

    total_rows_read += len(chunk)

    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")

    # Primeiro filtrar pelos poucos itemids de norepinefrina
    item_mask = chunk["itemid"].isin(norepinephrine_itemids)

    if item_mask.any():

        norepinephrine_chunk = chunk.loc[item_mask].copy()

        # Depois filtrar pelas 190 estadias de interesse
        norepinephrine_chunk = norepinephrine_chunk[
            norepinephrine_chunk["stay_id"].isin(target_stay_ids)
        ].copy()

        if not norepinephrine_chunk.empty:

            norepinephrine_chunks.append(norepinephrine_chunk)

            total_target_records += len(norepinephrine_chunk)

    print(
        f"Chunk {chunk_number:03d} | "
        f"linhas lidas: {total_rows_read:,} | "
        f"registros-alvo acumulados: {total_target_records:,}"
    )

    del chunk
    gc.collect()

# ------------------------------------------------------------
# 15. Consolidar registros de norepinefrina
# ------------------------------------------------------------

if norepinephrine_chunks:

    norepinephrine_events = pd.concat(norepinephrine_chunks, ignore_index=True)

else:

    norepinephrine_events = pd.DataFrame(columns=input_usecols)

del norepinephrine_chunks
gc.collect()

print("\nRegistros de norepinefrina encontrados:")
print(f"{len(norepinephrine_events):,}")

if not norepinephrine_events.empty:

    print(
        "Pacientes com algum registro de norepinefrina:",
        norepinephrine_events["subject_id"].nunique(),
    )

    print(
        "UTIs com algum registro de norepinefrina:",
        norepinephrine_events["stay_id"].nunique(),
    )

# ------------------------------------------------------------
# 16. Adicionar rótulo do d_items
# ------------------------------------------------------------

item_dictionary = (
    d_items[
        [
            column
            for column in ["itemid", "label", "abbreviation", "unitname"]
            if column in d_items.columns
        ]
    ]
    .drop_duplicates(subset=["itemid"])
    .copy()
)

item_dictionary["itemid"] = pd.to_numeric(item_dictionary["itemid"], errors="coerce")

norepinephrine_events = norepinephrine_events.merge(
    item_dictionary, on="itemid", how="left"
)

# ------------------------------------------------------------
# 17. Converter horários e valores numéricos
# ------------------------------------------------------------

for column in ["starttime", "endtime", "storetime"]:

    if column in norepinephrine_events.columns:

        norepinephrine_events[column] = pd.to_datetime(
            norepinephrine_events[column], errors="coerce"
        )

if "rate" in norepinephrine_events.columns:

    norepinephrine_events["rate_numeric"] = pd.to_numeric(
        norepinephrine_events["rate"], errors="coerce"
    )

else:

    norepinephrine_events["rate_numeric"] = np.nan

# ------------------------------------------------------------
# 18. Vincular o tempo do azul aos registros de norepinefrina
# ------------------------------------------------------------

mb_times = cohort[["subject_id", "hadm_id", "stay_id", "mb_time_preliminary"]].copy()

norepinephrine_events = norepinephrine_events.merge(
    mb_times,
    on=["subject_id", "hadm_id", "stay_id"],
    how="left",
    validate="many_to_one",
)

# ------------------------------------------------------------
# 19. Definir norepinefrina ativa no momento do azul
# ------------------------------------------------------------

norepinephrine_events["temporal_overlap_with_mb"] = (
    norepinephrine_events["starttime"].notna()
    & norepinephrine_events["endtime"].notna()
    & norepinephrine_events["mb_time_preliminary"].notna()
    & (
        norepinephrine_events["starttime"]
        <= norepinephrine_events["mb_time_preliminary"]
    )
    & (norepinephrine_events["endtime"] >= norepinephrine_events["mb_time_preliminary"])
)

# Definição estrita:
# - intervalo temporal cobre o momento do azul
# - taxa de infusão positiva

norepinephrine_events["active_norepinephrine_strict"] = (
    norepinephrine_events["temporal_overlap_with_mb"]
    & norepinephrine_events["rate_numeric"].notna()
    & (norepinephrine_events["rate_numeric"] > 0)
)

# Definição ampla para controle de qualidade:
# qualquer registro temporalmente sobreposto,
# mesmo que a taxa esteja ausente.

norepinephrine_events["active_norepinephrine_broad"] = norepinephrine_events[
    "temporal_overlap_with_mb"
]

# ------------------------------------------------------------
# 20. Calcular distância temporal
# ------------------------------------------------------------

norepinephrine_events["hours_ne_start_to_mb"] = (
    norepinephrine_events["mb_time_preliminary"] - norepinephrine_events["starttime"]
).dt.total_seconds() / 3600

norepinephrine_events["hours_mb_to_ne_end"] = (
    norepinephrine_events["endtime"] - norepinephrine_events["mb_time_preliminary"]
).dt.total_seconds() / 3600

# ------------------------------------------------------------
# 21. Resumir norepinefrina no momento do azul por paciente
# ------------------------------------------------------------

active_events = norepinephrine_events[
    norepinephrine_events["active_norepinephrine_strict"]
].copy()

broad_active_events = norepinephrine_events[
    norepinephrine_events["active_norepinephrine_broad"]
].copy()

# Se houver mais de um registro simultâneo, manteremos:
# - número de registros ativos
# - taxa máxima
# - taxa somada
# - registro que começou mais recentemente

if active_events.empty:

    ne_summary = pd.DataFrame(
        columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "n_active_ne_records",
            "ne_rate_max_raw",
            "ne_rate_sum_raw",
            "ne_rate_unit_at_mb",
            "ne_itemid_at_mb",
            "ne_starttime_at_mb",
            "ne_endtime_at_mb",
        ]
    )

else:

    ne_aggregate = active_events.groupby(
        ["subject_id", "hadm_id", "stay_id"], as_index=False
    ).agg(
        n_active_ne_records=("itemid", "size"),
        ne_rate_max_raw=("rate_numeric", "max"),
        ne_rate_sum_raw=("rate_numeric", "sum"),
    )

    most_recent_active = (
        active_events.sort_values(
            ["subject_id", "mb_time_preliminary", "starttime"],
            ascending=[True, True, False],
        )
        .drop_duplicates(subset=["subject_id", "hadm_id", "stay_id"], keep="first")[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "itemid",
                "label",
                "starttime",
                "endtime",
                "rate_numeric",
                "rateuom",
                "patientweight",
                "statusdescription",
            ]
        ]
        .rename(
            columns={
                "itemid": "ne_itemid_at_mb",
                "label": "ne_label_at_mb",
                "starttime": "ne_starttime_at_mb",
                "endtime": "ne_endtime_at_mb",
                "rate_numeric": "ne_rate_most_recent_raw",
                "rateuom": "ne_rate_unit_at_mb",
                "patientweight": "ne_patientweight_at_mb",
                "statusdescription": "ne_status_at_mb",
            }
        )
    )

    ne_summary = ne_aggregate.merge(
        most_recent_active,
        on=["subject_id", "hadm_id", "stay_id"],
        how="left",
        validate="one_to_one",
    )

# ------------------------------------------------------------
# 22. Criar flags por paciente
# ------------------------------------------------------------

strict_active_ids = set(active_events["subject_id"].dropna().astype(int).tolist())

broad_active_ids = set(broad_active_events["subject_id"].dropna().astype(int).tolist())

cohort["norepinephrine_active_at_mb"] = cohort["subject_id"].isin(strict_active_ids)

cohort["norepinephrine_temporal_overlap_at_mb"] = cohort["subject_id"].isin(
    broad_active_ids
)

cohort = cohort.merge(
    ne_summary,
    on=["subject_id", "hadm_id", "stay_id"],
    how="left",
    validate="one_to_one",
)

# ------------------------------------------------------------
# 23. Definir critérios de inclusão
# ------------------------------------------------------------

cohort["eligible_sepsis"] = cohort["sepsis_diagnosis"]

cohort["eligible_norepinephrine"] = cohort["norepinephrine_active_at_mb"]

cohort["eligible_no_methemoglobinemia"] = ~cohort["methemoglobinemia_diagnosis"]

cohort["primary_cohort_eligible"] = (
    cohort["eligible_sepsis"]
    & cohort["eligible_norepinephrine"]
    & cohort["eligible_no_methemoglobinemia"]
)

# ------------------------------------------------------------
# 24. Construir coortes intermediárias
# ------------------------------------------------------------

sepsis_cohort = cohort[cohort["eligible_sepsis"]].copy()

sepsis_ne_cohort = cohort[
    cohort["eligible_sepsis"] & cohort["eligible_norepinephrine"]
].copy()

primary_cohort = cohort[cohort["primary_cohort_eligible"]].copy()

# ------------------------------------------------------------
# 25. Fluxo de seleção
# ------------------------------------------------------------

n_broad = cohort["subject_id"].nunique()

n_sepsis = sepsis_cohort["subject_id"].nunique()

n_sepsis_ne = sepsis_ne_cohort["subject_id"].nunique()

n_methemoglobinemia_among_sepsis_ne = sepsis_ne_cohort.loc[
    sepsis_ne_cohort["methemoglobinemia_diagnosis"], "subject_id"
].nunique()

n_primary = primary_cohort["subject_id"].nunique()

selection_flow = pd.DataFrame(
    {
        "selection_step": [
            "IV methylene blue during ICU",
            "Diagnosis of sepsis",
            "Sepsis with active norepinephrine at MB time",
            "Methemoglobinemia excluded",
            "Primary cohort",
        ],
        "n_patients": [
            n_broad,
            n_sepsis,
            n_sepsis_ne,
            n_methemoglobinemia_among_sepsis_ne,
            n_primary,
        ],
        "interpretation": [
            "Broad exposure cohort",
            "Sepsis diagnosis present",
            "Norepinephrine infusion overlapping MB time with positive rate",
            "Number removed from sepsis + norepinephrine cohort",
            "Sepsis + active norepinephrine + no methemoglobinemia",
        ],
    }
)

# ------------------------------------------------------------
# 26. Resumo das unidades de norepinefrina
# ------------------------------------------------------------

if active_events.empty:

    ne_unit_summary = pd.DataFrame()

else:

    ne_unit_summary = (
        active_events.groupby(["rateuom", "itemid", "label"], dropna=False)
        .agg(
            n_records=("subject_id", "size"),
            n_patients=("subject_id", "nunique"),
            rate_median=("rate_numeric", "median"),
            rate_p25=("rate_numeric", lambda x: x.quantile(0.25)),
            rate_p75=("rate_numeric", lambda x: x.quantile(0.75)),
            rate_min=("rate_numeric", "min"),
            rate_max=("rate_numeric", "max"),
        )
        .reset_index()
        .sort_values(["n_patients", "n_records"], ascending=[False, False])
    )

# ------------------------------------------------------------
# 27. Registros com sobreposição mas taxa ausente/não positiva
# ------------------------------------------------------------

overlap_not_strict = norepinephrine_events[
    norepinephrine_events["active_norepinephrine_broad"]
    & ~norepinephrine_events["active_norepinephrine_strict"]
].copy()

overlap_not_strict_summary = (
    overlap_not_strict.groupby(["rateuom", "statusdescription"], dropna=False)
    .agg(
        n_records=("subject_id", "size"),
        n_patients=("subject_id", "nunique"),
        rate_nonmissing=("rate_numeric", lambda x: x.notna().sum()),
    )
    .reset_index()
    .sort_values("n_records", ascending=False)
)

# ------------------------------------------------------------
# 28. Resumo preliminar da coorte principal
# ------------------------------------------------------------


def continuous_summary(data, variable, label):

    x = pd.to_numeric(data[variable], errors="coerce")

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


continuous_variables = [
    ("age_at_admission", "Age, years"),
    ("dose_mg_clean", "Methylene blue dose, mg"),
    ("hours_icu_to_mb", "Time from ICU admission to MB, hours"),
    ("ne_rate_most_recent_raw", "Norepinephrine rate at MB, raw units"),
    ("icu_los_days_calculated", "ICU length of stay, days"),
    ("hospital_los_days", "Hospital length of stay, days"),
]

primary_continuous_rows = []

for variable, label in continuous_variables:

    if variable in primary_cohort.columns:

        primary_continuous_rows.append(
            continuous_summary(primary_cohort, variable, label)
        )

primary_continuous_summary = pd.DataFrame(primary_continuous_rows)

# ------------------------------------------------------------
# 29. Variáveis categóricas preliminares da Tabela 1
# ------------------------------------------------------------


def categorical_summary(data, variable, variable_label):

    summary = (
        data[variable]
        .fillna("Missing")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("category")
        .reset_index(name="n")
    )

    summary["percent"] = 100 * summary["n"] / len(data) if len(data) > 0 else np.nan

    summary.insert(0, "variable", variable_label)

    return summary


categorical_tables = []

for variable, label in [
    ("sex", "Sex"),
    ("first_careunit", "ICU type"),
    ("admission_type", "Admission type"),
    ("race", "Race"),
    ("ne_rate_unit_at_mb", "Norepinephrine rate unit at MB"),
]:

    if variable in primary_cohort.columns:

        categorical_tables.append(categorical_summary(primary_cohort, variable, label))

if categorical_tables:

    primary_categorical_summary = pd.concat(categorical_tables, ignore_index=True)

else:

    primary_categorical_summary = pd.DataFrame()

# ------------------------------------------------------------
# 30. Mortalidade preliminar
# ------------------------------------------------------------

outcome_rows = []

for variable, label in [
    ("hospital_mortality", "Hospital mortality"),
    ("icu_mortality", "ICU mortality"),
]:

    if variable in primary_cohort.columns:

        values = pd.to_numeric(primary_cohort[variable], errors="coerce")

        denominator = int(values.notna().sum())

        numerator = int((values == 1).sum())

        outcome_rows.append(
            {
                "variable": label,
                "n_positive": numerator,
                "n_available": denominator,
                "percent": (
                    100 * numerator / denominator if denominator > 0 else np.nan
                ),
            }
        )

primary_outcome_summary = pd.DataFrame(outcome_rows)

# ------------------------------------------------------------
# 31. Mostrar resultados
# ------------------------------------------------------------

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 180)
pd.set_option("display.width", 280)

print("\n" + "=" * 90)
print("FLUXO DE SELEÇÃO")
print("=" * 90)

display(selection_flow)

print("\n" + "=" * 90)
print("UNIDADES E TAXAS DE NOREPINEFRINA NO MOMENTO DO AZUL")
print("=" * 90)

display(ne_unit_summary)

print("\n" + "=" * 90)
print("SOBREPOSIÇÃO TEMPORAL SEM TAXA POSITIVA")
print("=" * 90)

if overlap_not_strict_summary.empty:
    print("Nenhum registro com sobreposição temporal " "e taxa ausente/não positiva.")
else:
    display(overlap_not_strict_summary)

print("\n" + "=" * 90)
print("RESUMO CONTÍNUO PRELIMINAR DA COORTE PRINCIPAL")
print("=" * 90)

display(primary_continuous_summary)

print("\n" + "=" * 90)
print("RESUMO CATEGÓRICO PRELIMINAR DA COORTE PRINCIPAL")
print("=" * 90)

display(primary_categorical_summary)

print("\n" + "=" * 90)
print("DESFECHOS PRELIMINARES")
print("=" * 90)

display(primary_outcome_summary)

# ------------------------------------------------------------
# 32. Controle de qualidade final
# ------------------------------------------------------------

final_qc = pd.DataFrame(
    {
        "metric": [
            "Broad MB IV ICU cohort",
            "Patients with sepsis diagnosis",
            "Patients with any norepinephrine record",
            "Patients with norepinephrine temporal overlap",
            "Patients with strict active norepinephrine",
            "Sepsis + active norepinephrine",
            "Methemoglobinemia among sepsis + active norepinephrine",
            "Primary cohort",
            "Unique admissions in primary cohort",
            "Unique ICU stays in primary cohort",
            "MB time available in primary cohort",
            "Norepinephrine rate available in primary cohort",
            "Norepinephrine unit available in primary cohort",
        ],
        "value": [
            cohort["subject_id"].nunique(),
            cohort.loc[cohort["sepsis_diagnosis"], "subject_id"].nunique(),
            norepinephrine_events["subject_id"].nunique(),
            cohort.loc[
                cohort["norepinephrine_temporal_overlap_at_mb"], "subject_id"
            ].nunique(),
            cohort.loc[cohort["norepinephrine_active_at_mb"], "subject_id"].nunique(),
            sepsis_ne_cohort["subject_id"].nunique(),
            n_methemoglobinemia_among_sepsis_ne,
            primary_cohort["subject_id"].nunique(),
            primary_cohort["hadm_id"].nunique(),
            primary_cohort["stay_id"].nunique(),
            primary_cohort["mb_time_preliminary"].notna().sum(),
            primary_cohort["ne_rate_most_recent_raw"].notna().sum(),
            primary_cohort["ne_rate_unit_at_mb"].notna().sum(),
        ],
    }
)

print("\n" + "=" * 90)
print("CONTROLE DE QUALIDADE FINAL")
print("=" * 90)

display(final_qc)

# ------------------------------------------------------------
# 33. Salvar outputs
# ------------------------------------------------------------

timestamp = run_stamp()

broad_flags_path = (
    COHORT_DIR / f"methylene_blue_broad_cohort_sepsis_ne_flags_{timestamp}.csv.gz"
)

norepinephrine_events_path = (
    COHORT_DIR / f"methylene_blue_norepinephrine_events_{timestamp}.csv.gz"
)

active_ne_path = (
    COHORT_DIR / f"methylene_blue_active_norepinephrine_at_mb_{timestamp}.csv.gz"
)

sepsis_path = COHORT_DIR / f"methylene_blue_sepsis_cohort_{timestamp}.csv.gz"

sepsis_ne_path = (
    COHORT_DIR
    / f"methylene_blue_sepsis_active_norepinephrine_cohort_{timestamp}.csv.gz"
)


# Este é o arquivo que será enriquecido e usado na Tabela 1.
table1_base_path = (
    COHORT_DIR / f"TABLE1_BASE_methylene_blue_primary_cohort_{timestamp}.csv.gz"
)

selection_flow_path = (
    TABLE_DIR / f"methylene_blue_primary_cohort_selection_flow_{timestamp}.csv"
)

sepsis_codes_path = TABLE_DIR / f"methylene_blue_sepsis_codes_used_{timestamp}.csv"

ne_items_path = METADATA_DIR / f"norepinephrine_itemids_{timestamp}.csv"

ne_unit_summary_path = QC_DIR / f"norepinephrine_units_at_mb_{timestamp}.csv"

overlap_qc_path = (
    QC_DIR / f"norepinephrine_overlap_without_positive_rate_{timestamp}.csv"
)

continuous_path = TABLE_DIR / f"TABLE1_preliminary_continuous_{timestamp}.csv"

categorical_path = TABLE_DIR / f"TABLE1_preliminary_categorical_{timestamp}.csv"

outcomes_path = (
    TABLE_DIR / f"methylene_blue_primary_outcomes_preliminary_{timestamp}.csv"
)

final_qc_path = QC_DIR / f"methylene_blue_primary_cohort_qc_{timestamp}.csv"

report_path = QC_DIR / f"methylene_blue_code8_primary_cohort_report_{timestamp}.txt"

cohort.to_csv(broad_flags_path, index=False, compression="gzip")

norepinephrine_events.to_csv(
    norepinephrine_events_path, index=False, compression="gzip"
)

active_events.to_csv(active_ne_path, index=False, compression="gzip")

sepsis_cohort.to_csv(sepsis_path, index=False, compression="gzip")

sepsis_ne_cohort.to_csv(sepsis_ne_path, index=False, compression="gzip")


primary_cohort.to_csv(table1_base_path, index=False, compression="gzip")

selection_flow.to_csv(selection_flow_path, index=False)

sepsis_codes.to_csv(sepsis_codes_path, index=False)

norepinephrine_items.to_csv(ne_items_path, index=False)

ne_unit_summary.to_csv(ne_unit_summary_path, index=False)

overlap_not_strict.to_csv(overlap_qc_path, index=False)

primary_continuous_summary.to_csv(continuous_path, index=False)

primary_categorical_summary.to_csv(categorical_path, index=False)

primary_outcome_summary.to_csv(outcomes_path, index=False)

final_qc.to_csv(final_qc_path, index=False)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("METHYLENE BLUE PRIMARY COHORT\n")

    file.write("=" * 70 + "\n\n")

    file.write("Definition:\n")

    file.write(
        "Adult ICU patients with intravenous methylene blue, "
        "a diagnosis of sepsis, active norepinephrine infusion "
        "at the preliminary methylene blue time, and no diagnosis "
        "of methemoglobinemia.\n\n"
    )

    file.write(f"Broad MB cohort: {n_broad}\n")

    file.write(f"Sepsis: {n_sepsis}\n")

    file.write(f"Sepsis + active norepinephrine: {n_sepsis_ne}\n")

    file.write(
        "Methemoglobinemia among sepsis + norepinephrine: "
        f"{n_methemoglobinemia_among_sepsis_ne}\n"
    )

    file.write(f"Primary cohort: {n_primary}\n\n")

    file.write(f"Table 1 base file: {table1_base_path}\n")

print("\n" + "=" * 90)
print("OUTPUTS SALVOS")
print("=" * 90)

print(table1_base_path)
print(table1_base_path)
print(selection_flow_path)
print(sepsis_codes_path)
print(ne_items_path)
print(ne_unit_summary_path)
print(continuous_path)
print(categorical_path)
print(outcomes_path)
print(final_qc_path)
print(report_path)

gc.collect()

print("\nCÓDIGO 8 CONCLUÍDO COM SUCESSO.")
