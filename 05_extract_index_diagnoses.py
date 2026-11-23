"""Source: original notebook cell 26 (diagnosis extract only). Run through run.py."""

from common import artifact, mimic_source, phase_dir, run_stamp, source_file

# ============================================================
# CÓDIGO 7 — CARACTERIZAÇÃO INICIAL DA COORTE MIMIC-IV
#
# Coorte:
#   Primeira prescrição IV plausível de azul de metileno
#   iniciada durante uma estadia de UTI.
#
# Objetivos:
# 1. Adicionar idade, sexo e características da internação
# 2. Adicionar unidade e duração da UTI
# 3. Calcular mortalidade hospitalar e na UTI
# 4. Descrever dose e tempo até o azul de metileno
# 5. Identificar diagnósticos potencialmente relacionados
#    a metemoglobinemia
# 6. Gerar resumo preliminar da população
#
# IMPORTANTE:
# - O horário ainda deriva de prescriptions.starttime.
# - A administração real será validada posteriormente com eMAR.

# ============================================================

import pandas as pd
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

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Pasta-fonte não encontrada: {SOURCE_DIR}")

print("=" * 90)
print("CÓDIGO 7 — CARACTERIZAÇÃO INICIAL DA COORTE")
print("=" * 90)

print("\nFonte, somente leitura:")
print(SOURCE_DIR)

print("\nDiretório de outputs:")
print(PROJECT_DIR)

# ------------------------------------------------------------

DIAGNOSES_PATH = source_file(SOURCE_DIR, "diagnoses_icd.csv.gz")
D_ICD_DIAGNOSES_PATH = source_file(SOURCE_DIR, "d_icd_diagnoses.csv.gz")
INDEX_PATH = artifact(COHORT_DIR, "methylene_blue_preliminary_index_exposure_*.csv.gz")
# ------------------------------------------------------------
# 5. Carregar a exposição índice
# ------------------------------------------------------------

index_exposure = pd.read_csv(INDEX_PATH, compression="gzip", low_memory=False)

if index_exposure.empty:
    raise RuntimeError("O arquivo de exposição índice está vazio.")

required_index_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "dose_mg_clean",
    "mb_time_preliminary",
]

missing_index_columns = [
    col for col in required_index_columns if col not in index_exposure.columns
]

if missing_index_columns:
    raise RuntimeError(
        "Colunas ausentes na exposição índice: " f"{missing_index_columns}"
    )

index_exposure["mb_time_preliminary"] = pd.to_datetime(
    index_exposure["mb_time_preliminary"], errors="coerce"
)

print("\nCoorte índice carregada:")
print(f"Pacientes: {index_exposure['subject_id'].nunique():,}")
print(f"Internações: {index_exposure['hadm_id'].nunique():,}")
print(f"Estadias de UTI: {index_exposure['stay_id'].nunique():,}")

# ------------------------------------------------------------
# 6. Manter uma linha por paciente
# ------------------------------------------------------------

index_exposure = (
    index_exposure.sort_values(["subject_id", "mb_time_preliminary"])
    .drop_duplicates(subset=["subject_id"], keep="first")
    .copy()
)

target_subject_ids = set(index_exposure["subject_id"].dropna().astype(int))

target_hadm_ids = set(index_exposure["hadm_id"].dropna().astype(int))

target_stay_ids = set(index_exposure["stay_id"].dropna().astype(int))

# ------------------------------------------------------------
# 17. Ler diagnoses_icd em chunks
# ------------------------------------------------------------

diagnosis_header = pd.read_csv(DIAGNOSES_PATH, compression="gzip", nrows=0)

diagnosis_preferred_columns = [
    "subject_id",
    "hadm_id",
    "seq_num",
    "icd_code",
    "icd_version",
]

diagnosis_usecols = [
    col for col in diagnosis_preferred_columns if col in diagnosis_header.columns
]

diagnosis_chunks = []
chunk_size = 500_000
total_diagnosis_rows = 0

print("\nLendo diagnoses_icd em chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DIAGNOSES_PATH,
        compression="gzip",
        usecols=diagnosis_usecols,
        chunksize=chunk_size,
        low_memory=False,
    ),
    start=1,
):

    total_diagnosis_rows += len(chunk)

    matched = chunk[chunk["hadm_id"].isin(target_hadm_ids)].copy()

    if not matched.empty:
        diagnosis_chunks.append(matched)

    print(
        f"Chunk {chunk_number:03d} | "
        f"linhas lidas: {total_diagnosis_rows:,} | "
        f"registros da coorte acumulados: "
        f"{sum(len(x) for x in diagnosis_chunks):,}"
    )

    del chunk
    del matched
    gc.collect()

if diagnosis_chunks:
    cohort_diagnoses = pd.concat(diagnosis_chunks, ignore_index=True)
else:
    cohort_diagnoses = pd.DataFrame(columns=diagnosis_usecols)

del diagnosis_chunks
gc.collect()

print("\nDiagnósticos da coorte encontrados:", len(cohort_diagnoses))

# ------------------------------------------------------------
# 18. Carregar d_icd_diagnoses
# ------------------------------------------------------------

dictionary = pd.read_csv(D_ICD_DIAGNOSES_PATH, compression="gzip", low_memory=False)

dictionary_columns = [
    col
    for col in ["icd_code", "icd_version", "long_title"]
    if col in dictionary.columns
]

dictionary = dictionary[dictionary_columns].drop_duplicates(
    subset=["icd_code", "icd_version"]
)

cohort_diagnoses = cohort_diagnoses.merge(
    dictionary, on=["icd_code", "icd_version"], how="left"
)

# ------------------------------------------------------------
# 19. Normalizar o texto diagnóstico
# ------------------------------------------------------------

cohort_diagnoses["_diagnosis_text"] = (
    cohort_diagnoses["long_title"].fillna("").astype(str).str.lower()
)

timestamp = run_stamp()
cohort_diagnoses.to_csv(
    COHORT_DIR / f"methylene_blue_cohort_diagnoses_{timestamp}.csv.gz",
    index=False,
    compression="gzip",
)
