"""Source: original notebook cell 24. Run through run.py."""

from common import artifact, display, mimic_source, phase_dir, run_stamp

# ============================================================
# CÓDIGO 5 — VALIDAÇÃO DO AZUL DE METILENO EM PHARMACY
#
# Fonte:

#
# Relação principal:
#   pharmacy.pharmacy_id ↔ prescriptions.pharmacy_id
#
# Outputs:
#   HIRID/.../MIMIC_Phase
#
# Leitura em chunks para baixo consumo de RAM
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
METADATA_DIR = PROJECT_DIR / "metadata"

for folder in [COHORT_DIR, QC_DIR, METADATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

PHARMACY_PATH = SOURCE_DIR / "pharmacy.csv.gz"

if not PHARMACY_PATH.exists():
    candidates = list(SOURCE_DIR.rglob("pharmacy.csv.gz"))

    if len(candidates) == 1:
        PHARMACY_PATH = candidates[0]
    elif len(candidates) == 0:
        raise FileNotFoundError(
            "pharmacy.csv.gz não foi encontrado dentro de MIMIC-IV source."
        )
    else:
        print("Foram encontrados vários arquivos pharmacy.csv.gz:")
        for candidate in candidates:
            print(candidate)

        raise RuntimeError("Defina manualmente PHARMACY_PATH.")

print("Arquivo pharmacy:")
print(PHARMACY_PATH)

print("Tamanho compactado:", f"{PHARMACY_PATH.stat().st_size / (1024 ** 3):.2f} GB")

# ------------------------------------------------------------
# 2. Localizar o resultado mais recente de prescriptions
# ------------------------------------------------------------

PRESCRIPTION_HITS_PATH = artifact(
    COHORT_DIR, "methylene_blue_prescriptions_hits_*.csv.gz"
)

print("\nArquivo de prescriptions utilizado:")
print(PRESCRIPTION_HITS_PATH)

# ------------------------------------------------------------
# 3. Carregar apenas os hits já filtrados
# ------------------------------------------------------------

prescription_hits = pd.read_csv(
    PRESCRIPTION_HITS_PATH, compression="gzip", low_memory=False
)

print("\nRegistros de prescriptions carregados:", f"{len(prescription_hits):,}")

print("Pacientes:", f"{prescription_hits['subject_id'].nunique():,}")

if "pharmacy_id" not in prescription_hits.columns:
    raise RuntimeError(
        "A coluna pharmacy_id não está presente nos hits de prescriptions."
    )

prescription_hits["pharmacy_id"] = pd.to_numeric(
    prescription_hits["pharmacy_id"], errors="coerce"
).astype("Int64")

target_pharmacy_ids = set(
    prescription_hits["pharmacy_id"].dropna().astype(int).tolist()
)

print("Pharmacy IDs únicos a procurar:", f"{len(target_pharmacy_ids):,}")

# ------------------------------------------------------------
# 4. Inspecionar cabeçalho do pharmacy
# ------------------------------------------------------------

pharmacy_header = pd.read_csv(PHARMACY_PATH, compression="gzip", nrows=0)

all_columns = pharmacy_header.columns.tolist()

print("\nColunas disponíveis no pharmacy:")
print(all_columns)

# ------------------------------------------------------------
# 5. Selecionar colunas necessárias
# ------------------------------------------------------------

preferred_columns = [
    "subject_id",
    "hadm_id",
    "pharmacy_id",
    "poe_id",
    "starttime",
    "stoptime",
    "medication",
    "proc_type",
    "status",
    "entertime",
    "verifiedtime",
    "route",
    "frequency",
    "disp_sched",
    "infusion_type",
    "sliding_scale",
    "lockout_interval",
    "basal_rate",
    "one_hr_max",
    "doses_per_24_hrs",
    "duration",
    "duration_interval",
    "expiration_value",
    "expiration_unit",
    "expirationdate",
    "dispensation",
    "fill_quantity",
]

usecols = [col for col in preferred_columns if col in all_columns]

if "pharmacy_id" not in usecols:
    raise RuntimeError("A coluna pharmacy_id não foi encontrada no pharmacy.")

# Colunas textuais realmente existentes
potential_text_columns = [
    "medication",
    "proc_type",
    "status",
    "route",
    "frequency",
    "disp_sched",
    "infusion_type",
    "dispensation",
]

text_columns = [col for col in potential_text_columns if col in usecols]

print("\nColunas que serão lidas:")
print(usecols)

print("\nColunas textuais pesquisadas:")
print(text_columns)

# ------------------------------------------------------------
# 6. Padrões de busca
# ------------------------------------------------------------

search_patterns = {
    "methylene_blue": r"\bmethylene\s+blu[e]?\b",
    "methylene": r"\bmethylene\b",
    "methylthioninium": r"\bmethylthioninium\b",
    "methylthionine": r"\bmethylthionine\b",
    "provayblue": r"\bprovay\s*blue\b|\bprovayblue\b",
}

combined_pattern = "|".join(f"(?:{pattern})" for pattern in search_patterns.values())

# ------------------------------------------------------------
# 7. Ler pharmacy em chunks
# ------------------------------------------------------------

chunk_size = 200_000

matched_chunks = []

total_rows = 0
total_linked = 0
total_text_matches = 0
total_retained = 0

print("\nIniciando leitura em chunks...")
print(f"Chunk size: {chunk_size:,}")

reader = pd.read_csv(
    PHARMACY_PATH,
    compression="gzip",
    usecols=usecols,
    chunksize=chunk_size,
    low_memory=False,
)

for chunk_number, chunk in enumerate(reader, start=1):

    total_rows += len(chunk)

    chunk["pharmacy_id"] = pd.to_numeric(chunk["pharmacy_id"], errors="coerce").astype(
        "Int64"
    )

    # Correspondência direta com prescriptions
    linked_mask = chunk["pharmacy_id"].isin(target_pharmacy_ids)

    # Busca textual independente, para detectar possíveis registros
    # que não tenham sido corretamente ligados
    if text_columns:
        search_text = (
            chunk[text_columns]
            .fillna("")
            .astype(str)
            .agg(" | ".join, axis=1)
            .str.lower()
        )

        text_mask = search_text.str.contains(combined_pattern, regex=True, na=False)
    else:
        search_text = pd.Series("", index=chunk.index, dtype="object")

        text_mask = pd.Series(False, index=chunk.index)

    retain_mask = linked_mask | text_mask

    n_linked = int(linked_mask.sum())
    n_text = int(text_mask.sum())
    n_retained = int(retain_mask.sum())

    total_linked += n_linked
    total_text_matches += n_text
    total_retained += n_retained

    if n_retained > 0:

        retained = chunk.loc[retain_mask].copy()

        retained["linked_to_prescriptions"] = linked_mask.loc[retain_mask].values

        retained["independent_text_match"] = text_mask.loc[retain_mask].values

        retained["_pharmacy_search_text"] = search_text.loc[retain_mask].values

        for name, pattern in search_patterns.items():
            retained[f"match_{name}"] = retained["_pharmacy_search_text"].str.contains(
                pattern, regex=True, na=False
            )

        matched_chunks.append(retained)

    print(
        f"Chunk {chunk_number:03d} | "
        f"linhas lidas: {total_rows:,} | "
        f"ligadas: {n_linked:,} | "
        f"texto: {n_text:,} | "
        f"retidas: {n_retained:,}"
    )

    del chunk
    del linked_mask
    del text_mask
    del retain_mask
    del search_text

    gc.collect()

# ------------------------------------------------------------
# 8. Consolidar resultados
# ------------------------------------------------------------

if matched_chunks:
    pharmacy_hits = pd.concat(matched_chunks, ignore_index=True)
else:
    pharmacy_hits = pd.DataFrame()

del matched_chunks
gc.collect()

print("\n" + "=" * 90)
print("RESULTADOS DA BUSCA EM PHARMACY")
print("=" * 90)

print(f"Linhas examinadas: {total_rows:,}")
print(f"Registros ligados por pharmacy_id: {total_linked:,}")
print(f"Registros encontrados por texto: {total_text_matches:,}")
print(f"Registros retidos: {len(pharmacy_hits):,}")

if not pharmacy_hits.empty:

    if "subject_id" in pharmacy_hits.columns:
        print("Pacientes únicos:", pharmacy_hits["subject_id"].nunique())

    if "hadm_id" in pharmacy_hits.columns:
        print("Internações únicas:", pharmacy_hits["hadm_id"].nunique())

    print("Pharmacy IDs únicos:", pharmacy_hits["pharmacy_id"].nunique())

# ------------------------------------------------------------
# 9. Identificar registros encontrados somente no pharmacy
# ------------------------------------------------------------

if not pharmacy_hits.empty:

    pharmacy_only = pharmacy_hits[
        (~pharmacy_hits["linked_to_prescriptions"])
        & pharmacy_hits["independent_text_match"]
    ].copy()

else:
    pharmacy_only = pd.DataFrame()

print(
    "\nRegistros encontrados por texto apenas no pharmacy:", f"{len(pharmacy_only):,}"
)

if not pharmacy_only.empty and "subject_id" in pharmacy_only.columns:
    print("Pacientes adicionais potenciais:", pharmacy_only["subject_id"].nunique())

# ------------------------------------------------------------
# 10. Resumo das características no pharmacy
# ------------------------------------------------------------

summary_columns = [
    col
    for col in [
        "medication",
        "route",
        "frequency",
        "proc_type",
        "status",
        "infusion_type",
        "dispensation",
        "duration",
        "duration_interval",
    ]
    if col in pharmacy_hits.columns
]

if not pharmacy_hits.empty and summary_columns:

    pharmacy_formats = (
        pharmacy_hits[summary_columns]
        .fillna("<missing>")
        .value_counts(dropna=False)
        .reset_index(name="n_records")
        .sort_values("n_records", ascending=False)
        .reset_index(drop=True)
    )

    print("\nFormatos encontrados no pharmacy:")
    display(pharmacy_formats.head(100))

else:
    pharmacy_formats = pd.DataFrame()

# ------------------------------------------------------------
# 11. Cruzar prescriptions e pharmacy
# ------------------------------------------------------------

pharmacy_merge_columns = [
    col
    for col in pharmacy_hits.columns
    if col not in ["subject_id", "hadm_id", "_pharmacy_search_text"]
]

if not pharmacy_hits.empty:

    pharmacy_for_merge = (
        pharmacy_hits[pharmacy_merge_columns]
        .drop_duplicates(subset=["pharmacy_id"])
        .copy()
    )

    combined = prescription_hits.merge(
        pharmacy_for_merge, on="pharmacy_id", how="left", suffixes=("_rx", "_pharm")
    )

else:
    combined = prescription_hits.copy()

print("\nRegistros combinados prescriptions + pharmacy:")
print(f"{len(combined):,}")

# ------------------------------------------------------------
# 12. Classificação preliminar por via
# ------------------------------------------------------------

route_source = None

for candidate in ["route_rx", "route", "route_pharm"]:
    if candidate in combined.columns:
        route_source = candidate
        break

if route_source is not None:

    combined["_route_normalized"] = (
        combined[route_source].fillna("").astype(str).str.upper().str.strip()
    )

    combined["preliminary_iv_route"] = combined["_route_normalized"].str.contains(
        r"(^IV$|IV DRIP|INTRAVENOUS)", regex=True, na=False
    )

else:

    combined["_route_normalized"] = ""
    combined["preliminary_iv_route"] = False

print("\nClassificação preliminar da via:")

route_summary = (
    combined.groupby(["_route_normalized", "preliminary_iv_route"], dropna=False)
    .agg(n_records=("subject_id", "size"), n_patients=("subject_id", "nunique"))
    .reset_index()
    .sort_values("n_records", ascending=False)
)

display(route_summary)

# ------------------------------------------------------------
# 13. Contagem preliminar da população IV
# ------------------------------------------------------------

iv_candidates = combined[combined["preliminary_iv_route"]].copy()

print("\n" + "=" * 90)
print("COORTE INTRAVENOSA PRELIMINAR")
print("=" * 90)

print(f"Registros IV: {len(iv_candidates):,}")

if not iv_candidates.empty:
    print("Pacientes IV únicos:", iv_candidates["subject_id"].nunique())

    if "hadm_id" in iv_candidates.columns:
        print("Internações IV únicas:", iv_candidates["hadm_id"].nunique())

# ------------------------------------------------------------
# 14. Distribuição preliminar das doses IV
# ------------------------------------------------------------

if not iv_candidates.empty and "dose_val_rx" in iv_candidates.columns:

    iv_candidates["_dose_numeric"] = pd.to_numeric(
        iv_candidates["dose_val_rx"], errors="coerce"
    )

    dose_unit_col = "dose_unit_rx" if "dose_unit_rx" in iv_candidates.columns else None

    if dose_unit_col:

        dose_summary = (
            iv_candidates.groupby(dose_unit_col, dropna=False)
            .agg(
                n_records=("subject_id", "size"),
                n_patients=("subject_id", "nunique"),
                dose_nonmissing=("_dose_numeric", lambda x: x.notna().sum()),
                dose_median=("_dose_numeric", "median"),
                dose_p25=("_dose_numeric", lambda x: x.quantile(0.25)),
                dose_p75=("_dose_numeric", lambda x: x.quantile(0.75)),
                dose_min=("_dose_numeric", "min"),
                dose_max=("_dose_numeric", "max"),
            )
            .reset_index()
        )

        print("\nDistribuição preliminar das doses IV:")
        display(dose_summary)

    else:
        dose_summary = pd.DataFrame()

else:
    dose_summary = pd.DataFrame()

# ------------------------------------------------------------
# 15. Salvar outputs
# ------------------------------------------------------------

timestamp = run_stamp()

pharmacy_hits_path = COHORT_DIR / f"methylene_blue_pharmacy_hits_{timestamp}.csv.gz"

pharmacy_only_path = QC_DIR / f"methylene_blue_pharmacy_only_hits_{timestamp}.csv"

formats_path = QC_DIR / f"methylene_blue_pharmacy_formats_{timestamp}.csv"

combined_path = (
    COHORT_DIR / f"methylene_blue_prescriptions_pharmacy_combined_{timestamp}.csv.gz"
)

iv_path = COHORT_DIR / f"methylene_blue_preliminary_iv_candidates_{timestamp}.csv.gz"

route_summary_path = QC_DIR / f"methylene_blue_route_summary_{timestamp}.csv"

dose_summary_path = (
    QC_DIR / f"methylene_blue_preliminary_iv_dose_summary_{timestamp}.csv"
)

report_path = QC_DIR / f"methylene_blue_pharmacy_validation_report_{timestamp}.txt"

pharmacy_hits.to_csv(pharmacy_hits_path, index=False, compression="gzip")

pharmacy_only.to_csv(pharmacy_only_path, index=False)

pharmacy_formats.to_csv(formats_path, index=False)

combined.to_csv(combined_path, index=False, compression="gzip")

iv_candidates.to_csv(iv_path, index=False, compression="gzip")

route_summary.to_csv(route_summary_path, index=False)

dose_summary.to_csv(dose_summary_path, index=False)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("METHYLENE BLUE PHARMACY VALIDATION\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Pharmacy source: {PHARMACY_PATH}\n")
    file.write(f"Prescription hits source: " f"{PRESCRIPTION_HITS_PATH}\n\n")

    file.write(f"Pharmacy rows examined: {total_rows}\n")
    file.write(f"Records linked through pharmacy_id: " f"{total_linked}\n")
    file.write(f"Independent text matches: " f"{total_text_matches}\n")
    file.write(f"Total pharmacy records retained: " f"{len(pharmacy_hits)}\n")
    file.write(f"Pharmacy-only text matches: " f"{len(pharmacy_only)}\n\n")

    file.write(f"Preliminary IV records: " f"{len(iv_candidates)}\n")

    if not iv_candidates.empty:
        file.write(
            f"Preliminary IV patients: " f"{iv_candidates['subject_id'].nunique()}\n"
        )

print("\nOutputs salvos:")
print(pharmacy_hits_path)
print(pharmacy_only_path)
print(formats_path)
print(combined_path)
print(iv_path)
print(route_summary_path)
print(dose_summary_path)
print(report_path)

print("\nCÓDIGO 5 CONCLUÍDO.")
