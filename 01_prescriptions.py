"""Source: original notebook cell 23. Run through run.py."""

from common import display, mimic_source, phase_dir, run_stamp

# ============================================================
# CÓDIGO 4 — MEDICATION FINDER EM PRESCRIPTIONS

# Output: HiRID_Methylene_Blue.../MIMIC_Phase
# Uso controlado de RAM
# ============================================================

import pandas as pd
import gc

# ------------------------------------------------------------
# 1. Caminhos
# ------------------------------------------------------------

SOURCE = mimic_source()

OUTPUT = phase_dir("mimic_iv")

QC_DIR = OUTPUT / "qc"
COHORT_DIR = OUTPUT / "cohorts"
METADATA_DIR = OUTPUT / "metadata"

for folder in [OUTPUT, QC_DIR, COHORT_DIR, METADATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(f"Pasta-fonte não encontrada: {SOURCE}")

print("Fonte MIMIC-IV, somente leitura:")
print(SOURCE)

print("\nPasta de outputs:")
print(OUTPUT)

# ------------------------------------------------------------
# 2. Localizar prescriptions.csv.gz
# ------------------------------------------------------------

direct_path = SOURCE / "prescriptions.csv.gz"

if direct_path.exists():
    PRESCRIPTIONS_PATH = direct_path
else:
    candidates = list(SOURCE.rglob("prescriptions.csv.gz"))

    if len(candidates) == 0:
        raise FileNotFoundError(
            "prescriptions.csv.gz não foi encontrado dentro de MIMIC-IV source."
        )

    if len(candidates) > 1:
        print("\nForam encontrados vários prescriptions.csv.gz:")
        for i, path in enumerate(candidates, start=1):
            print(f"{i}. {path}")

        raise RuntimeError(
            "Há mais de um arquivo prescriptions.csv.gz. "
            "Defina manualmente PRESCRIPTIONS_PATH."
        )

    PRESCRIPTIONS_PATH = candidates[0]

print("\nArquivo localizado:")
print(PRESCRIPTIONS_PATH)

file_size_gb = PRESCRIPTIONS_PATH.stat().st_size / (1024**3)
print(f"Tamanho compactado: {file_size_gb:.2f} GB")

# ------------------------------------------------------------
# 3. Ler apenas o cabeçalho
# ------------------------------------------------------------

header = pd.read_csv(PRESCRIPTIONS_PATH, compression="gzip", nrows=0)

all_columns = header.columns.tolist()

print("\nColunas disponíveis:")
print(all_columns)

# ------------------------------------------------------------
# 4. Selecionar colunas úteis
# ------------------------------------------------------------

id_time_columns = [
    "subject_id",
    "hadm_id",
    "pharmacy_id",
    "poe_id",
    "poe_seq",
    "order_provider_id",
    "starttime",
    "stoptime",
    "order_type",
    "order_status",
    "route",
    "frequency",
    "disp_sched",
    "doses_per_24_hrs",
    "duration",
    "duration_interval",
    "dose_val_rx",
    "dose_unit_rx",
    "form_val_disp",
    "form_unit_disp",
]

potential_text_columns = [
    "drug",
    "drug_name_poe",
    "drug_name_generic",
    "formulary_drug_cd",
    "gsn",
    "ndc",
    "prod_strength",
    "form_rx",
]

text_columns = [col for col in potential_text_columns if col in all_columns]

usecols = [
    col for col in id_time_columns + potential_text_columns if col in all_columns
]

if not text_columns:
    raise RuntimeError("Nenhuma coluna textual de medicamento foi encontrada.")

print("\nColunas textuais pesquisadas:")
print(text_columns)

print("\nTotal de colunas lidas:")
print(len(usecols))

# ------------------------------------------------------------
# 5. Termos de busca
# ------------------------------------------------------------

search_patterns = {
    "methylene": r"\bmethylene\b",
    "methylene_blue": r"\bmethylene\s+blue\b",
    "methylthioninium": r"\bmethylthioninium\b",
    "methylthionine": r"\bmethylthionine\b",
    "provayblue": r"\bprovay\s*blue\b|\bprovayblue\b",
    "uroleen_blue": r"\buroleen\s*blue\b",
}

combined_pattern = "|".join(f"(?:{pattern})" for pattern in search_patterns.values())

print("\nPadrão de busca:")
print(combined_pattern)

# ------------------------------------------------------------
# 6. Leitura em chunks
# ------------------------------------------------------------

chunk_size = 150_000
matched_chunks = []

total_rows = 0
total_matches = 0

print("\nIniciando leitura em chunks...")
print(f"Tamanho de cada chunk: {chunk_size:,} linhas")

reader = pd.read_csv(
    PRESCRIPTIONS_PATH,
    compression="gzip",
    usecols=usecols,
    chunksize=chunk_size,
    low_memory=False,
)

for chunk_number, chunk in enumerate(reader, start=1):

    total_rows += len(chunk)

    search_text = (
        chunk[text_columns].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
    )

    mask = search_text.str.contains(combined_pattern, regex=True, na=False)

    n_matches = int(mask.sum())
    total_matches += n_matches

    if n_matches > 0:
        matched = chunk.loc[mask].copy()
        matched["_search_text"] = search_text.loc[mask].values

        for name, pattern in search_patterns.items():
            matched[f"match_{name}"] = matched["_search_text"].str.contains(
                pattern, regex=True, na=False
            )

        matched_chunks.append(matched)

    print(
        f"Chunk {chunk_number:03d} | "
        f"linhas lidas: {total_rows:,} | "
        f"matches no chunk: {n_matches:,} | "
        f"matches acumulados: {total_matches:,}"
    )

    del chunk
    del search_text
    del mask
    gc.collect()

# ------------------------------------------------------------
# 7. Consolidar somente os registros encontrados
# ------------------------------------------------------------

if matched_chunks:
    prescriptions_hits = pd.concat(matched_chunks, ignore_index=True)
else:
    prescriptions_hits = pd.DataFrame(columns=usecols)

del matched_chunks
gc.collect()

print("\n" + "=" * 90)
print("RESULTADO DA BUSCA EM PRESCRIPTIONS")
print("=" * 90)

print(f"Linhas examinadas: {total_rows:,}")
print(f"Registros encontrados: {len(prescriptions_hits):,}")

if not prescriptions_hits.empty:
    print("Pacientes únicos:", prescriptions_hits["subject_id"].nunique())

    if "hadm_id" in prescriptions_hits.columns:
        print("Internações únicas:", prescriptions_hits["hadm_id"].nunique())

# ------------------------------------------------------------
# 8. Mostrar nomes encontrados
# ------------------------------------------------------------

name_columns = [
    col
    for col in [
        "drug",
        "drug_name_poe",
        "drug_name_generic",
        "formulary_drug_cd",
        "prod_strength",
        "route",
        "dose_val_rx",
        "dose_unit_rx",
        "form_val_disp",
        "form_unit_disp",
    ]
    if col in prescriptions_hits.columns
]

if prescriptions_hits.empty:

    medication_names = pd.DataFrame()

    print("\nNenhum registro compatível foi encontrado.")

else:

    medication_names = (
        prescriptions_hits[name_columns]
        .fillna("<missing>")
        .value_counts(dropna=False)
        .reset_index(name="n_records")
        .sort_values("n_records", ascending=False)
        .reset_index(drop=True)
    )

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 160)
    pd.set_option("display.width", 240)

    print("\nNomes, formulações, doses e vias encontradas:")
    display(medication_names.head(100))

    print("\nPrimeiros registros individuais:")
    display(
        prescriptions_hits[
            [
                col
                for col in [
                    "subject_id",
                    "hadm_id",
                    "pharmacy_id",
                    "starttime",
                    "stoptime",
                    "drug",
                    "drug_name_poe",
                    "drug_name_generic",
                    "formulary_drug_cd",
                    "prod_strength",
                    "dose_val_rx",
                    "dose_unit_rx",
                    "route",
                    "order_status",
                ]
                if col in prescriptions_hits.columns
            ]
        ].head(50)
    )

# ------------------------------------------------------------
# 9. Salvar outputs
# ------------------------------------------------------------

timestamp = run_stamp()

hits_path = COHORT_DIR / f"methylene_blue_prescriptions_hits_{timestamp}.csv.gz"

names_path = QC_DIR / f"methylene_blue_prescription_names_{timestamp}.csv"

report_path = QC_DIR / f"methylene_blue_prescriptions_report_{timestamp}.txt"

prescriptions_hits.to_csv(hits_path, index=False, compression="gzip")

medication_names.to_csv(names_path, index=False)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("METHYLENE BLUE SEARCH IN PRESCRIPTIONS\n")
    file.write("=" * 60 + "\n\n")
    file.write(f"Source: {PRESCRIPTIONS_PATH}\n")
    file.write(f"Output directory: {OUTPUT}\n")
    file.write(f"Rows examined: {total_rows}\n")
    file.write(f"Matched records: {len(prescriptions_hits)}\n")

    if not prescriptions_hits.empty:
        file.write(
            f"Unique patients: " f"{prescriptions_hits['subject_id'].nunique()}\n"
        )

        if "hadm_id" in prescriptions_hits.columns:
            file.write(
                f"Unique admissions: " f"{prescriptions_hits['hadm_id'].nunique()}\n"
            )

print("\nOutputs salvos em:")
print(hits_path)
print(names_path)
print(report_path)

print("\nCÓDIGO CONCLUÍDO.")
