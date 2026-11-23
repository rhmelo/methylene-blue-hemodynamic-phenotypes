"""Source: original notebook cell 37. Run through run.py."""

from common import artifact, display, mimic_source, phase_dir, run_stamp, source_file

# ============================================================
# CÓDIGO 10 — LABORATÓRIOS BASAIS DA TABELA 1
#
# População:
#   59 pacientes com fenótipo hemodinâmico classificável
#
# Exames:
#   - Lactato
#   - Creatinina
#   - Bilirrubina total
#   - Plaquetas
#
# Definição basal:
#   Último valor disponível nas 24 horas anteriores ao
#   horário preliminar do azul de metileno.
#
# Fonte:

#
# Outputs:
#   HIRID/.../MIMIC_Phase
#
# labevents será lido em chunks para limitar o uso de RAM.
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
TABLE_DIR = PROJECT_DIR / "tables"
QC_DIR = PROJECT_DIR / "qc"
METADATA_DIR = PROJECT_DIR / "metadata"

for directory in [COHORT_DIR, TABLE_DIR, QC_DIR, METADATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Pasta-fonte não encontrada: {SOURCE_DIR}")

print("=" * 100)
print("CÓDIGO 10 — LABORATÓRIOS BASAIS DA TABELA 1")
print("=" * 100)

print("\nFonte, somente leitura:")
print(SOURCE_DIR)

print("\nOutputs:")
print(PROJECT_DIR)

# ------------------------------------------------------------
# 2. Funções auxiliares
# ------------------------------------------------------------


def locate_source_file(filename):
    return source_file(SOURCE_DIR, filename)


def find_latest_file(patterns):
    return artifact(PROJECT_DIR, patterns)


# ------------------------------------------------------------
# 3. Localizar a base hemodinâmica mais recente
# ------------------------------------------------------------

TABLE1_BASE_PATH = find_latest_file(
    [
        "TABLE1_BASE_FINAL_combined_bp_phenotype_*.csv.gz",
        "TABLE1_BASE_combined_bp_phenotype_*.csv.gz",
        "TABLE1_BASE_with_invasive_and_noninvasive_phenotype_*.csv.gz",
    ]
)

if TABLE1_BASE_PATH is None:

    print("\nArquivos disponíveis em cohorts:")

    for path in sorted(COHORT_DIR.glob("*.csv.gz")):
        print(path.name)

    raise FileNotFoundError("Não encontrei a base final com os fenótipos.")

D_LABITEMS_PATH = locate_source_file("d_labitems.csv.gz")

LABEVENTS_PATH = locate_source_file("labevents.csv.gz")

print("\nBase selecionada:")
print(TABLE1_BASE_PATH)

print("\nd_labitems:")
print(D_LABITEMS_PATH)

print("\nlabevents:")
print(LABEVENTS_PATH)

# ------------------------------------------------------------
# 4. Carregar a base e manter somente os classificáveis
# ------------------------------------------------------------

cohort = pd.read_csv(TABLE1_BASE_PATH, compression="gzip", low_memory=False)

required_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "mb_time_preliminary",
    "hemodynamic_phenotype",
    "phenotype_data_available",
]

missing_columns = [
    column for column in required_columns if column not in cohort.columns
]

if missing_columns:
    raise RuntimeError("Colunas obrigatórias ausentes na base: " f"{missing_columns}")

for column in ["subject_id", "hadm_id", "stay_id"]:
    cohort[column] = pd.to_numeric(cohort[column], errors="coerce").astype("Int64")

cohort["mb_time_preliminary"] = pd.to_datetime(
    cohort["mb_time_preliminary"], errors="coerce"
)

phenotype_available = (
    cohort["phenotype_data_available"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes"])
)

cohort = cohort[phenotype_available].copy()

cohort = (
    cohort.sort_values(["subject_id", "mb_time_preliminary"])
    .drop_duplicates(subset=["subject_id"], keep="first")
    .reset_index(drop=True)
)

print("\nCoorte classificável:")
print("Pacientes:", cohort["subject_id"].nunique())
print("Internações:", cohort["hadm_id"].nunique())

print("\nFenótipos:")

display(
    cohort["hemodynamic_phenotype"]
    .value_counts()
    .rename_axis("phenotype")
    .reset_index(name="n")
)

if len(cohort) != 59:
    print(
        "\nATENÇÃO: a base classificável contém " f"{len(cohort)} pacientes, e não 59."
    )

# ------------------------------------------------------------
# 5. Carregar e validar o dicionário laboratorial
# ------------------------------------------------------------

d_labitems = pd.read_csv(D_LABITEMS_PATH, compression="gzip", low_memory=False)

d_labitems["itemid"] = pd.to_numeric(d_labitems["itemid"], errors="coerce")

d_labitems["_label_normalized"] = (
    d_labitems["label"].fillna("").astype(str).str.lower().str.strip()
)

# ------------------------------------------------------------
# 6. Identificar itemids por rótulo
# ------------------------------------------------------------

# Itemids padrão esperados no MIMIC-IV:
# 50813 = Lactate
# 50912 = Creatinine
# 50885 = Bilirubin, Total
# 51265 = Platelet Count

LAB_DEFINITION = {
    "lactate": {"preferred_itemid": 50813, "exact_labels": ["lactate"]},
    "creatinine": {"preferred_itemid": 50912, "exact_labels": ["creatinine"]},
    "bilirubin_total": {
        "preferred_itemid": 50885,
        "exact_labels": ["bilirubin, total", "bilirubin total"],
    },
    "platelets": {
        "preferred_itemid": 51265,
        "exact_labels": ["platelet count", "platelets"],
    },
}

selected_lab_rows = []

for variable_name, definition in LAB_DEFINITION.items():

    preferred_itemid = definition["preferred_itemid"]

    preferred_match = d_labitems[d_labitems["itemid"].eq(preferred_itemid)].copy()

    if not preferred_match.empty:

        selected = preferred_match.iloc[0].copy()

    else:

        label_match = d_labitems[
            d_labitems["_label_normalized"].isin(definition["exact_labels"])
        ].copy()

        if label_match.empty:
            raise RuntimeError(
                f"Não foi possível identificar o itemid de " f"{variable_name}."
            )

        selected = label_match.iloc[0].copy()

    selected_lab_rows.append(
        {
            "variable": variable_name,
            "itemid": int(selected["itemid"]),
            "label": selected.get("label", np.nan),
            "fluid": selected.get("fluid", np.nan),
            "category": selected.get("category", np.nan),
        }
    )

selected_labitems = pd.DataFrame(selected_lab_rows)

print("\n" + "=" * 100)
print("ITEMIDS LABORATORIAIS SELECIONADOS")
print("=" * 100)

display(selected_labitems)

LAB_ITEM_MAP = dict(zip(selected_labitems["itemid"], selected_labitems["variable"]))

TARGET_LAB_ITEMIDS = set(LAB_ITEM_MAP.keys())

# ------------------------------------------------------------
# 7. Preparar janelas individuais
# ------------------------------------------------------------

time_table = cohort[["subject_id", "hadm_id", "stay_id", "mb_time_preliminary"]].copy()

time_table["lab_window_start"] = time_table["mb_time_preliminary"] - pd.Timedelta(
    hours=24
)

time_table["lab_window_end"] = time_table["mb_time_preliminary"]

target_hadm_ids = set(cohort["hadm_id"].dropna().astype(int).tolist())

# ------------------------------------------------------------
# 8. Inspecionar cabeçalho de labevents
# ------------------------------------------------------------

lab_header = pd.read_csv(LABEVENTS_PATH, compression="gzip", nrows=0)

available_lab_columns = lab_header.columns.tolist()

preferred_lab_columns = [
    "labevent_id",
    "subject_id",
    "hadm_id",
    "specimen_id",
    "itemid",
    "order_provider_id",
    "charttime",
    "storetime",
    "value",
    "valuenum",
    "valueuom",
    "ref_range_lower",
    "ref_range_upper",
    "flag",
    "priority",
    "comments",
]

lab_usecols = [
    column for column in preferred_lab_columns if column in available_lab_columns
]

required_lab_columns = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]

missing_lab_columns = [
    column for column in required_lab_columns if column not in lab_usecols
]

if missing_lab_columns:
    raise RuntimeError(
        "Colunas obrigatórias ausentes em labevents: " f"{missing_lab_columns}"
    )

print("\nColunas lidas de labevents:")
print(lab_usecols)

# ------------------------------------------------------------
# 9. Ler labevents em chunks
# ------------------------------------------------------------

chunk_size = 500_000

matched_chunks = []

total_rows = 0
total_target_item_rows = 0
total_target_admission_rows = 0
total_window_rows = 0

print("\nIniciando leitura de labevents em chunks...")
print(f"Chunk size: {chunk_size:,}")

reader = pd.read_csv(
    LABEVENTS_PATH,
    compression="gzip",
    usecols=lab_usecols,
    chunksize=chunk_size,
    low_memory=False,
)

for chunk_number, chunk in enumerate(reader, start=1):

    total_rows += len(chunk)

    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")

    chunk = chunk[chunk["itemid"].isin(TARGET_LAB_ITEMIDS)].copy()

    total_target_item_rows += len(chunk)

    if chunk.empty:

        if chunk_number % 10 == 0:
            print(
                f"Chunk {chunk_number:03d} | "
                f"linhas lidas: {total_rows:,} | "
                f"janela acumulada: {total_window_rows:,}"
            )

        gc.collect()
        continue

    chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")

    chunk = chunk[chunk["hadm_id"].isin(target_hadm_ids)].copy()

    total_target_admission_rows += len(chunk)

    if chunk.empty:
        gc.collect()
        continue

    chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce").astype(
        "Int64"
    )

    chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")

    chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")

    chunk = chunk.merge(
        time_table[
            [
                "subject_id",
                "hadm_id",
                "mb_time_preliminary",
                "lab_window_start",
                "lab_window_end",
            ]
        ],
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="many_to_one",
    )

    chunk = chunk[
        chunk["charttime"].notna()
        & chunk["valuenum"].notna()
        & (chunk["charttime"] >= chunk["lab_window_start"])
        & (chunk["charttime"] <= chunk["lab_window_end"])
    ].copy()

    if not chunk.empty:

        chunk["lab_variable"] = chunk["itemid"].astype(int).map(LAB_ITEM_MAP)

        matched_chunks.append(chunk)

        total_window_rows += len(chunk)

    if chunk_number % 5 == 0:

        print(
            f"Chunk {chunk_number:03d} | "
            f"linhas lidas: {total_rows:,} | "
            f"registros na janela: {total_window_rows:,}"
        )

    gc.collect()

# ------------------------------------------------------------
# 10. Consolidar exames
# ------------------------------------------------------------

if matched_chunks:

    lab_events = pd.concat(matched_chunks, ignore_index=True)

else:

    lab_events = pd.DataFrame()

del matched_chunks
gc.collect()

print("\n" + "=" * 100)
print("EXTRAÇÃO LABORATORIAL CONCLUÍDA")
print("=" * 100)

print("Linhas totais examinadas:", f"{total_rows:,}")
print("Registros dos itemids-alvo:", f"{total_target_item_rows:,}")
print("Registros das internações-alvo:", f"{total_target_admission_rows:,}")
print("Registros dentro da janela de 24 h:", f"{len(lab_events):,}")

if lab_events.empty:
    raise RuntimeError("Nenhum exame foi encontrado na janela basal.")

# ------------------------------------------------------------
# 11. Limpeza fisiológica simples
# ------------------------------------------------------------

lab_events["physiologically_valid"] = True
lab_events["invalid_reason"] = ""

plausibility_ranges = {
    "lactate": (0.1, 30.0),
    "creatinine": (0.1, 25.0),
    "bilirubin_total": (0.0, 80.0),
    "platelets": (1.0, 2000.0),
}

for variable_name, limits in plausibility_ranges.items():

    lower_limit, upper_limit = limits

    invalid_mask = lab_events["lab_variable"].eq(variable_name) & ~lab_events[
        "valuenum"
    ].between(lower_limit, upper_limit, inclusive="both")

    lab_events.loc[invalid_mask, "physiologically_valid"] = False

    lab_events.loc[invalid_mask, "invalid_reason"] = (
        f"{variable_name}_outside_" f"{lower_limit}_{upper_limit}"
    )

valid_lab_events = lab_events[lab_events["physiologically_valid"]].copy()

print("\nRegistros fisiologicamente válidos:", len(valid_lab_events))

print("Registros removidos:", len(lab_events) - len(valid_lab_events))

# ------------------------------------------------------------
# 12. Selecionar o último valor antes do azul
# ------------------------------------------------------------

valid_lab_events["hours_before_mb"] = (
    valid_lab_events["mb_time_preliminary"] - valid_lab_events["charttime"]
).dt.total_seconds() / 3600

last_labs = (
    valid_lab_events.sort_values(
        ["subject_id", "lab_variable", "charttime", "storetime"], na_position="first"
    )
    .drop_duplicates(subset=["subject_id", "lab_variable"], keep="last")
    .copy()
)

# ------------------------------------------------------------
# 13. Criar formato largo
# ------------------------------------------------------------

lab_values_wide = last_labs.pivot(
    index=["subject_id", "hadm_id"], columns="lab_variable", values="valuenum"
).reset_index()

lab_values_wide.columns.name = None

lab_values_wide = lab_values_wide.rename(
    columns={
        "lactate": "lactate_pre_mb",
        "creatinine": "creatinine_pre_mb",
        "bilirubin_total": "bilirubin_total_pre_mb",
        "platelets": "platelets_pre_mb",
    }
)

lab_time_wide = last_labs.pivot(
    index=["subject_id", "hadm_id"], columns="lab_variable", values="hours_before_mb"
).reset_index()

lab_time_wide.columns.name = None

lab_time_wide = lab_time_wide.rename(
    columns={
        "lactate": "lactate_hours_before_mb",
        "creatinine": "creatinine_hours_before_mb",
        "bilirubin_total": "bilirubin_total_hours_before_mb",
        "platelets": "platelets_hours_before_mb",
    }
)

lab_summary = lab_values_wide.merge(
    lab_time_wide, on=["subject_id", "hadm_id"], how="outer", validate="one_to_one"
)

# ------------------------------------------------------------
# 14. Integrar à base da Tabela 1
# ------------------------------------------------------------

laboratory_columns = [
    "lactate_pre_mb",
    "creatinine_pre_mb",
    "bilirubin_total_pre_mb",
    "platelets_pre_mb",
    "lactate_hours_before_mb",
    "creatinine_hours_before_mb",
    "bilirubin_total_hours_before_mb",
    "platelets_hours_before_mb",
]

cohort_without_old_labs = cohort.drop(
    columns=[column for column in laboratory_columns if column in cohort.columns],
    errors="ignore",
)

table1_labs = cohort_without_old_labs.merge(
    lab_summary, on=["subject_id", "hadm_id"], how="left", validate="one_to_one"
)

# ------------------------------------------------------------
# 15. Disponibilidade
# ------------------------------------------------------------

availability_rows = []

lab_labels = {
    "lactate_pre_mb": "Lactate",
    "creatinine_pre_mb": "Creatinine",
    "bilirubin_total_pre_mb": "Total bilirubin",
    "platelets_pre_mb": "Platelet count",
}

for variable_name, label in lab_labels.items():

    n_available = int(table1_labs[variable_name].notna().sum())

    availability_rows.append(
        {
            "variable": label,
            "n_available": n_available,
            "n_total": len(table1_labs),
            "percent_available": (100 * n_available / len(table1_labs)),
        }
    )

lab_availability = pd.DataFrame(availability_rows)

# ------------------------------------------------------------
# 16. Resumo dos exames
# ------------------------------------------------------------

summary_rows = []

for variable_name, label in lab_labels.items():

    values = pd.to_numeric(table1_labs[variable_name], errors="coerce")

    summary_rows.append(
        {
            "variable": label,
            "n": int(values.notna().sum()),
            "median": values.median(),
            "p25": values.quantile(0.25),
            "p75": values.quantile(0.75),
            "min": values.min(),
            "max": values.max(),
        }
    )

lab_descriptive_summary = pd.DataFrame(summary_rows)

# ------------------------------------------------------------
# 17. Resumo por fenótipo
# ------------------------------------------------------------

phenotype_lab_rows = []

phenotype_order = [
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
    "Preserved diastolic pressure",
]

for phenotype in phenotype_order:

    phenotype_data = table1_labs[table1_labs["hemodynamic_phenotype"].eq(phenotype)]

    for variable_name, label in lab_labels.items():

        values = pd.to_numeric(phenotype_data[variable_name], errors="coerce")

        phenotype_lab_rows.append(
            {
                "phenotype": phenotype,
                "variable": label,
                "n": int(values.notna().sum()),
                "median": values.median(),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
            }
        )

phenotype_lab_summary = pd.DataFrame(phenotype_lab_rows)

# ------------------------------------------------------------
# 18. Mostrar resultados
# ------------------------------------------------------------

pd.set_option("display.max_rows", 200)

pd.set_option("display.max_columns", 50)

pd.set_option("display.width", 280)

print("\n" + "=" * 100)
print("DISPONIBILIDADE DOS EXAMES")
print("=" * 100)

display(lab_availability)

print("\n" + "=" * 100)
print("RESUMO GERAL DOS EXAMES")
print("=" * 100)

display(lab_descriptive_summary)

print("\n" + "=" * 100)
print("RESUMO DOS EXAMES POR FENÓTIPO")
print("=" * 100)

display(phenotype_lab_summary)

# ------------------------------------------------------------
# 19. Controle de qualidade
# ------------------------------------------------------------

qc_summary = pd.DataFrame(
    {
        "metric": [
            "Classifiable cohort patients",
            "Lactate available",
            "Creatinine available",
            "Total bilirubin available",
            "Platelet count available",
            "Patients with all four laboratory variables",
        ],
        "value": [
            len(table1_labs),
            table1_labs["lactate_pre_mb"].notna().sum(),
            table1_labs["creatinine_pre_mb"].notna().sum(),
            table1_labs["bilirubin_total_pre_mb"].notna().sum(),
            table1_labs["platelets_pre_mb"].notna().sum(),
            table1_labs[
                [
                    "lactate_pre_mb",
                    "creatinine_pre_mb",
                    "bilirubin_total_pre_mb",
                    "platelets_pre_mb",
                ]
            ]
            .notna()
            .all(axis=1)
            .sum(),
        ],
    }
)

print("\n" + "=" * 100)
print("CONTROLE DE QUALIDADE")
print("=" * 100)

display(qc_summary)

# ------------------------------------------------------------
# 20. Salvar outputs
# ------------------------------------------------------------

timestamp = run_stamp()

table1_labs_path = COHORT_DIR / f"TABLE1_BASE_with_baseline_labs_{timestamp}.csv.gz"

raw_lab_events_path = (
    COHORT_DIR / f"methylene_blue_baseline_lab_events_{timestamp}.csv.gz"
)

last_labs_path = COHORT_DIR / f"methylene_blue_last_baseline_labs_{timestamp}.csv.gz"

availability_path = (
    TABLE_DIR / f"TABLE1_baseline_laboratory_availability_{timestamp}.csv"
)

descriptive_path = TABLE_DIR / f"TABLE1_baseline_laboratory_summary_{timestamp}.csv"

phenotype_summary_path = (
    TABLE_DIR / f"TABLE1_baseline_labs_by_phenotype_{timestamp}.csv"
)

qc_path = QC_DIR / f"methylene_blue_code10_laboratory_qc_{timestamp}.csv"

itemids_path = METADATA_DIR / f"methylene_blue_table1_lab_itemids_{timestamp}.csv"

report_path = QC_DIR / f"methylene_blue_code10_laboratory_report_{timestamp}.txt"

table1_labs.to_csv(table1_labs_path, index=False, compression="gzip")

valid_lab_events.to_csv(raw_lab_events_path, index=False, compression="gzip")

last_labs.to_csv(last_labs_path, index=False, compression="gzip")

lab_availability.to_csv(availability_path, index=False)

lab_descriptive_summary.to_csv(descriptive_path, index=False)

phenotype_lab_summary.to_csv(phenotype_summary_path, index=False)

qc_summary.to_csv(qc_path, index=False)

selected_labitems.to_csv(itemids_path, index=False)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("METHYLENE BLUE — BASELINE LABORATORIES\n")

    file.write("=" * 70 + "\n\n")

    file.write(
        "Baseline definition: the last available laboratory "
        "measurement during the 24 hours preceding the "
        "preliminary methylene blue time.\n\n"
    )

    file.write(f"Input cohort: {TABLE1_BASE_PATH}\n")

    file.write(f"Classifiable patients: {len(table1_labs)}\n")

    file.write(f"Updated Table 1 base: {table1_labs_path}\n")

print("\n" + "=" * 100)
print("OUTPUTS SALVOS")
print("=" * 100)

print(table1_labs_path)
print(raw_lab_events_path)
print(last_labs_path)
print(availability_path)
print(descriptive_path)
print(phenotype_summary_path)
print(qc_path)
print(report_path)

gc.collect()

print("\nCÓDIGO 10 CONCLUÍDO COM SUCESSO.")
