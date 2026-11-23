"""Source: original notebook cell 35. Run through run.py."""

from common import artifact, display, mimic_source, phase_dir, run_stamp, source_file

# ============================================================
# CÓDIGO 9B FINAL
# HEMODINÂMICA PRÉ-AZUL:
# PRESSÃO INVASIVA COM FALLBACK NÃO INVASIVO
#
# Coorte:
#   Sepse + norepinefrina ativa no momento do azul
#   + exclusão de metemoglobinemia
#
# Janela basal:
#   -60 a 0 minutos antes do azul de metileno
#
# Estratégia:
#   1. Usar PAS/PAD invasivas pareadas quando disponíveis
#   2. Caso contrário, usar PAS/PAD não invasivas pareadas
#   3. Nunca combinar modalidades diferentes
#   4. Não avaliar concordância entre modalidades
#
# Fenótipos:
#   Vasoplegia-compatible:
#       PAD < 50 mmHg e PP >= 40 mmHg
#
#   Mixed/low-pulsatility:
#       PAD < 50 mmHg e PP < 40 mmHg
#
#   Preserved diastolic pressure:
#       PAD >= 50 mmHg
#
# Fonte:

#   SOMENTE LEITURA
#
# Outputs:

#   HiRID_Methylene_Blue_Hemodynamic_Phenotypes/MIMIC_Phase
# ============================================================

import pandas as pd
import numpy as np
import gc

# ------------------------------------------------------------
# 1. Caminhos fixos
# ------------------------------------------------------------

SOURCE_DIR = mimic_source()

PROJECT_DIR = phase_dir("mimic_iv")

COHORT_DIR = PROJECT_DIR / "cohorts"
QC_DIR = PROJECT_DIR / "qc"
TABLE_DIR = PROJECT_DIR / "tables"
METADATA_DIR = PROJECT_DIR / "metadata"

for directory in [COHORT_DIR, QC_DIR, TABLE_DIR, METADATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Pasta-fonte não encontrada: {SOURCE_DIR}")

print("=" * 100)
print("CÓDIGO 9B FINAL — PRESSÃO INVASIVA + FALLBACK NÃO INVASIVO")
print("=" * 100)

print("\nFonte MIMIC-IV, somente leitura:")
print(SOURCE_DIR)

print("\nDiretório de outputs:")
print(PROJECT_DIR)

# ------------------------------------------------------------
# 2. Funções auxiliares
# ------------------------------------------------------------


def locate_source_file(filename):
    return source_file(SOURCE_DIR, filename)


def find_latest_file(patterns):
    return artifact(PROJECT_DIR, patterns)


# ------------------------------------------------------------
# 3. Localizar a coorte principal
# ------------------------------------------------------------

primary_patterns = [
    "TABLE1_BASE_methylene_blue_primary_cohort_*.csv.gz",
    "TABLE1_BASE_methylene_blue_primary_cohort_*.csv.gz",
    "methylene_blue_sepsis_active_norepinephrine_cohort_*.csv.gz",
]

PRIMARY_COHORT_PATH = artifact(
    PROJECT_DIR, "TABLE1_BASE_methylene_blue_primary_cohort_*.csv.gz"
)

using_broad_fallback = False

if PRIMARY_COHORT_PATH is None:

    broad_patterns = ["methylene_blue_broad_cohort_sepsis_ne_flags_*.csv.gz"]

    PRIMARY_COHORT_PATH = find_latest_file(broad_patterns)

    using_broad_fallback = True

if PRIMARY_COHORT_PATH is None:

    print("\nArquivos disponíveis na pasta cohorts:")

    for path in sorted(COHORT_DIR.glob("*.csv.gz")):
        print(path.name)

    raise FileNotFoundError(
        "Não encontrei a coorte principal nem a coorte ampla "
        "com flags de elegibilidade."
    )

print("\nArquivo de coorte selecionado:")
print(PRIMARY_COHORT_PATH)

# ------------------------------------------------------------
# 4. Localizar arquivos do MIMIC-IV
# ------------------------------------------------------------

CHARTEVENTS_PATH = locate_source_file("chartevents.csv.gz")

D_ITEMS_PATH = locate_source_file("d_items.csv.gz")

print("\nChartevents:")
print(CHARTEVENTS_PATH)

print("\nDicionário de itens:")
print(D_ITEMS_PATH)

# ------------------------------------------------------------
# 5. Carregar a coorte
# ------------------------------------------------------------

cohort = pd.read_csv(PRIMARY_COHORT_PATH, compression="gzip", low_memory=False)

if cohort.empty:
    raise RuntimeError("O arquivo de coorte selecionado está vazio.")

# ------------------------------------------------------------
# 6. Aplicar elegibilidade quando a origem for a coorte ampla
# ------------------------------------------------------------

if using_broad_fallback or "primary_cohort_eligible" in cohort.columns:

    if "primary_cohort_eligible" not in cohort.columns:
        raise RuntimeError(
            "Foi carregada uma coorte ampla, mas a variável "
            "primary_cohort_eligible não está disponível."
        )

    eligibility = (
        cohort["primary_cohort_eligible"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    n_before = len(cohort)

    cohort = cohort[eligibility].copy()

    print("\nFiltro primary_cohort_eligible aplicado: " f"{n_before} → {len(cohort)}")

# ------------------------------------------------------------
# 7. Validar identificadores e horário
# ------------------------------------------------------------

required_cohort_columns = ["subject_id", "hadm_id", "stay_id", "mb_time_preliminary"]

missing_cohort_columns = [
    column for column in required_cohort_columns if column not in cohort.columns
]

if missing_cohort_columns:
    raise RuntimeError(
        "Colunas obrigatórias ausentes na coorte: " f"{missing_cohort_columns}"
    )

for column in ["subject_id", "hadm_id", "stay_id"]:
    cohort[column] = pd.to_numeric(cohort[column], errors="coerce").astype("Int64")

cohort["mb_time_preliminary"] = pd.to_datetime(
    cohort["mb_time_preliminary"], errors="coerce"
)

cohort = cohort[
    cohort["subject_id"].notna()
    & cohort["hadm_id"].notna()
    & cohort["stay_id"].notna()
    & cohort["mb_time_preliminary"].notna()
].copy()

cohort = (
    cohort.sort_values(["subject_id", "mb_time_preliminary"])
    .drop_duplicates(subset=["subject_id"], keep="first")
    .reset_index(drop=True)
)

print("\nCoorte carregada:")
print("Pacientes:", cohort["subject_id"].nunique())
print("Internações:", cohort["hadm_id"].nunique())
print("Estadias de UTI:", cohort["stay_id"].nunique())

if len(cohort) != 66:
    print("\nATENÇÃO: a coorte contém " f"{len(cohort)} pacientes, e não 66.")

# ------------------------------------------------------------
# 8. Definir itemids
# ------------------------------------------------------------

ITEM_DEFINITION = {
    # Frequência cardíaca
    220045: {"variable": "heart_rate", "modality": "general"},
    # Pressão invasiva
    220050: {"variable": "sbp", "modality": "invasive"},
    220051: {"variable": "dbp", "modality": "invasive"},
    220052: {"variable": "map", "modality": "invasive"},
    # Pressão não invasiva
    220179: {"variable": "sbp", "modality": "noninvasive"},
    220180: {"variable": "dbp", "modality": "noninvasive"},
    220181: {"variable": "map", "modality": "noninvasive"},
}

TARGET_ITEMIDS = set(ITEM_DEFINITION.keys())

# ------------------------------------------------------------
# 9. Validar itemids no d_items
# ------------------------------------------------------------

d_items = pd.read_csv(D_ITEMS_PATH, compression="gzip", low_memory=False)

d_items["itemid"] = pd.to_numeric(d_items["itemid"], errors="coerce")

validation_columns = [
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
    if column in d_items.columns
]

validated_items = (
    d_items[d_items["itemid"].isin(TARGET_ITEMIDS)][validation_columns]
    .drop_duplicates()
    .sort_values("itemid")
    .reset_index(drop=True)
)

print("\n" + "=" * 100)
print("VALIDAÇÃO DOS ITEMIDS")
print("=" * 100)

display(validated_items)

found_itemids = set(validated_items["itemid"].dropna().astype(int))

missing_itemids = TARGET_ITEMIDS - found_itemids

if missing_itemids:
    raise RuntimeError("Itemids ausentes no d_items: " f"{sorted(missing_itemids)}")

# ------------------------------------------------------------
# 10. Construir a janela −60 a 0 minutos
# ------------------------------------------------------------

mb_times = cohort[["subject_id", "hadm_id", "stay_id", "mb_time_preliminary"]].copy()

mb_times["window_start"] = mb_times["mb_time_preliminary"] - pd.Timedelta(minutes=60)

mb_times["window_end"] = mb_times["mb_time_preliminary"]

target_stay_ids = set(cohort["stay_id"].dropna().astype(int).tolist())

# ------------------------------------------------------------
# 11. Cabeçalho do chartevents
# ------------------------------------------------------------

chart_header = pd.read_csv(CHARTEVENTS_PATH, compression="gzip", nrows=0)

available_chart_columns = chart_header.columns.tolist()

preferred_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "charttime",
    "storetime",
    "itemid",
    "value",
    "valuenum",
    "valueuom",
    "warning",
]

chart_usecols = [
    column for column in preferred_columns if column in available_chart_columns
]

required_chart_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "charttime",
    "itemid",
    "valuenum",
]

missing_chart_columns = [
    column for column in required_chart_columns if column not in chart_usecols
]

if missing_chart_columns:
    raise RuntimeError(
        "Colunas obrigatórias ausentes em chartevents: " f"{missing_chart_columns}"
    )

print("\nColunas lidas de chartevents:")
print(chart_usecols)

# ------------------------------------------------------------
# 12. Ler chartevents em chunks
# ------------------------------------------------------------

chunk_size = 500_000

matched_chunks = []

total_rows = 0
total_window_rows = 0

print("\nIniciando leitura em chunks...")
print(f"Chunk size: {chunk_size:,}")

reader = pd.read_csv(
    CHARTEVENTS_PATH,
    compression="gzip",
    usecols=chart_usecols,
    chunksize=chunk_size,
    low_memory=False,
)

for chunk_number, chunk in enumerate(reader, start=1):

    total_rows += len(chunk)

    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")

    chunk = chunk[chunk["itemid"].isin(TARGET_ITEMIDS)].copy()

    if chunk.empty:

        if chunk_number % 10 == 0:
            print(
                f"Chunk {chunk_number:03d} | "
                f"linhas lidas: {total_rows:,} | "
                f"janela acumulada: {total_window_rows:,}"
            )

        gc.collect()
        continue

    chunk = chunk[chunk["stay_id"].isin(target_stay_ids)].copy()

    if chunk.empty:
        gc.collect()
        continue

    for column in ["subject_id", "hadm_id", "stay_id"]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("Int64")

    chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")

    chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")

    chunk = chunk.merge(
        mb_times,
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
        validate="many_to_one",
    )

    chunk = chunk[
        chunk["charttime"].notna()
        & (chunk["charttime"] >= chunk["window_start"])
        & (chunk["charttime"] <= chunk["window_end"])
    ].copy()

    if not chunk.empty:

        chunk["variable"] = (
            chunk["itemid"]
            .astype(int)
            .map(lambda itemid: ITEM_DEFINITION[itemid]["variable"])
        )

        chunk["modality"] = (
            chunk["itemid"]
            .astype(int)
            .map(lambda itemid: ITEM_DEFINITION[itemid]["modality"])
        )

        matched_chunks.append(chunk)

        total_window_rows += len(chunk)

    if chunk_number % 5 == 0:

        print(
            f"Chunk {chunk_number:03d} | "
            f"linhas lidas: {total_rows:,} | "
            f"janela acumulada: {total_window_rows:,}"
        )

    gc.collect()

# ------------------------------------------------------------
# 13. Consolidar eventos
# ------------------------------------------------------------

if matched_chunks:

    events = pd.concat(matched_chunks, ignore_index=True)

else:

    events = pd.DataFrame()

del matched_chunks
gc.collect()

print("\nEventos encontrados na janela:", len(events))

if events.empty:
    raise RuntimeError("Nenhum evento hemodinâmico foi encontrado na janela.")

# ------------------------------------------------------------
# 14. Limpeza fisiológica
# ------------------------------------------------------------

valid_mask = pd.Series(True, index=events.index)

valid_mask &= ~(
    events["variable"].eq("sbp")
    & ~events["valuenum"].between(30, 300, inclusive="both")
)

valid_mask &= ~(
    events["variable"].eq("dbp")
    & ~events["valuenum"].between(10, 200, inclusive="both")
)

valid_mask &= ~(
    events["variable"].eq("map")
    & ~events["valuenum"].between(20, 250, inclusive="both")
)

valid_mask &= ~(
    events["variable"].eq("heart_rate")
    & ~events["valuenum"].between(20, 250, inclusive="both")
)

valid_events = events[valid_mask].copy()

print("Eventos válidos:", len(valid_events))
print("Eventos removidos:", len(events) - len(valid_events))

# ------------------------------------------------------------
# 15. Função para resumir uma modalidade
# ------------------------------------------------------------


def summarize_bp(data, modality, prefix):

    identifiers = ["subject_id", "hadm_id", "stay_id"]

    selected = data[data["modality"].eq(modality)].copy()

    if selected.empty:

        return pd.DataFrame(
            columns=(
                identifiers
                + [
                    f"{prefix}_sbp",
                    f"{prefix}_dbp",
                    f"{prefix}_map",
                    f"{prefix}_pp",
                    f"{prefix}_n_paired",
                ]
            )
        )

    median_values = selected.groupby(identifiers + ["variable"], as_index=False).agg(
        median_value=("valuenum", "median")
    )

    median_wide = median_values.pivot(
        index=identifiers, columns="variable", values="median_value"
    ).reset_index()

    median_wide.columns.name = None

    median_wide = median_wide.rename(
        columns={"sbp": f"{prefix}_sbp", "dbp": f"{prefix}_dbp", "map": f"{prefix}_map"}
    )

    bp_only = selected[selected["variable"].isin(["sbp", "dbp"])].copy()

    paired = bp_only.pivot_table(
        index=identifiers + ["charttime"],
        columns="variable",
        values="valuenum",
        aggfunc="median",
    ).reset_index()

    paired.columns.name = None

    if {"sbp", "dbp"}.issubset(paired.columns):

        paired["pp"] = paired["sbp"] - paired["dbp"]

        paired = paired[paired["pp"].between(5, 200, inclusive="both")].copy()

    else:

        paired = pd.DataFrame()

    if paired.empty:

        pp_summary = pd.DataFrame(
            columns=(identifiers + [f"{prefix}_pp", f"{prefix}_n_paired"])
        )

    else:

        pp_summary = paired.groupby(identifiers, as_index=False).agg(
            **{f"{prefix}_pp": ("pp", "median"), f"{prefix}_n_paired": ("pp", "size")}
        )

    summary = median_wide.merge(
        pp_summary, on=identifiers, how="outer", validate="one_to_one"
    )

    if f"{prefix}_n_paired" not in summary.columns:
        summary[f"{prefix}_n_paired"] = 0

    summary[f"{prefix}_n_paired"] = summary[f"{prefix}_n_paired"].fillna(0).astype(int)

    return summary


# ------------------------------------------------------------
# 16. Resumir invasiva e não invasiva
# ------------------------------------------------------------

inv_summary = summarize_bp(valid_events, "invasive", "inv")

nibp_summary = summarize_bp(valid_events, "noninvasive", "nibp")

# ------------------------------------------------------------
# 17. Frequência cardíaca
# ------------------------------------------------------------

hr_events = valid_events[valid_events["variable"].eq("heart_rate")].copy()

if hr_events.empty:

    hr_summary = pd.DataFrame(
        columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "hr_pre_mb_median",
            "n_hr_measurements",
        ]
    )

else:

    hr_summary = hr_events.groupby(
        ["subject_id", "hadm_id", "stay_id"], as_index=False
    ).agg(
        hr_pre_mb_median=("valuenum", "median"), n_hr_measurements=("valuenum", "size")
    )

# ------------------------------------------------------------
# 18. Integrar por paciente
# ------------------------------------------------------------

identifiers = ["subject_id", "hadm_id", "stay_id"]

hemo = (
    cohort[identifiers]
    .merge(inv_summary, on=identifiers, how="left", validate="one_to_one")
    .merge(nibp_summary, on=identifiers, how="left", validate="one_to_one")
    .merge(hr_summary, on=identifiers, how="left", validate="one_to_one")
)

required_numeric_columns = [
    "inv_sbp",
    "inv_dbp",
    "inv_map",
    "inv_pp",
    "nibp_sbp",
    "nibp_dbp",
    "nibp_map",
    "nibp_pp",
    "hr_pre_mb_median",
]

for column in required_numeric_columns:

    if column not in hemo.columns:
        hemo[column] = np.nan

for column in ["inv_n_paired", "nibp_n_paired", "n_hr_measurements"]:

    if column not in hemo.columns:
        hemo[column] = 0

    hemo[column] = hemo[column].fillna(0).astype(int)

# ------------------------------------------------------------
# 19. Selecionar modalidade
# ------------------------------------------------------------

hemo["invasive_available"] = hemo["inv_dbp"].notna() & hemo["inv_pp"].notna()

hemo["noninvasive_available"] = hemo["nibp_dbp"].notna() & hemo["nibp_pp"].notna()

hemo["bp_modality_selected"] = np.select(
    [
        hemo["invasive_available"],
        (~hemo["invasive_available"] & hemo["noninvasive_available"]),
    ],
    ["Invasive", "Non-invasive"],
    default="Unavailable",
)

use_inv = hemo["bp_modality_selected"].eq("Invasive")

use_nibp = hemo["bp_modality_selected"].eq("Non-invasive")

# ------------------------------------------------------------
# 20. Criar variáveis finais
# ------------------------------------------------------------

hemo["sbp_pre_mb_median"] = np.select(
    [use_inv, use_nibp], [hemo["inv_sbp"], hemo["nibp_sbp"]], default=np.nan
)

hemo["dbp_pre_mb_median"] = np.select(
    [use_inv, use_nibp], [hemo["inv_dbp"], hemo["nibp_dbp"]], default=np.nan
)

hemo["map_pre_mb_median"] = np.select(
    [use_inv, use_nibp], [hemo["inv_map"], hemo["nibp_map"]], default=np.nan
)

hemo["pp_pre_mb_median"] = np.select(
    [use_inv, use_nibp], [hemo["inv_pp"], hemo["nibp_pp"]], default=np.nan
)

hemo["n_selected_paired_bp"] = np.select(
    [use_inv, use_nibp], [hemo["inv_n_paired"], hemo["nibp_n_paired"]], default=0
).astype(int)

hemo["phenotype_data_available"] = (
    hemo["dbp_pre_mb_median"].notna() & hemo["pp_pre_mb_median"].notna()
)

# ------------------------------------------------------------
# 21. Classificar fenótipos
# ------------------------------------------------------------

hemo["hemodynamic_phenotype"] = np.select(
    [
        (
            hemo["phenotype_data_available"]
            & (hemo["dbp_pre_mb_median"] < 50)
            & (hemo["pp_pre_mb_median"] >= 40)
        ),
        (
            hemo["phenotype_data_available"]
            & (hemo["dbp_pre_mb_median"] < 50)
            & (hemo["pp_pre_mb_median"] < 40)
        ),
        (hemo["phenotype_data_available"] & (hemo["dbp_pre_mb_median"] >= 50)),
    ],
    ["Vasoplegia-compatible", "Mixed/low-pulsatility", "Preserved diastolic pressure"],
    default="Unavailable",
)

hemo["phenotype_vasoplegia_compatible"] = hemo["hemodynamic_phenotype"].eq(
    "Vasoplegia-compatible"
)

hemo["phenotype_mixed_low_pulsatility"] = hemo["hemodynamic_phenotype"].eq(
    "Mixed/low-pulsatility"
)

hemo["phenotype_preserved_dbp"] = hemo["hemodynamic_phenotype"].eq(
    "Preserved diastolic pressure"
)

# ------------------------------------------------------------
# 22. Integrar à Tabela 1
# ------------------------------------------------------------

overlapping_columns = [
    column
    for column in hemo.columns
    if (column not in identifiers and column in cohort.columns)
]

cohort_clean = cohort.drop(columns=overlapping_columns, errors="ignore")

table1_base = cohort_clean.merge(
    hemo, on=identifiers, how="left", validate="one_to_one"
)

# ------------------------------------------------------------
# 23. Resumos
# ------------------------------------------------------------

availability_summary = pd.DataFrame(
    {
        "variable": [
            "Classified using invasive BP",
            "Classified using non-invasive BP",
            "Phenotype classifiable",
            "Phenotype unavailable",
        ],
        "n": [
            int(table1_base["bp_modality_selected"].eq("Invasive").sum()),
            int(table1_base["bp_modality_selected"].eq("Non-invasive").sum()),
            int(table1_base["phenotype_data_available"].sum()),
            int(table1_base["bp_modality_selected"].eq("Unavailable").sum()),
        ],
    }
)

availability_summary["total"] = len(table1_base)

availability_summary["percent"] = (
    100 * availability_summary["n"] / availability_summary["total"]
)

phenotype_summary = (
    table1_base["hemodynamic_phenotype"]
    .value_counts(dropna=False)
    .rename_axis("hemodynamic_phenotype")
    .reset_index(name="n_patients")
)

phenotype_summary["percent_total"] = (
    100 * phenotype_summary["n_patients"] / len(table1_base)
)

n_classifiable = int(table1_base["phenotype_data_available"].sum())

phenotype_summary["percent_classifiable"] = np.where(
    phenotype_summary["hemodynamic_phenotype"].ne("Unavailable"),
    (100 * phenotype_summary["n_patients"] / n_classifiable),
    np.nan,
)

classifiable = table1_base[table1_base["phenotype_data_available"]].copy()

if classifiable.empty:

    physiology_summary = pd.DataFrame()

else:

    aggregations = {
        "n_patients": ("subject_id", "nunique"),
        "invasive_n": ("bp_modality_selected", lambda x: int((x == "Invasive").sum())),
        "noninvasive_n": (
            "bp_modality_selected",
            lambda x: int((x == "Non-invasive").sum()),
        ),
        "dbp_median": ("dbp_pre_mb_median", "median"),
        "pp_median": ("pp_pre_mb_median", "median"),
        "map_median": ("map_pre_mb_median", "median"),
        "hr_median": ("hr_pre_mb_median", "median"),
    }

    optional_variables = {
        "norepinephrine_median": ("ne_rate_most_recent_raw", "median"),
        "mb_dose_median": ("dose_mg_clean", "median"),
        "hospital_mortality_percent": ("hospital_mortality", lambda x: 100 * x.mean()),
    }

    for output_name, aggregation in optional_variables.items():

        if aggregation[0] in classifiable.columns:

            aggregations[output_name] = aggregation

    physiology_summary = classifiable.groupby(
        "hemodynamic_phenotype", as_index=False
    ).agg(**aggregations)

# ------------------------------------------------------------
# 24. Mostrar resultados
# ------------------------------------------------------------

pd.set_option("display.max_rows", 200)

pd.set_option("display.max_columns", 60)

pd.set_option("display.width", 300)

print("\n" + "=" * 100)
print("DISPONIBILIDADE HEMODINÂMICA")
print("=" * 100)

display(availability_summary)

print("\n" + "=" * 100)
print("DISTRIBUIÇÃO DOS FENÓTIPOS")
print("=" * 100)

display(phenotype_summary)

print("\n" + "=" * 100)
print("RESUMO FISIOLÓGICO POR FENÓTIPO")
print("=" * 100)

display(physiology_summary)

# ------------------------------------------------------------
# 25. Controle de qualidade
# ------------------------------------------------------------

qc_summary = pd.DataFrame(
    {
        "metric": [
            "Primary cohort",
            "Patients using invasive BP",
            "Patients using non-invasive BP",
            "Classifiable phenotype",
            "Vasoplegia-compatible",
            "Mixed/low-pulsatility",
            "Preserved diastolic pressure",
            "Unavailable phenotype",
        ],
        "value": [
            len(table1_base),
            int(table1_base["bp_modality_selected"].eq("Invasive").sum()),
            int(table1_base["bp_modality_selected"].eq("Non-invasive").sum()),
            int(table1_base["phenotype_data_available"].sum()),
            int(table1_base["phenotype_vasoplegia_compatible"].sum()),
            int(table1_base["phenotype_mixed_low_pulsatility"].sum()),
            int(table1_base["phenotype_preserved_dbp"].sum()),
            int(table1_base["hemodynamic_phenotype"].eq("Unavailable").sum()),
        ],
    }
)

print("\n" + "=" * 100)
print("CONTROLE DE QUALIDADE FINAL")
print("=" * 100)

display(qc_summary)

# ------------------------------------------------------------
# 26. Salvar outputs
# ------------------------------------------------------------

timestamp = run_stamp()

table1_path = COHORT_DIR / f"TABLE1_BASE_FINAL_combined_bp_phenotype_{timestamp}.csv.gz"

hemo_path = COHORT_DIR / f"methylene_blue_final_pre_mb_hemodynamics_{timestamp}.csv.gz"

events_path = COHORT_DIR / f"methylene_blue_pre_mb_bp_events_{timestamp}.csv.gz"

availability_path = TABLE_DIR / f"methylene_blue_final_bp_availability_{timestamp}.csv"

phenotype_path = (
    TABLE_DIR / f"methylene_blue_final_phenotype_distribution_{timestamp}.csv"
)

physiology_path = (
    TABLE_DIR / f"methylene_blue_final_phenotype_physiology_{timestamp}.csv"
)

qc_path = QC_DIR / f"methylene_blue_code9b_final_qc_{timestamp}.csv"

itemids_path = METADATA_DIR / f"methylene_blue_final_bp_itemids_{timestamp}.csv"

report_path = QC_DIR / f"methylene_blue_code9b_final_report_{timestamp}.txt"

table1_base.to_csv(table1_path, index=False, compression="gzip")

hemo.to_csv(hemo_path, index=False, compression="gzip")

valid_events.to_csv(events_path, index=False, compression="gzip")

availability_summary.to_csv(availability_path, index=False)

phenotype_summary.to_csv(phenotype_path, index=False)

physiology_summary.to_csv(physiology_path, index=False)

qc_summary.to_csv(qc_path, index=False)

validated_items.to_csv(itemids_path, index=False)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("METHYLENE BLUE — FINAL PRE-TREATMENT HEMODYNAMICS\n")

    file.write("=" * 75 + "\n\n")

    file.write(f"Input cohort: {PRIMARY_COHORT_PATH}\n")

    file.write(f"Primary cohort: {len(table1_base)}\n")

    file.write(
        "Invasive BP used: "
        f"{int(table1_base['bp_modality_selected'].eq('Invasive').sum())}\n"
    )

    file.write(
        "Non-invasive BP used: "
        f"{int(table1_base['bp_modality_selected'].eq('Non-invasive').sum())}\n"
    )

    file.write(
        "Phenotype classifiable: "
        f"{int(table1_base['phenotype_data_available'].sum())}\n"
    )

    file.write(f"Updated Table 1 base: {table1_path}\n")

print("\n" + "=" * 100)
print("OUTPUTS SALVOS")
print("=" * 100)

print(table1_path)
print(hemo_path)
print(events_path)
print(availability_path)
print(phenotype_path)
print(physiology_path)
print(qc_path)
print(report_path)

gc.collect()

print("\nCÓDIGO 9B FINAL CONCLUÍDO COM SUCESSO.")
