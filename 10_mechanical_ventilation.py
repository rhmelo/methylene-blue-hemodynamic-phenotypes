"""Source: original notebook cell 41, 44 and 45 (final classification). Run through run.py."""

from common import artifact, display, mimic_source, phase_dir, run_stamp, source_file

# ============================================================
# CÓDIGO 11C — VENTILAÇÃO MECÂNICA INVASIVA NO BASAL
#
# Objetivo:
#   Identificar se o paciente estava em ventilação mecânica
#   invasiva no momento do azul de metileno.
#
# Estratégia:
#   - usar documentação de dispositivo de oxigênio e modo
#     ventilatório;
#   - buscar o último estado documentado nas 14 horas anteriores
#     ao azul;
#   - distinguir:
#       InvasiveVent
#       NonInvasiveVent
#       HFNC
#       SupplementalOxygen
#       Tracheostomy
#       None
#       Unclassified
#
# A variável que entrará na Tabela 1 será:
#   invasive_mechanical_ventilation_at_mb = True/False
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
TABLE_DIR = PROJECT_DIR / "tables"
QC_DIR = PROJECT_DIR / "qc"
METADATA_DIR = PROJECT_DIR / "metadata"

for directory in [COHORT_DIR, TABLE_DIR, QC_DIR, METADATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("CÓDIGO 11C — VENTILAÇÃO MECÂNICA INVASIVA NO BASAL")
print("=" * 100)

# ------------------------------------------------------------
# 2. Funções auxiliares
# ------------------------------------------------------------


def locate_source_file(filename):
    return source_file(SOURCE_DIR, filename)


def latest_file(pattern):
    return artifact(PROJECT_DIR, pattern)


def normalize_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip().lower().replace("\xa0", " ")


# ------------------------------------------------------------
# 3. Localizar arquivos
# ------------------------------------------------------------

TABLE1_NEE_PATH = latest_file("TABLE1_BASE_with_baseline_NEE_*.csv.gz")

if TABLE1_NEE_PATH is None:

    raise FileNotFoundError("Não encontrei TABLE1_BASE_with_baseline_NEE.")

D_ITEMS_PATH = locate_source_file("d_items.csv.gz")

CHARTEVENTS_PATH = locate_source_file("chartevents.csv.gz")

print("\nBase selecionada:")
print(TABLE1_NEE_PATH)

print("\nChartevents:")
print(CHARTEVENTS_PATH)

# ------------------------------------------------------------
# 4. Carregar a coorte
# ------------------------------------------------------------

cohort = pd.read_csv(TABLE1_NEE_PATH, compression="gzip", low_memory=False)

required_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "mb_time_preliminary",
    "hemodynamic_phenotype",
]

missing_columns = [
    column for column in required_columns if column not in cohort.columns
]

if missing_columns:
    raise RuntimeError(f"Colunas obrigatórias ausentes: {missing_columns}")

for column in ["subject_id", "hadm_id", "stay_id"]:
    cohort[column] = pd.to_numeric(cohort[column], errors="coerce").astype("Int64")

cohort["mb_time_preliminary"] = pd.to_datetime(
    cohort["mb_time_preliminary"], errors="coerce"
)

cohort = (
    cohort.sort_values(["subject_id", "mb_time_preliminary"])
    .drop_duplicates(subset=["subject_id"], keep="first")
    .reset_index(drop=True)
)

print("\nPacientes na coorte:", len(cohort))

# ------------------------------------------------------------
# 5. Identificar itemids de modo e dispositivo
# ------------------------------------------------------------

d_items = pd.read_csv(D_ITEMS_PATH, compression="gzip", low_memory=False)

d_items["itemid"] = pd.to_numeric(d_items["itemid"], errors="coerce")

d_items["_label_normalized"] = (
    d_items["label"].fillna("").astype(str).str.lower().str.strip()
)

candidate_mask = d_items["_label_normalized"].str.contains(
    r"ventilator mode|o2 delivery device|oxygen delivery device", regex=True, na=False
)

ventilation_items = (
    d_items.loc[candidate_mask][
        [
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
    ]
    .drop_duplicates()
    .sort_values("itemid")
    .reset_index(drop=True)
)

# Manter somente itens provenientes de chartevents
if "linksto" in ventilation_items.columns:

    ventilation_items = ventilation_items[
        ventilation_items["linksto"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("chartevents")
    ].copy()

if ventilation_items.empty:

    raise RuntimeError(
        "Nenhum item de modo ventilatório ou dispositivo "
        "de oxigênio foi identificado no d_items."
    )

TARGET_ITEMIDS = set(ventilation_items["itemid"].dropna().astype(int).tolist())

print("\n" + "=" * 100)
print("ITEMIDS DE VENTILAÇÃO IDENTIFICADOS")
print("=" * 100)

display(ventilation_items)

# ------------------------------------------------------------
# 6. Definir valores reconhecidos
# ------------------------------------------------------------

INVASIVE_MODES = {
    "(s) cmv",
    "aprv",
    "aprv/biphasic+apnpress",
    "aprv/biphasic+apnvol",
    "apv (cmv)",
    "ambient",
    "apnea ventilation",
    "cmv",
    "cmv/assist",
    "cmv/assist/autoflow",
    "cmv/autoflow",
    "cpap/pps",
    "cpap/psv",
    "cpap/psv+apn tcpl",
    "cpap/psv+apnpres",
    "cpap/psv+apnvol",
    "mmv",
    "mmv/autoflow",
    "mmv/psv",
    "mmv/psv/autoflow",
    "p-cmv",
    "pcv+",
    "pcv+/psv",
    "pcv+assist",
    "pres/ac",
    "prvc/ac",
    "prvc/simv",
    "psv/sbt",
    "simv",
    "simv/autoflow",
    "simv/pres",
    "simv/psv",
    "simv/psv/autoflow",
    "simv/vol",
    "synchron master",
    "synchron slave",
    "vol/ac",
    "apv (simv)",
    "p-simv",
    "vs",
    "asv",
}

NONINVASIVE_MODES = {"duopap", "niv", "niv-st"}

INVASIVE_DEVICES = {"endotracheal tube"}

TRACHEOSTOMY_DEVICES = {"tracheostomy tube", "trach mask"}

NONINVASIVE_DEVICES = {"bipap mask", "cpap mask"}

HFNC_DEVICES = {"high flow nasal cannula"}

SUPPLEMENTAL_OXYGEN_DEVICES = {
    "non-rebreather",
    "face tent",
    "aerosol-cool",
    "venti mask",
    "medium conc mask",
    "ultrasonic neb",
    "vapomist",
    "oxymizer",
    "high flow neb",
    "nasal cannula",
}

NO_OXYGEN_DEVICES = {"none"}

# ------------------------------------------------------------
# 7. Preparar janelas individuais
# ------------------------------------------------------------

time_table = cohort[["subject_id", "hadm_id", "stay_id", "mb_time_preliminary"]].copy()

time_table["vent_window_start"] = time_table["mb_time_preliminary"] - pd.Timedelta(
    hours=14
)

time_table["vent_window_end"] = time_table["mb_time_preliminary"]

target_stay_ids = set(cohort["stay_id"].dropna().astype(int).tolist())

# ------------------------------------------------------------
# 8. Inspecionar cabeçalho do chartevents
# ------------------------------------------------------------

chart_header = pd.read_csv(CHARTEVENTS_PATH, compression="gzip", nrows=0)

available_columns = chart_header.columns.tolist()

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

chart_usecols = [column for column in preferred_columns if column in available_columns]

required_chart_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "charttime",
    "itemid",
    "value",
]

missing_chart_columns = [
    column for column in required_chart_columns if column not in chart_usecols
]

if missing_chart_columns:
    raise RuntimeError(
        "Colunas obrigatórias ausentes em chartevents: " f"{missing_chart_columns}"
    )

# ------------------------------------------------------------
# 9. Ler chartevents em chunks
# ------------------------------------------------------------

chunk_size = 500_000

matched_chunks = []

total_rows = 0
total_window_rows = 0

print("\nIniciando leitura de chartevents...")
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

        if chunk_number % 20 == 0:
            print(
                f"Chunk {chunk_number:03d} | "
                f"linhas lidas: {total_rows:,} | "
                f"eventos acumulados: {total_window_rows:,}"
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

    chunk = chunk.merge(
        time_table,
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
        validate="many_to_one",
    )

    chunk = chunk[
        chunk["charttime"].notna()
        & (chunk["charttime"] >= chunk["vent_window_start"])
        & (chunk["charttime"] <= chunk["vent_window_end"])
    ].copy()

    if not chunk.empty:

        chunk["value_normalized"] = chunk["value"].apply(normalize_text)

        matched_chunks.append(chunk)

        total_window_rows += len(chunk)

    if chunk_number % 10 == 0:

        print(
            f"Chunk {chunk_number:03d} | "
            f"linhas lidas: {total_rows:,} | "
            f"eventos acumulados: {total_window_rows:,}"
        )

    gc.collect()

# ------------------------------------------------------------
# 10. Consolidar eventos
# ------------------------------------------------------------

if matched_chunks:

    ventilation_events = pd.concat(matched_chunks, ignore_index=True)

else:

    ventilation_events = pd.DataFrame()

del matched_chunks
gc.collect()

print("\nEventos ventilatórios encontrados:", len(ventilation_events))

if ventilation_events.empty:
    raise RuntimeError("Nenhum evento ventilatório foi encontrado.")

# ============================================================
# CORREÇÃO DO CÓDIGO 11C
# CLASSIFICAÇÃO HIERÁRQUICA DO SUPORTE VENTILATÓRIO
#
# Prioridade:
#   1. Dispositivo explícito
#   2. Modo ventilatório inequívoco
#   3. Modos ambíguos permanecem não classificados
#
# SPONT e CPAP isolados NÃO serão considerados VM invasiva.
#
# Não relê chartevents.
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# 1. Preparar eventos válidos
# ------------------------------------------------------------

vent_events_corrected = ventilation_events.copy()

vent_events_corrected["value_normalized"] = (
    vent_events_corrected["value"].fillna("").astype(str).str.strip().str.lower()
)

# Ignorar registros vazios
vent_events_corrected = vent_events_corrected[
    vent_events_corrected["value_normalized"].ne("")
].copy()

# ------------------------------------------------------------
# 2. Identificar o tipo de item
# ------------------------------------------------------------

DEVICE_ITEMID = 226732

MODE_ITEMIDS = {223849, 229314}

vent_events_corrected["event_type"] = np.select(
    [
        vent_events_corrected["itemid"].eq(DEVICE_ITEMID),
        vent_events_corrected["itemid"].isin(MODE_ITEMIDS),
    ],
    ["device", "mode"],
    default="other",
)

# ------------------------------------------------------------
# 3. Conjuntos de dispositivos
# ------------------------------------------------------------

INVASIVE_AIRWAY_DEVICES = {
    "endotracheal tube",
    "ett",
    "endotracheal tube/trach",
    "endotracheal tube / trach",
}

TRACHEOSTOMY_DEVICES = {"tracheostomy tube", "trach tube"}

TRACH_SPONTANEOUS_DEVICES = {"trach mask", "tracheostomy mask"}

NONINVASIVE_DEVICES = {
    "bipap mask",
    "cpap mask",
    "non-invasive ventilation mask",
    "noninvasive ventilation mask",
}

HFNC_DEVICES = {"high flow nasal cannula", "high-flow nasal cannula", "hfnc"}

SUPPLEMENTAL_OXYGEN_DEVICES = {
    "nasal cannula",
    "non-rebreather",
    "non rebreather",
    "non-rebreather mask",
    "face mask",
    "face tent",
    "venti mask",
    "venturi mask",
    "simple mask",
    "aerosol mask",
    "oxymizer",
    "medium conc mask",
}

NO_SUPPORT_DEVICES = {"none", "room air"}

# ------------------------------------------------------------
# 4. Modos ventilatórios
#
# Somente modos inequivocamente ventilatórios entram como
# evidência de ventilação invasiva.
#
# SPONT e CPAP são deliberadamente excluídos.
# ------------------------------------------------------------

UNAMBIGUOUS_INVASIVE_MODES = {
    "ac",
    "a/c",
    "assist control",
    "assist-control",
    "cmv",
    "(s) cmv",
    "vol/ac",
    "volume control",
    "volume control/assist control",
    "pres/ac",
    "pressure control",
    "pressure control/assist control",
    "pcv",
    "pcv+",
    "prvc",
    "prvc/ac",
    "prvc/simv",
    "simv",
    "simv/vol",
    "simv/pres",
    "simv/psv",
    "aprv",
    "apv (cmv)",
    "apv (simv)",
    "apvcmv",
    "apvsimv",
    "p-cmv",
    "p-simv",
    "asv",
    "intellivent-asv",
}

UNAMBIGUOUS_NONINVASIVE_MODES = {"niv", "niv-st", "bipap"}

AMBIGUOUS_MODES = {
    "spont",
    "cpap",
    "cpap/psv",
    "psv",
    "pressure support",
    "psv/sbt",
    "sbt",
}

# ------------------------------------------------------------
# 5. Selecionar o último dispositivo e o último modo
# ------------------------------------------------------------

identifiers = ["subject_id", "hadm_id", "stay_id"]

device_events = vent_events_corrected[
    vent_events_corrected["event_type"].eq("device")
].copy()

mode_events = vent_events_corrected[
    vent_events_corrected["event_type"].eq("mode")
].copy()

if device_events.empty:

    latest_device = pd.DataFrame(
        columns=identifiers + ["latest_device_time", "latest_device"]
    )

else:

    latest_device = (
        device_events.sort_values("charttime")
        .drop_duplicates(subset=identifiers, keep="last")[
            identifiers + ["charttime", "value_normalized"]
        ]
        .rename(
            columns={
                "charttime": "latest_device_time",
                "value_normalized": "latest_device",
            }
        )
    )

if mode_events.empty:

    latest_mode = pd.DataFrame(
        columns=identifiers + ["latest_mode_time", "latest_mode"]
    )

else:

    latest_mode = (
        mode_events.sort_values("charttime")
        .drop_duplicates(subset=identifiers, keep="last")[
            identifiers + ["charttime", "value_normalized"]
        ]
        .rename(
            columns={"charttime": "latest_mode_time", "value_normalized": "latest_mode"}
        )
    )

# ------------------------------------------------------------
# 6. Integrar por paciente
# ------------------------------------------------------------

ventilation_corrected = (
    cohort[identifiers + ["mb_time_preliminary"]]
    .merge(latest_device, on=identifiers, how="left", validate="one_to_one")
    .merge(latest_mode, on=identifiers, how="left", validate="one_to_one")
)

# ------------------------------------------------------------
# 7. Classificação hierárquica
# ------------------------------------------------------------


def classify_patient_ventilation(row):

    device = (
        str(row["latest_device"]).strip().lower()
        if pd.notna(row["latest_device"])
        else ""
    )

    mode = (
        str(row["latest_mode"]).strip().lower() if pd.notna(row["latest_mode"]) else ""
    )

    # 1. Dispositivo invasivo inequívoco
    if device in INVASIVE_AIRWAY_DEVICES:
        return "InvasiveVent"

    # 2. Máscara de NIV explícita
    if device in NONINVASIVE_DEVICES:
        return "NonInvasiveVent"

    # 3. Alto fluxo
    if device in HFNC_DEVICES:
        return "HFNC"

    # 4. Oxigênio convencional
    if device in SUPPLEMENTAL_OXYGEN_DEVICES:
        return "SupplementalOxygen"

    # 5. Sem suporte
    if device in NO_SUPPORT_DEVICES:
        return "None"

    # 6. Trach mask não significa ventilação invasiva
    if device in TRACH_SPONTANEOUS_DEVICES:
        return "SupplementalOxygen"

    # 7. Traqueostomia + modo invasivo inequívoco
    if device in TRACHEOSTOMY_DEVICES and mode in UNAMBIGUOUS_INVASIVE_MODES:
        return "InvasiveVent"

    # 8. Modo invasivo inequívoco, sem dispositivo conflitante
    if mode in UNAMBIGUOUS_INVASIVE_MODES:
        return "InvasiveVent"

    # 9. Modo NIV inequívoco
    if mode in UNAMBIGUOUS_NONINVASIVE_MODES:
        return "NonInvasiveVent"

    # 10. SPONT, CPAP, PSV etc. isolados permanecem ambíguos
    if mode in AMBIGUOUS_MODES:
        return "Unclassified"

    return "Unclassified"


# ============================================================
# CORREÇÃO FINAL DA CLASSIFICAÇÃO VENTILATÓRIA
#
# Inclui CMV/ASSIST/AUTOFLOW como modo invasivo inequívoco.
# Não relê chartevents.
# ============================================================

UNAMBIGUOUS_INVASIVE_MODES.update({"cmv/assist/autoflow", "cmv/autoflow", "cmv/assist"})

ventilation_corrected["ventilation_status_corrected"] = ventilation_corrected.apply(
    classify_patient_ventilation, axis=1
)

ventilation_corrected["invasive_mechanical_ventilation_at_mb"] = ventilation_corrected[
    "ventilation_status_corrected"
].eq("InvasiveVent")

# Atualizar novamente a base
identifiers = ["subject_id", "hadm_id", "stay_id"]

columns_to_remove = [
    "latest_device_time",
    "latest_device",
    "latest_mode_time",
    "latest_mode",
    "ventilation_status_corrected",
    "invasive_mechanical_ventilation_at_mb",
]

table1_ventilation_final = cohort.drop(
    columns=[column for column in columns_to_remove if column in cohort.columns],
    errors="ignore",
)

table1_ventilation_final = table1_ventilation_final.merge(
    ventilation_corrected[
        identifiers
        + [
            "latest_device_time",
            "latest_device",
            "latest_mode_time",
            "latest_mode",
            "ventilation_status_corrected",
            "invasive_mechanical_ventilation_at_mb",
        ]
    ],
    on=identifiers,
    how="left",
    validate="one_to_one",
)

# Resumo final
status_summary_final = (
    table1_ventilation_final["ventilation_status_corrected"]
    .value_counts(dropna=False)
    .rename_axis("ventilation_status")
    .reset_index(name="n_patients")
)

status_summary_final["percent"] = (
    100 * status_summary_final["n_patients"] / len(table1_ventilation_final)
)

vm_by_phenotype_final = table1_ventilation_final.groupby(
    "hemodynamic_phenotype", as_index=False
).agg(
    n_patients=("subject_id", "nunique"),
    invasive_vm_n=("invasive_mechanical_ventilation_at_mb", "sum"),
    unclassified_n=(
        "ventilation_status_corrected",
        lambda x: int((x == "Unclassified").sum()),
    ),
)

vm_by_phenotype_final["invasive_vm_percent_of_total"] = (
    100 * vm_by_phenotype_final["invasive_vm_n"] / vm_by_phenotype_final["n_patients"]
)

print("=" * 100)
print("ESTADO VENTILATÓRIO FINAL")
print("=" * 100)

display(status_summary_final)

print("\n" + "=" * 100)
print("VENTILAÇÃO INVASIVA POR FENÓTIPO — FINAL")
print("=" * 100)

display(vm_by_phenotype_final)

print("\n" + "=" * 100)
print("PACIENTE AINDA NÃO CLASSIFICADO")
print("=" * 100)

display(
    table1_ventilation_final[
        table1_ventilation_final["ventilation_status_corrected"].eq("Unclassified")
    ][
        identifiers
        + [
            "mb_time_preliminary",
            "latest_device",
            "latest_device_time",
            "latest_mode",
            "latest_mode_time",
        ]
    ]
)

# Salvar versão final

PROJECT_DIR = phase_dir("mimic_iv")

COHORT_DIR = PROJECT_DIR / "cohorts"
TABLE_DIR = PROJECT_DIR / "tables"
QC_DIR = PROJECT_DIR / "qc"

timestamp = run_stamp()

table1_final_path = (
    COHORT_DIR / f"TABLE1_BASE_with_final_mechanical_ventilation_{timestamp}.csv.gz"
)

status_final_path = TABLE_DIR / f"TABLE1_final_ventilation_status_{timestamp}.csv"

phenotype_final_path = TABLE_DIR / f"TABLE1_final_VM_by_phenotype_{timestamp}.csv"

table1_ventilation_final.to_csv(table1_final_path, index=False, compression="gzip")

status_summary_final.to_csv(status_final_path, index=False)

vm_by_phenotype_final.to_csv(phenotype_final_path, index=False)

print("\nArquivos salvos:")
print(table1_final_path)
print(status_final_path)
print(phenotype_final_path)
