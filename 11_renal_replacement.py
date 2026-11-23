"""Source: original notebook cell 47. Run through run.py."""

from common import artifact, display, mimic_source, phase_dir, run_stamp, source_file

# ============================================================
# CÓDIGO 11D-B — RRT ATIVA NO MOMENTO DO AZUL
#
# Objetivo:
#   Criar a variável binária:
#
#       rrt_at_mb_baseline = True / False
#
# Fontes:
#   - procedureevents.csv.gz
#   - inputevents.csv.gz
#
# Evidências consideradas:
#
# 1. Procedimento de RRT com sobreposição ao horário do azul:
#       225441 — Hemodialysis
#       225802 — Dialysis - CRRT
#       225803 — Dialysis - CVVHD
#       225809 — Dialysis - CVVHDF
#       225955 — Dialysis - SCUF
#
# 2. Prismasate ativo no horário do azul:
#       225162 — Prismasate K2
#       225163 — Prismasate K4
#
# 3. Troca de filtro de CRRT em ±6 horas:
#       225436 — CRRT Filter Change
#
# Fonte original:

#
# Outputs:
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
print("CÓDIGO 11D-B — RRT ATIVA NO MOMENTO DO AZUL")
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
# 3. Localizar arquivos
# ------------------------------------------------------------

TABLE1_PATH = artifact(
    PROJECT_DIR, "TABLE1_BASE_with_final_mechanical_ventilation_*.csv.gz"
)

if TABLE1_PATH is None:

    print("\nArquivos disponíveis em cohorts:")

    for path in sorted(COHORT_DIR.glob("*.csv.gz")):
        print(path.name)

    raise FileNotFoundError("Não encontrei a base final com ventilação mecânica.")

PROCEDUREEVENTS_PATH = locate_source_file("procedureevents.csv.gz")

INPUTEVENTS_PATH = locate_source_file("inputevents.csv.gz")

print("\nBase selecionada:")
print(TABLE1_PATH)

print("\nProcedureevents:")
print(PROCEDUREEVENTS_PATH)

print("\nInputevents:")
print(INPUTEVENTS_PATH)

# ------------------------------------------------------------
# 4. Carregar a coorte
# ------------------------------------------------------------

cohort = pd.read_csv(TABLE1_PATH, compression="gzip", low_memory=False)

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
    raise RuntimeError("Colunas obrigatórias ausentes na base: " f"{missing_columns}")

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

if len(cohort) != 59:
    print("\nATENÇÃO: a coorte contém " f"{len(cohort)} pacientes, e não 59.")

# ------------------------------------------------------------
# 5. Itemids congelados
# ------------------------------------------------------------

RRT_PROCEDURE_ITEMIDS = {
    225441: "Hemodialysis",
    225802: "Dialysis - CRRT",
    225803: "Dialysis - CVVHD",
    225809: "Dialysis - CVVHDF",
    225955: "Dialysis - SCUF",
}

CRRT_FILTER_CHANGE_ITEMID = 225436

PRISMASATE_ITEMIDS = {225162: "Prismasate K2", 225163: "Prismasate K4"}

PROCEDURE_TARGET_ITEMIDS = set(RRT_PROCEDURE_ITEMIDS.keys()) | {
    CRRT_FILTER_CHANGE_ITEMID
}

INPUT_TARGET_ITEMIDS = set(PRISMASATE_ITEMIDS.keys())

# ------------------------------------------------------------
# 6. Tabela de tempos individuais
# ------------------------------------------------------------

time_table = cohort[["subject_id", "hadm_id", "stay_id", "mb_time_preliminary"]].copy()

time_table["filter_window_start"] = time_table["mb_time_preliminary"] - pd.Timedelta(
    hours=6
)

time_table["filter_window_end"] = time_table["mb_time_preliminary"] + pd.Timedelta(
    hours=6
)

target_stay_ids = set(cohort["stay_id"].dropna().astype(int).tolist())

# ------------------------------------------------------------
# 7. Ler procedureevents em chunks
# ------------------------------------------------------------

procedure_header = pd.read_csv(PROCEDUREEVENTS_PATH, compression="gzip", nrows=0)

procedure_available_columns = procedure_header.columns.tolist()

procedure_preferred_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "storetime",
    "itemid",
    "value",
    "valueuom",
    "statusdescription",
    "orderid",
    "linkorderid",
]

procedure_usecols = [
    column
    for column in procedure_preferred_columns
    if column in procedure_available_columns
]

procedure_required_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "itemid",
]

procedure_missing_columns = [
    column for column in procedure_required_columns if column not in procedure_usecols
]

if procedure_missing_columns:
    raise RuntimeError(
        "Colunas obrigatórias ausentes em procedureevents: "
        f"{procedure_missing_columns}"
    )

procedure_chunks = []

procedure_total_rows = 0
procedure_target_rows = 0

procedure_reader = pd.read_csv(
    PROCEDUREEVENTS_PATH,
    compression="gzip",
    usecols=procedure_usecols,
    chunksize=250_000,
    low_memory=False,
)

print("\nLendo procedureevents...")

for chunk_number, chunk in enumerate(procedure_reader, start=1):

    procedure_total_rows += len(chunk)

    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")

    chunk = chunk[chunk["itemid"].isin(PROCEDURE_TARGET_ITEMIDS)].copy()

    if chunk.empty:

        if chunk_number % 10 == 0:
            print(
                f"Procedure chunk {chunk_number:03d} | "
                f"linhas lidas: {procedure_total_rows:,}"
            )

        gc.collect()
        continue

    chunk = chunk[chunk["stay_id"].isin(target_stay_ids)].copy()

    procedure_target_rows += len(chunk)

    if chunk.empty:
        gc.collect()
        continue

    for column in ["subject_id", "hadm_id", "stay_id"]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("Int64")

    chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")

    chunk["endtime"] = pd.to_datetime(chunk["endtime"], errors="coerce")

    chunk = chunk.merge(
        time_table,
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
        validate="many_to_one",
    )

    procedure_chunks.append(chunk)

    gc.collect()

if procedure_chunks:

    procedure_events = pd.concat(procedure_chunks, ignore_index=True)

else:

    procedure_events = pd.DataFrame()

del procedure_chunks
gc.collect()

# ------------------------------------------------------------
# 8. Classificar evidências de procedureevents
# ------------------------------------------------------------

if procedure_events.empty:

    overlapping_rrt_procedures = pd.DataFrame()
    nearby_filter_changes = pd.DataFrame()

else:

    procedure_events["procedure_label"] = (
        procedure_events["itemid"]
        .astype(int)
        .map({**RRT_PROCEDURE_ITEMIDS, CRRT_FILTER_CHANGE_ITEMID: "CRRT Filter Change"})
    )

    rrt_procedure_mask = (
        procedure_events["itemid"].isin(RRT_PROCEDURE_ITEMIDS.keys())
        & procedure_events["starttime"].notna()
        & procedure_events["endtime"].notna()
        & (procedure_events["starttime"] <= procedure_events["mb_time_preliminary"])
        & (procedure_events["endtime"] >= procedure_events["mb_time_preliminary"])
    )

    overlapping_rrt_procedures = procedure_events[rrt_procedure_mask].copy()

    filter_change_mask = (
        procedure_events["itemid"].eq(CRRT_FILTER_CHANGE_ITEMID)
        & procedure_events["starttime"].notna()
        & (procedure_events["starttime"] >= procedure_events["filter_window_start"])
        & (procedure_events["starttime"] <= procedure_events["filter_window_end"])
    )

    nearby_filter_changes = procedure_events[filter_change_mask].copy()

    nearby_filter_changes["hours_filter_change_from_mb"] = (
        nearby_filter_changes["starttime"]
        - nearby_filter_changes["mb_time_preliminary"]
    ).dt.total_seconds() / 3600

# ------------------------------------------------------------
# 9. Ler inputevents em chunks
# ------------------------------------------------------------

input_header = pd.read_csv(INPUTEVENTS_PATH, compression="gzip", nrows=0)

input_available_columns = input_header.columns.tolist()

input_preferred_columns = [
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
    "statusdescription",
    "orderid",
    "linkorderid",
]

input_usecols = [
    column for column in input_preferred_columns if column in input_available_columns
]

input_required_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "itemid",
]

input_missing_columns = [
    column for column in input_required_columns if column not in input_usecols
]

if input_missing_columns:
    raise RuntimeError(
        "Colunas obrigatórias ausentes em inputevents: " f"{input_missing_columns}"
    )

input_chunks = []

input_total_rows = 0
input_target_rows = 0

input_reader = pd.read_csv(
    INPUTEVENTS_PATH,
    compression="gzip",
    usecols=input_usecols,
    chunksize=300_000,
    low_memory=False,
)

print("\nLendo inputevents...")

for chunk_number, chunk in enumerate(input_reader, start=1):

    input_total_rows += len(chunk)

    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")

    chunk = chunk[chunk["itemid"].isin(INPUT_TARGET_ITEMIDS)].copy()

    if chunk.empty:

        if chunk_number % 10 == 0:
            print(
                f"Input chunk {chunk_number:03d} | "
                f"linhas lidas: {input_total_rows:,}"
            )

        gc.collect()
        continue

    chunk = chunk[chunk["stay_id"].isin(target_stay_ids)].copy()

    input_target_rows += len(chunk)

    if chunk.empty:
        gc.collect()
        continue

    for column in ["subject_id", "hadm_id", "stay_id"]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("Int64")

    chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")

    chunk["endtime"] = pd.to_datetime(chunk["endtime"], errors="coerce")

    chunk = chunk.merge(
        time_table,
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
        validate="many_to_one",
    )

    input_chunks.append(chunk)

    gc.collect()

if input_chunks:

    prismasate_events = pd.concat(input_chunks, ignore_index=True)

else:

    prismasate_events = pd.DataFrame()

del input_chunks
gc.collect()

# ------------------------------------------------------------
# 10. Prismasate ativo no horário do azul
# ------------------------------------------------------------

if prismasate_events.empty:

    active_prismasate = pd.DataFrame()

else:

    prismasate_events["input_label"] = (
        prismasate_events["itemid"].astype(int).map(PRISMASATE_ITEMIDS)
    )

    prismasate_overlap_mask = (
        prismasate_events["starttime"].notna()
        & prismasate_events["endtime"].notna()
        & (prismasate_events["starttime"] <= prismasate_events["mb_time_preliminary"])
        & (prismasate_events["endtime"] >= prismasate_events["mb_time_preliminary"])
    )

    active_prismasate = prismasate_events[prismasate_overlap_mask].copy()

# ------------------------------------------------------------
# 11. Consolidar evidências por paciente
# ------------------------------------------------------------

identifiers = ["subject_id", "hadm_id", "stay_id"]

rrt_evidence = cohort[identifiers].copy()


def patient_flag(data, output_column):

    if data.empty:

        return pd.DataFrame(columns=(identifiers + [output_column]))

    result = data[identifiers].drop_duplicates().copy()

    result[output_column] = True

    return result


procedure_flag = patient_flag(overlapping_rrt_procedures, "rrt_procedure_overlap")

prismasate_flag = patient_flag(active_prismasate, "active_prismasate")

filter_change_flag = patient_flag(nearby_filter_changes, "nearby_crrt_filter_change")

rrt_evidence = (
    rrt_evidence.merge(
        procedure_flag, on=identifiers, how="left", validate="one_to_one"
    )
    .merge(prismasate_flag, on=identifiers, how="left", validate="one_to_one")
    .merge(filter_change_flag, on=identifiers, how="left", validate="one_to_one")
)

for column in [
    "rrt_procedure_overlap",
    "active_prismasate",
    "nearby_crrt_filter_change",
]:

    rrt_evidence[column] = rrt_evidence[column].fillna(False).astype(bool)

rrt_evidence["rrt_at_mb_baseline"] = (
    rrt_evidence["rrt_procedure_overlap"]
    | rrt_evidence["active_prismasate"]
    | rrt_evidence["nearby_crrt_filter_change"]
)

rrt_evidence["rrt_evidence_source"] = np.select(
    [
        (rrt_evidence["rrt_procedure_overlap"] & rrt_evidence["active_prismasate"]),
        rrt_evidence["rrt_procedure_overlap"],
        rrt_evidence["active_prismasate"],
        rrt_evidence["nearby_crrt_filter_change"],
    ],
    ["Procedure + Prismasate", "Procedure", "Prismasate", "CRRT filter change"],
    default="None",
)

# ------------------------------------------------------------
# 12. Atualizar a base final da Tabela 1
# ------------------------------------------------------------

rrt_columns = [
    "rrt_procedure_overlap",
    "active_prismasate",
    "nearby_crrt_filter_change",
    "rrt_at_mb_baseline",
    "rrt_evidence_source",
]

cohort_clean = cohort.drop(
    columns=[column for column in rrt_columns if column in cohort.columns],
    errors="ignore",
)

table1_final_base = cohort_clean.merge(
    rrt_evidence, on=identifiers, how="left", validate="one_to_one"
)

# ------------------------------------------------------------
# 13. Resumo geral
# ------------------------------------------------------------

rrt_summary = pd.DataFrame(
    {
        "variable": [
            "RRT active at MB baseline",
            "Procedure overlap",
            "Active Prismasate",
            "CRRT filter change within ±6 h",
        ],
        "n_positive": [
            int(table1_final_base["rrt_at_mb_baseline"].sum()),
            int(table1_final_base["rrt_procedure_overlap"].sum()),
            int(table1_final_base["active_prismasate"].sum()),
            int(table1_final_base["nearby_crrt_filter_change"].sum()),
        ],
        "n_total": len(table1_final_base),
    }
)

rrt_summary["percent"] = 100 * rrt_summary["n_positive"] / rrt_summary["n_total"]

# ------------------------------------------------------------
# 14. RRT por fenótipo
# ------------------------------------------------------------

rrt_by_phenotype = table1_final_base.groupby(
    "hemodynamic_phenotype", as_index=False
).agg(n_patients=("subject_id", "nunique"), rrt_n=("rrt_at_mb_baseline", "sum"))

rrt_by_phenotype["rrt_percent"] = (
    100 * rrt_by_phenotype["rrt_n"] / rrt_by_phenotype["n_patients"]
)

# ------------------------------------------------------------
# 15. Mostrar resultados
# ------------------------------------------------------------

pd.set_option("display.max_rows", 200)

pd.set_option("display.max_columns", 40)

pd.set_option("display.width", 280)

print("\n" + "=" * 100)
print("RESUMO DE RRT BASAL")
print("=" * 100)

display(rrt_summary)

print("\n" + "=" * 100)
print("RRT BASAL POR FENÓTIPO")
print("=" * 100)

display(rrt_by_phenotype)

print("\n" + "=" * 100)
print("FONTES DE EVIDÊNCIA DE RRT")
print("=" * 100)

display(
    table1_final_base.loc[
        table1_final_base["rrt_at_mb_baseline"],
        identifiers
        + [
            "hemodynamic_phenotype",
            "rrt_procedure_overlap",
            "active_prismasate",
            "nearby_crrt_filter_change",
            "rrt_evidence_source",
        ],
    ].sort_values(["rrt_evidence_source", "subject_id"])
)

# ------------------------------------------------------------
# 16. Controle de qualidade
# ------------------------------------------------------------

qc_summary = pd.DataFrame(
    {
        "metric": [
            "Patients in Table 1 cohort",
            "Patients classified for RRT",
            "Patients with RRT at baseline",
            "Patients without RRT at baseline",
            "RRT procedure overlap records",
            "Active Prismasate records",
            "Nearby CRRT filter-change records",
        ],
        "value": [
            len(table1_final_base),
            table1_final_base["rrt_at_mb_baseline"].notna().sum(),
            table1_final_base["rrt_at_mb_baseline"].sum(),
            (~table1_final_base["rrt_at_mb_baseline"]).sum(),
            len(overlapping_rrt_procedures),
            len(active_prismasate),
            len(nearby_filter_changes),
        ],
    }
)

print("\n" + "=" * 100)
print("CONTROLE DE QUALIDADE")
print("=" * 100)

display(qc_summary)

# ------------------------------------------------------------
# 17. Salvar outputs
# ------------------------------------------------------------

timestamp = run_stamp()

table1_final_path = COHORT_DIR / f"TABLE1_BASE_FINAL_COMPLETE_{timestamp}.csv.gz"

rrt_evidence_path = (
    COHORT_DIR / f"methylene_blue_baseline_RRT_evidence_{timestamp}.csv.gz"
)

procedure_events_path = (
    COHORT_DIR / f"methylene_blue_RRT_procedure_events_{timestamp}.csv.gz"
)

prismasate_events_path = (
    COHORT_DIR / f"methylene_blue_active_prismasate_events_{timestamp}.csv.gz"
)

filter_change_path = (
    COHORT_DIR / f"methylene_blue_nearby_CRTT_filter_changes_{timestamp}.csv.gz"
)

rrt_summary_path = TABLE_DIR / f"TABLE1_baseline_RRT_summary_{timestamp}.csv"

rrt_phenotype_path = TABLE_DIR / f"TABLE1_baseline_RRT_by_phenotype_{timestamp}.csv"

qc_path = QC_DIR / f"methylene_blue_code11d_b_RRT_qc_{timestamp}.csv"

report_path = QC_DIR / f"methylene_blue_code11d_b_RRT_report_{timestamp}.txt"

table1_final_base.to_csv(table1_final_path, index=False, compression="gzip")

rrt_evidence.to_csv(rrt_evidence_path, index=False, compression="gzip")

overlapping_rrt_procedures.to_csv(
    procedure_events_path, index=False, compression="gzip"
)

active_prismasate.to_csv(prismasate_events_path, index=False, compression="gzip")

nearby_filter_changes.to_csv(filter_change_path, index=False, compression="gzip")

rrt_summary.to_csv(rrt_summary_path, index=False)

rrt_by_phenotype.to_csv(rrt_phenotype_path, index=False)

qc_summary.to_csv(qc_path, index=False)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("METHYLENE BLUE — BASELINE RENAL REPLACEMENT THERAPY\n")

    file.write("=" * 75 + "\n\n")

    file.write(f"Input cohort: {TABLE1_PATH}\n")

    file.write(f"Patients: {len(table1_final_base)}\n")

    file.write(
        "Patients with RRT at baseline: "
        f"{int(table1_final_base['rrt_at_mb_baseline'].sum())}\n"
    )

    file.write(f"Final complete Table 1 base: {table1_final_path}\n")

print("\n" + "=" * 100)
print("OUTPUTS SALVOS")
print("=" * 100)

print(table1_final_path)
print(rrt_evidence_path)
print(procedure_events_path)
print(prismasate_events_path)
print(filter_change_path)
print(rrt_summary_path)
print(rrt_phenotype_path)
print(qc_path)
print(report_path)

gc.collect()

print("\nCÓDIGO 11D-B CONCLUÍDO COM SUCESSO.")
