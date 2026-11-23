"""Source: original notebook cell 39. Run through run.py."""

from common import artifact, display, mimic_source, phase_dir, run_stamp, source_file

# ============================================================
# CÓDIGO 11B — VASOATIVOS BASAIS E NEE
#
# População:
#   59 pacientes com fenótipo classificável e laboratórios
#   basais já integrados.
#
# Janela basal:
#   -60 a 0 minutos em relação ao horário preliminar do azul.
#
# Estratégia:
#   1. Extrair infusões vasoativas sobrepostas à janela
#   2. Padronizar as doses
#   3. Reconstruir a dose ativa em cada minuto
#   4. Calcular NEE em cada minuto
#   5. Definir NEE basal como a mediana dos 61 minutos
#
# Fórmula:
#   NEE =
#       norepinephrine
#       + epinephrine
#       + phenylephrine / 10
#       + dopamine / 100
#       + vasopressin * 2.5
#       + angiotensin II * 10
#
# Dobutamina:
#   Extraída separadamente e NÃO incluída na NEE.
#
# Fonte:

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

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Pasta-fonte não encontrada: {SOURCE_DIR}")

print("=" * 100)
print("CÓDIGO 11B — VASOATIVOS BASAIS E NEE")
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


def normalize_unit(unit):

    if pd.isna(unit):
        return ""

    unit = str(unit).strip().lower()

    unit = (
        unit.replace("μ", "u")
        .replace("µ", "u")
        .replace("micrograms", "mcg")
        .replace("microgram", "mcg")
        .replace("units", "unit")
        .replace("international unit", "unit")
        .replace("iu", "unit")
        .replace(" ", "")
    )

    return unit


# ------------------------------------------------------------
# 3. Localizar a base com laboratórios
# ------------------------------------------------------------

TABLE1_LABS_PATH = find_latest_file(["TABLE1_BASE_with_baseline_labs_*.csv.gz"])

if TABLE1_LABS_PATH is None:

    print("\nArquivos disponíveis em cohorts:")

    for path in sorted(COHORT_DIR.glob("*.csv.gz")):
        print(path.name)

    raise FileNotFoundError("Não encontrei TABLE1_BASE_with_baseline_labs.")

INPUTEVENTS_PATH = locate_source_file("inputevents.csv.gz")

D_ITEMS_PATH = locate_source_file("d_items.csv.gz")

print("\nBase selecionada:")
print(TABLE1_LABS_PATH)

print("\nInputevents:")
print(INPUTEVENTS_PATH)

# ------------------------------------------------------------
# 4. Carregar a coorte
# ------------------------------------------------------------

cohort = pd.read_csv(TABLE1_LABS_PATH, compression="gzip", low_memory=False)

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
    raise RuntimeError("Colunas obrigatórias ausentes: " f"{missing_columns}")

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

print("\nCoorte carregada:")
print("Pacientes:", len(cohort))

if len(cohort) != 59:
    print("\nATENÇÃO: a coorte contém " f"{len(cohort)} pacientes, e não 59.")

# ------------------------------------------------------------
# 5. Itemids congelados
# ------------------------------------------------------------

DRUG_ITEM_MAP = {
    # Norepinefrina
    221906: "norepinephrine",
    # Epinefrina
    221289: "epinephrine",
    229617: "epinephrine",
    # Vasopressina
    222315: "vasopressin",
    # Fenilefrina
    221749: "phenylephrine",
    229630: "phenylephrine",
    229631: "phenylephrine",
    229632: "phenylephrine",
    # Dopamina
    221662: "dopamine",
    # Angiotensina II
    229709: "angiotensin_ii",
    229764: "angiotensin_ii",
    # Dobutamina — fora da NEE
    221653: "dobutamine",
}

TARGET_ITEMIDS = set(DRUG_ITEM_MAP.keys())

# ------------------------------------------------------------
# 6. Validar os itens
# ------------------------------------------------------------

d_items = pd.read_csv(D_ITEMS_PATH, compression="gzip", low_memory=False)

d_items["itemid"] = pd.to_numeric(d_items["itemid"], errors="coerce")

item_columns = [
    column
    for column in ["itemid", "label", "abbreviation", "category", "unitname", "linksto"]
    if column in d_items.columns
]

validated_items = (
    d_items[d_items["itemid"].isin(TARGET_ITEMIDS)][item_columns]
    .drop_duplicates()
    .sort_values("itemid")
    .reset_index(drop=True)
)

validated_items["drug_variable"] = (
    validated_items["itemid"].astype(int).map(DRUG_ITEM_MAP)
)

print("\n" + "=" * 100)
print("ITEMIDS UTILIZADOS")
print("=" * 100)

display(validated_items)

# ------------------------------------------------------------
# 7. Janelas individuais
# ------------------------------------------------------------

time_table = cohort[["subject_id", "hadm_id", "stay_id", "mb_time_preliminary"]].copy()

time_table["window_start"] = time_table["mb_time_preliminary"] - pd.Timedelta(
    minutes=60
)

time_table["window_end"] = time_table["mb_time_preliminary"]

target_stay_ids = set(cohort["stay_id"].dropna().astype(int).tolist())

# ------------------------------------------------------------
# 8. Inspecionar inputevents
# ------------------------------------------------------------

input_header = pd.read_csv(INPUTEVENTS_PATH, compression="gzip", nrows=0)

available_columns = input_header.columns.tolist()

preferred_columns = [
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
    "patientweight",
    "statusdescription",
    "originalamount",
    "originalrate",
]

input_usecols = [column for column in preferred_columns if column in available_columns]

required_input_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "itemid",
    "rate",
    "rateuom",
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
# 9. Ler inputevents em chunks
# ------------------------------------------------------------

chunk_size = 300_000

matched_chunks = []

total_rows = 0
total_target_rows = 0
total_window_overlap_rows = 0

print("\nIniciando leitura de inputevents...")
print(f"Chunk size: {chunk_size:,}")

reader = pd.read_csv(
    INPUTEVENTS_PATH,
    compression="gzip",
    usecols=input_usecols,
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
                f"eventos acumulados: {total_window_overlap_rows:,}"
            )

        gc.collect()
        continue

    chunk = chunk[chunk["stay_id"].isin(target_stay_ids)].copy()

    total_target_rows += len(chunk)

    if chunk.empty:
        gc.collect()
        continue

    for column in ["subject_id", "hadm_id", "stay_id"]:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("Int64")

    chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")

    chunk["endtime"] = pd.to_datetime(chunk["endtime"], errors="coerce")

    chunk["rate"] = pd.to_numeric(chunk["rate"], errors="coerce")

    if "patientweight" in chunk.columns:
        chunk["patientweight"] = pd.to_numeric(chunk["patientweight"], errors="coerce")
    else:
        chunk["patientweight"] = np.nan

    chunk = chunk.merge(
        time_table,
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
        validate="many_to_one",
    )

    # Qualquer sobreposição com a janela basal
    overlap_mask = (
        chunk["starttime"].notna()
        & chunk["endtime"].notna()
        & (chunk["starttime"] <= chunk["window_end"])
        & (chunk["endtime"] >= chunk["window_start"])
    )

    chunk = chunk[overlap_mask].copy()

    if not chunk.empty:

        chunk["drug"] = chunk["itemid"].astype(int).map(DRUG_ITEM_MAP)

        matched_chunks.append(chunk)

        total_window_overlap_rows += len(chunk)

    if chunk_number % 5 == 0:

        print(
            f"Chunk {chunk_number:03d} | "
            f"linhas lidas: {total_rows:,} | "
            f"eventos sobrepostos acumulados: "
            f"{total_window_overlap_rows:,}"
        )

    gc.collect()

# ------------------------------------------------------------
# 10. Consolidar os eventos
# ------------------------------------------------------------

if matched_chunks:

    vasoactive_events = pd.concat(matched_chunks, ignore_index=True)

else:

    vasoactive_events = pd.DataFrame()

del matched_chunks
gc.collect()

print("\n" + "=" * 100)
print("EXTRAÇÃO CONCLUÍDA")
print("=" * 100)

print("Linhas totais examinadas:", f"{total_rows:,}")
print("Registros-alvo nas UTIs:", f"{total_target_rows:,}")
print("Infusões com sobreposição à janela:", f"{len(vasoactive_events):,}")

if vasoactive_events.empty:
    raise RuntimeError("Nenhuma infusão vasoativa foi encontrada.")

# ------------------------------------------------------------
# 11. Normalizar unidades
# ------------------------------------------------------------

vasoactive_events["_rateuom"] = vasoactive_events["rateuom"].apply(normalize_unit)

vasoactive_events["rate_standardized"] = np.nan
vasoactive_events["standard_unit"] = ""
vasoactive_events["conversion_method"] = ""
vasoactive_events["conversion_valid"] = False
vasoactive_events["conversion_issue"] = ""

# ------------------------------------------------------------
# 12. Conversão das doses
# ------------------------------------------------------------


def convert_rate(row):

    drug = row["drug"]
    rate = row["rate"]
    unit = row["_rateuom"]
    weight = row["patientweight"]

    if pd.isna(rate):
        return (np.nan, "", "", False, "missing_rate")

    if rate < 0:
        return (np.nan, "", "", False, "negative_rate")

    # --------------------------------------------------------
    # Vasopressina -> U/min
    # --------------------------------------------------------

    if drug == "vasopressin":

        if unit in ["unit/min", "units/min", "u/min"]:
            return (rate, "U/min", "direct_U_per_min", True, "")

        if unit in ["unit/hour", "unit/hr", "units/hour", "units/hr", "u/hour", "u/hr"]:
            return (rate / 60, "U/min", "U_per_hour_divided_by_60", True, "")

        return (np.nan, "U/min", "", False, f"unexpected_vasopressin_unit:{unit}")

    # --------------------------------------------------------
    # Angiotensina II -> mcg/kg/min
    # --------------------------------------------------------

    if drug == "angiotensin_ii":

        if unit in ["ng/kg/min", "ng/kg/minute"]:
            return (
                rate / 1000,
                "mcg/kg/min",
                "ng_per_kg_min_divided_by_1000",
                True,
                "",
            )

        if unit in ["mcg/kg/min", "ug/kg/min"]:
            return (rate, "mcg/kg/min", "direct_mcg_per_kg_min", True, "")

        return (np.nan, "mcg/kg/min", "", False, f"unexpected_angiotensin_unit:{unit}")

    # --------------------------------------------------------
    # Demais drogas -> mcg/kg/min
    # --------------------------------------------------------

    if unit in ["mcg/kg/min", "ug/kg/min"]:
        return (rate, "mcg/kg/min", "direct_mcg_per_kg_min", True, "")

    if unit in ["mcg/min", "ug/min"]:

        if pd.isna(weight) or weight <= 0:
            return (np.nan, "mcg/kg/min", "", False, "missing_weight_for_mcg_per_min")

        return (rate / weight, "mcg/kg/min", "mcg_per_min_divided_by_weight", True, "")

    if unit in ["mg/hour", "mg/hr"]:

        if pd.isna(weight) or weight <= 0:
            return (np.nan, "mcg/kg/min", "", False, "missing_weight_for_mg_per_hour")

        converted = rate * 1000 / 60 / weight

        return (converted, "mcg/kg/min", "mg_per_hour_to_mcg_per_kg_min", True, "")

    return (np.nan, "mcg/kg/min", "", False, f"unexpected_unit:{unit}")


conversion_results = vasoactive_events.apply(convert_rate, axis=1, result_type="expand")

conversion_results.columns = [
    "rate_standardized",
    "standard_unit",
    "conversion_method",
    "conversion_valid",
    "conversion_issue",
]

for column in conversion_results.columns:

    vasoactive_events[column] = conversion_results[column]

# ------------------------------------------------------------
# 13. Auditoria das unidades e conversões
# ------------------------------------------------------------

unit_qc = (
    vasoactive_events.groupby(
        [
            "drug",
            "rateuom",
            "_rateuom",
            "standard_unit",
            "conversion_method",
            "conversion_valid",
            "conversion_issue",
        ],
        dropna=False,
    )
    .agg(
        n_records=("subject_id", "size"),
        n_patients=("subject_id", "nunique"),
        raw_rate_median=("rate", "median"),
        standardized_rate_median=("rate_standardized", "median"),
        standardized_rate_min=("rate_standardized", "min"),
        standardized_rate_max=("rate_standardized", "max"),
    )
    .reset_index()
    .sort_values(
        ["drug", "conversion_valid", "n_records"], ascending=[True, False, False]
    )
)

print("\n" + "=" * 100)
print("AUDITORIA DAS UNIDADES")
print("=" * 100)

display(unit_qc)

# ------------------------------------------------------------
# 14. Faixas de plausibilidade
# ------------------------------------------------------------

plausibility_limits = {
    "norepinephrine": (0, 5),
    "epinephrine": (0, 5),
    "phenylephrine": (0, 20),
    "dopamine": (0, 100),
    "vasopressin": (0, 1),
    "angiotensin_ii": (0, 1),
    "dobutamine": (0, 50),
}

vasoactive_events["plausible_rate"] = False

for drug, limits in plausibility_limits.items():

    lower, upper = limits

    mask = (
        vasoactive_events["drug"].eq(drug)
        & vasoactive_events["conversion_valid"]
        & vasoactive_events["rate_standardized"].between(lower, upper, inclusive="both")
    )

    vasoactive_events.loc[mask, "plausible_rate"] = True

vasoactive_events["usable_rate"] = (
    vasoactive_events["conversion_valid"] & vasoactive_events["plausible_rate"]
)

usable_events = vasoactive_events[vasoactive_events["usable_rate"]].copy()

print("\nEventos utilizáveis:", len(usable_events))

print(
    "Eventos excluídos por unidade/conversão/plausibilidade:",
    len(vasoactive_events) - len(usable_events),
)

# ------------------------------------------------------------
# 15. Criar grade minuto a minuto
# ------------------------------------------------------------

minute_rows = []

for _, row in cohort.iterrows():

    minute_grid = pd.date_range(
        start=(row["mb_time_preliminary"] - pd.Timedelta(minutes=60)),
        end=row["mb_time_preliminary"],
        freq="1min",
    )

    for minute_time in minute_grid:

        minute_rows.append(
            {
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"],
                "stay_id": row["stay_id"],
                "mb_time_preliminary": (row["mb_time_preliminary"]),
                "minute_time": minute_time,
            }
        )

minute_grid = pd.DataFrame(minute_rows)

print("\nLinhas da grade minuto a minuto:", len(minute_grid))

# ------------------------------------------------------------
# 16. Expandir infusões ativas sobre a grade
# ------------------------------------------------------------

active_minute_rows = []

for _, patient_minute in minute_grid.iterrows():

    patient_events = usable_events[
        usable_events["stay_id"].eq(patient_minute["stay_id"])
        & (usable_events["starttime"] <= patient_minute["minute_time"])
        & (usable_events["endtime"] >= patient_minute["minute_time"])
    ]

    if patient_events.empty:
        continue

    # Somar infusões simultâneas do mesmo agente
    drug_rates = patient_events.groupby("drug", as_index=False).agg(
        rate_standardized=("rate_standardized", "sum")
    )

    for _, drug_row in drug_rates.iterrows():

        active_minute_rows.append(
            {
                "subject_id": (patient_minute["subject_id"]),
                "hadm_id": (patient_minute["hadm_id"]),
                "stay_id": (patient_minute["stay_id"]),
                "minute_time": (patient_minute["minute_time"]),
                "drug": drug_row["drug"],
                "rate_standardized": (drug_row["rate_standardized"]),
            }
        )

if active_minute_rows:

    active_minute_events = pd.DataFrame(active_minute_rows)

else:

    active_minute_events = pd.DataFrame(
        columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "minute_time",
            "drug",
            "rate_standardized",
        ]
    )

# ------------------------------------------------------------
# 17. Criar formato largo por minuto
# ------------------------------------------------------------

minute_drugs_wide = active_minute_events.pivot_table(
    index=["subject_id", "hadm_id", "stay_id", "minute_time"],
    columns="drug",
    values="rate_standardized",
    aggfunc="sum",
    fill_value=0,
).reset_index()

minute_drugs_wide.columns.name = None

drug_columns = [
    "norepinephrine",
    "epinephrine",
    "phenylephrine",
    "dopamine",
    "vasopressin",
    "angiotensin_ii",
    "dobutamine",
]

for column in drug_columns:

    if column not in minute_drugs_wide.columns:
        minute_drugs_wide[column] = 0.0

# ------------------------------------------------------------
# 18. Garantir minutos sem outro vasopressor
# ------------------------------------------------------------

minute_complete = minute_grid[
    ["subject_id", "hadm_id", "stay_id", "minute_time"]
].merge(
    minute_drugs_wide,
    on=["subject_id", "hadm_id", "stay_id", "minute_time"],
    how="left",
    validate="one_to_one",
)

for column in drug_columns:

    minute_complete[column] = minute_complete[column].fillna(0.0)

# ------------------------------------------------------------
# 19. Calcular NEE minuto a minuto
# ------------------------------------------------------------

minute_complete["nee"] = (
    minute_complete["norepinephrine"]
    + minute_complete["epinephrine"]
    + (minute_complete["phenylephrine"] / 10)
    + (minute_complete["dopamine"] / 100)
    + (minute_complete["vasopressin"] * 2.5)
    + (minute_complete["angiotensin_ii"] * 10)
)

# ------------------------------------------------------------
# 20. Resumo basal por paciente
# ------------------------------------------------------------

patient_summary = minute_complete.groupby(
    ["subject_id", "hadm_id", "stay_id"], as_index=False
).agg(
    nee_baseline_median=("nee", "median"),
    nee_baseline_mean=("nee", "mean"),
    nee_baseline_min=("nee", "min"),
    nee_baseline_max=("nee", "max"),
    norepinephrine_baseline_median=("norepinephrine", "median"),
    epinephrine_baseline_median=("epinephrine", "median"),
    vasopressin_baseline_median=("vasopressin", "median"),
    phenylephrine_baseline_median=("phenylephrine", "median"),
    dopamine_baseline_median=("dopamine", "median"),
    angiotensin_ii_baseline_median=("angiotensin_ii", "median"),
    dobutamine_baseline_median=("dobutamine", "median"),
    minutes_with_any_vasopressor=("nee", lambda values: int((values > 0).sum())),
)

# ------------------------------------------------------------
# 21. Flags de uso
# ------------------------------------------------------------

for drug in [
    "norepinephrine",
    "epinephrine",
    "vasopressin",
    "phenylephrine",
    "dopamine",
    "angiotensin_ii",
    "dobutamine",
]:

    dose_column = f"{drug}_baseline_median"

    use_column = f"{drug}_active_baseline"

    patient_summary[use_column] = patient_summary[dose_column] > 0

# ------------------------------------------------------------
# 22. Integrar à base da Tabela 1
# ------------------------------------------------------------

new_columns = [
    column
    for column in patient_summary.columns
    if column not in ["subject_id", "hadm_id", "stay_id"]
]

cohort_clean = cohort.drop(
    columns=[column for column in new_columns if column in cohort.columns],
    errors="ignore",
)

table1_vasoactive = cohort_clean.merge(
    patient_summary,
    on=["subject_id", "hadm_id", "stay_id"],
    how="left",
    validate="one_to_one",
)

# ------------------------------------------------------------
# 23. Disponibilidade e uso
# ------------------------------------------------------------

usage_rows = []

for drug in [
    "norepinephrine",
    "vasopressin",
    "epinephrine",
    "phenylephrine",
    "dopamine",
    "angiotensin_ii",
    "dobutamine",
]:

    use_column = f"{drug}_active_baseline"

    dose_column = f"{drug}_baseline_median"

    usage_rows.append(
        {
            "drug": drug,
            "n_active": int(table1_vasoactive[use_column].fillna(False).sum()),
            "n_total": len(table1_vasoactive),
            "percent_active": (
                100 * table1_vasoactive[use_column].fillna(False).mean()
            ),
            "dose_median_among_active": (
                table1_vasoactive.loc[
                    table1_vasoactive[use_column].fillna(False), dose_column
                ].median()
            ),
        }
    )

drug_usage_summary = pd.DataFrame(usage_rows)

# ------------------------------------------------------------
# 24. NEE por fenótipo
# ------------------------------------------------------------

nee_by_phenotype = table1_vasoactive.groupby(
    "hemodynamic_phenotype", as_index=False
).agg(
    n_patients=("subject_id", "nunique"),
    nee_median=("nee_baseline_median", "median"),
    nee_p25=("nee_baseline_median", lambda values: values.quantile(0.25)),
    nee_p75=("nee_baseline_median", lambda values: values.quantile(0.75)),
    norepinephrine_median=("norepinephrine_baseline_median", "median"),
    vasopressin_n=("vasopressin_active_baseline", "sum"),
    epinephrine_n=("epinephrine_active_baseline", "sum"),
    dobutamine_n=("dobutamine_active_baseline", "sum"),
)

# ------------------------------------------------------------
# 25. Mostrar resultados
# ------------------------------------------------------------

pd.set_option("display.max_rows", 200)

pd.set_option("display.max_columns", 60)

pd.set_option("display.width", 300)

print("\n" + "=" * 100)
print("USO DE DROGAS NA JANELA BASAL")
print("=" * 100)

display(drug_usage_summary)

print("\n" + "=" * 100)
print("NEE BASAL POR FENÓTIPO")
print("=" * 100)

display(nee_by_phenotype)

# ------------------------------------------------------------
# 26. Controle de qualidade
# ------------------------------------------------------------

qc_summary = pd.DataFrame(
    {
        "metric": [
            "Patients in Table 1 cohort",
            "Patients with reconstructed baseline minute grid",
            "Patients with NEE > 0",
            "Patients with norepinephrine > 0",
            "Patients with complete 61-minute grid",
            "Events with valid standardized rate",
            "Events excluded from dose reconstruction",
        ],
        "value": [
            len(table1_vasoactive),
            patient_summary["subject_id"].nunique(),
            int((table1_vasoactive["nee_baseline_median"] > 0).sum()),
            int((table1_vasoactive["norepinephrine_baseline_median"] > 0).sum()),
            int((table1_vasoactive["minutes_with_any_vasopressor"] <= 61).sum()),
            len(usable_events),
            len(vasoactive_events) - len(usable_events),
        ],
    }
)

print("\n" + "=" * 100)
print("CONTROLE DE QUALIDADE")
print("=" * 100)

display(qc_summary)

# ------------------------------------------------------------
# 27. Salvar outputs
# ------------------------------------------------------------

timestamp = run_stamp()

table1_path = COHORT_DIR / f"TABLE1_BASE_with_baseline_NEE_{timestamp}.csv.gz"

events_path = (
    COHORT_DIR / f"methylene_blue_baseline_vasoactive_events_{timestamp}.csv.gz"
)

minute_path = COHORT_DIR / f"methylene_blue_baseline_NEE_minute_grid_{timestamp}.csv.gz"

patient_summary_path = (
    COHORT_DIR / f"methylene_blue_baseline_NEE_patient_summary_{timestamp}.csv.gz"
)

unit_qc_path = QC_DIR / f"methylene_blue_baseline_vasoactive_unit_qc_{timestamp}.csv"

drug_usage_path = TABLE_DIR / f"TABLE1_baseline_drug_usage_{timestamp}.csv"

nee_phenotype_path = TABLE_DIR / f"TABLE1_baseline_NEE_by_phenotype_{timestamp}.csv"

qc_path = QC_DIR / f"methylene_blue_code11b_NEE_qc_{timestamp}.csv"

itemids_path = METADATA_DIR / f"methylene_blue_NEE_itemids_{timestamp}.csv"

report_path = QC_DIR / f"methylene_blue_code11b_NEE_report_{timestamp}.txt"

table1_vasoactive.to_csv(table1_path, index=False, compression="gzip")

vasoactive_events.to_csv(events_path, index=False, compression="gzip")

minute_complete.to_csv(minute_path, index=False, compression="gzip")

patient_summary.to_csv(patient_summary_path, index=False, compression="gzip")

unit_qc.to_csv(unit_qc_path, index=False)

drug_usage_summary.to_csv(drug_usage_path, index=False)

nee_by_phenotype.to_csv(nee_phenotype_path, index=False)

qc_summary.to_csv(qc_path, index=False)

validated_items.to_csv(itemids_path, index=False)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("METHYLENE BLUE — BASELINE NOREPINEPHRINE-EQUIVALENT DOSE\n")

    file.write("=" * 80 + "\n\n")

    file.write(
        "Baseline window: 60 minutes immediately preceding "
        "the preliminary methylene blue time.\n\n"
    )

    file.write(
        "NEE = norepinephrine + epinephrine "
        "+ phenylephrine/10 + dopamine/100 "
        "+ vasopressin*2.5 + angiotensin II*10.\n\n"
    )

    file.write(
        "Catecholamines and phenylephrine were expressed in "
        "mcg/kg/min, vasopressin in U/min, and angiotensin II "
        "in mcg/kg/min.\n\n"
    )

    file.write(
        "Baseline NEE was defined as the median of the "
        "minute-level NEE values during the 60-minute window.\n\n"
    )

    file.write(f"Input cohort: {TABLE1_LABS_PATH}\n")

    file.write(f"Patients: {len(table1_vasoactive)}\n")

    file.write(f"Updated Table 1 base: {table1_path}\n")

print("\n" + "=" * 100)
print("OUTPUTS SALVOS")
print("=" * 100)

print(table1_path)
print(events_path)
print(minute_path)
print(patient_summary_path)
print(unit_qc_path)
print(drug_usage_path)
print(nee_phenotype_path)
print(qc_path)
print(report_path)

gc.collect()

print("\nCÓDIGO 11B CONCLUÍDO COM SUCESSO.")
