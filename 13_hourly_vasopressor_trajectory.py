"""Source: original notebook cell 54. Run through run.py."""

from common import artifact, display, mimic_source, phase_dir, run_stamp, source_file

# ============================================================
# CÓDIGO 13A — TRAJETÓRIA HORÁRIA DA NEE
#
# Janela:
#   -60 min até +72 h após o azul de metileno
#
# Camada técnica:
#   Reconstrução minuto a minuto dentro de cada paciente
#
# Output analítico:
#   Mediana da NEE em cada intervalo horário:
#       Baseline: -60 a 0 min
#       Hour 0:     0 a <1 h
#       Hour 1:     1 a <2 h
#       ...
#       Hour 71:   71 a <72 h
#
# Seguimento:
#   Até 72 horas ou alta da UTI, o que ocorrer primeiro
#
# Fórmula:
#   NEE =
#       norepinephrine
#       + epinephrine
#       + phenylephrine / 10
#       + dopamine / 100
#       + 2.5 × vasopressin
#       + 10 × angiotensin II
#
# Dobutamina:
#   Reconstruída separadamente; não integra a NEE
# ============================================================

import gc

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 0. Instalar python-docx, se necessário
# ------------------------------------------------------------

from docx import Document
from docx.shared import Pt

# ------------------------------------------------------------

# ------------------------------------------------------------

ROOT = phase_dir("mimic_iv")

SOURCE_DIR = mimic_source()

PROJECT_DIR = phase_dir("mimic_iv")

COHORT_DIR = PROJECT_DIR / "cohorts"
TABLE_DIR = PROJECT_DIR / "tables"
QC_DIR = PROJECT_DIR / "qc"
DOCUMENT_DIR = PROJECT_DIR / "documents"

for directory in [COHORT_DIR, TABLE_DIR, QC_DIR, DOCUMENT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"MIMIC-IV source não encontrado em: {SOURCE_DIR}")

print("=" * 100)
print("CÓDIGO 13A — TRAJETÓRIA HORÁRIA DA NEE")
print("=" * 100)

print("\nDiretório do projeto:")
print(ROOT)

print("\nMIMIC-IV source, somente leitura:")
print(SOURCE_DIR)

print("\nOutputs:")
print(PROJECT_DIR)

# ------------------------------------------------------------
# 2. Funções auxiliares
# ------------------------------------------------------------


def latest_project_file(patterns):
    return artifact(PROJECT_DIR, patterns)


def locate_source_file(filename):
    return source_file(SOURCE_DIR, filename)


def normalize_unit(unit):

    if pd.isna(unit):
        return ""

    return (
        str(unit)
        .strip()
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("micrograms", "mcg")
        .replace("microgram", "mcg")
        .replace("units", "unit")
        .replace(" ", "")
    )


# ------------------------------------------------------------
# 3. Localizar coorte congelada e inputevents
# ------------------------------------------------------------

COHORT_PATH = artifact(
    PROJECT_DIR, "METHYLENE_BLUE_ANALYTIC_COHORT_FROZEN_FINAL_*.csv.gz"
)

if COHORT_PATH is None:
    raise FileNotFoundError("Não encontrei a coorte analítica congelada.")

INPUTEVENTS_PATH = locate_source_file("inputevents.csv.gz")

print("\nCoorte selecionada:")
print(COHORT_PATH)

print("\nInputevents:")
print(INPUTEVENTS_PATH)

# ------------------------------------------------------------
# 4. Carregar e preparar a coorte
# ------------------------------------------------------------

cohort = pd.read_csv(COHORT_PATH, compression="gzip", low_memory=False)

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

# Identificar a coluna da alta da UTI
ICU_OUTTIME_COLUMN = None

for candidate in ["outtime", "icu_outtime", "outtime_icu"]:
    if candidate in cohort.columns:
        ICU_OUTTIME_COLUMN = candidate
        break

if ICU_OUTTIME_COLUMN is not None:

    cohort["icu_outtime_analysis"] = pd.to_datetime(
        cohort[ICU_OUTTIME_COLUMN], errors="coerce"
    )

    print("\nColuna de alta da UTI:")
    print(ICU_OUTTIME_COLUMN)

else:

    cohort["icu_outtime_analysis"] = pd.NaT

    print(
        "\nATENÇÃO: horário de alta da UTI não encontrado. "
        "Todos serão administrativamente encerrados em 72 h."
    )

cohort = (
    cohort.dropna(subset=["subject_id", "hadm_id", "stay_id", "mb_time_preliminary"])
    .sort_values(["subject_id", "mb_time_preliminary"])
    .drop_duplicates(subset=["subject_id"], keep="first")
    .reset_index(drop=True)
)

cohort["analysis_start"] = cohort["mb_time_preliminary"] - pd.Timedelta(hours=1)

cohort["administrative_end"] = cohort["mb_time_preliminary"] + pd.Timedelta(hours=72)

cohort["analysis_end"] = cohort["administrative_end"]

valid_outtime = cohort["icu_outtime_analysis"].notna() & (
    cohort["icu_outtime_analysis"] > cohort["mb_time_preliminary"]
)

cohort.loc[valid_outtime, "analysis_end"] = cohort.loc[
    valid_outtime, ["administrative_end", "icu_outtime_analysis"]
].min(axis=1)

cohort["followup_hours"] = (
    cohort["analysis_end"] - cohort["mb_time_preliminary"]
).dt.total_seconds() / 3600

print("\nPacientes:", len(cohort))

display(cohort["followup_hours"].describe(percentiles=[0.25, 0.50, 0.75]))

# ------------------------------------------------------------
# 5. Itemids dos vasoativos
# ------------------------------------------------------------

DRUG_ITEM_MAP = {
    221906: "norepinephrine",
    221289: "epinephrine",
    229617: "epinephrine",
    222315: "vasopressin",
    221749: "phenylephrine",
    229630: "phenylephrine",
    229631: "phenylephrine",
    229632: "phenylephrine",
    221662: "dopamine",
    229709: "angiotensin_ii",
    229764: "angiotensin_ii",
    221653: "dobutamine",
}

TARGET_ITEMIDS = set(DRUG_ITEM_MAP.keys())

target_stay_ids = set(cohort["stay_id"].dropna().astype(int))

time_table = cohort[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "mb_time_preliminary",
        "analysis_start",
        "analysis_end",
    ]
].copy()

# ------------------------------------------------------------
# 6. Inspecionar inputevents
# ------------------------------------------------------------

header = pd.read_csv(INPUTEVENTS_PATH, compression="gzip", nrows=0)

available_columns = header.columns.tolist()

preferred_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "storetime",
    "itemid",
    "rate",
    "rateuom",
    "patientweight",
    "statusdescription",
    "orderid",
    "linkorderid",
]

usecols = [column for column in preferred_columns if column in available_columns]

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
    column for column in required_input_columns if column not in usecols
]

if missing_input_columns:
    raise RuntimeError(
        "Colunas obrigatórias ausentes em inputevents: " f"{missing_input_columns}"
    )

print("\nColunas lidas:")
print(usecols)

# ------------------------------------------------------------
# 7. Ler inputevents em chunks
# ------------------------------------------------------------

chunk_size = 300_000

matched_chunks = []

total_rows = 0
target_rows = 0
overlap_rows = 0

reader = pd.read_csv(
    INPUTEVENTS_PATH,
    compression="gzip",
    usecols=usecols,
    chunksize=chunk_size,
    low_memory=False,
)

print("\nIniciando leitura em chunks...")

for chunk_number, chunk in enumerate(reader, start=1):

    total_rows += len(chunk)

    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")

    chunk = chunk[chunk["itemid"].isin(TARGET_ITEMIDS)].copy()

    if chunk.empty:

        if chunk_number % 10 == 0:
            print(
                f"Chunk {chunk_number:03d} | "
                f"linhas lidas: {total_rows:,} | "
                f"eventos acumulados: {overlap_rows:,}"
            )

        gc.collect()
        continue

    chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")

    chunk = chunk[chunk["stay_id"].isin(target_stay_ids)].copy()

    target_rows += len(chunk)

    if chunk.empty:
        gc.collect()
        continue

    for column in ["subject_id", "hadm_id"]:
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

    overlap_mask = (
        chunk["starttime"].notna()
        & chunk["endtime"].notna()
        & (chunk["starttime"] < chunk["analysis_end"])
        & (chunk["endtime"] > chunk["analysis_start"])
    )

    chunk = chunk[overlap_mask].copy()

    if not chunk.empty:

        chunk["drug"] = chunk["itemid"].astype(int).map(DRUG_ITEM_MAP)

        matched_chunks.append(chunk)

        overlap_rows += len(chunk)

    if chunk_number % 5 == 0:
        print(
            f"Chunk {chunk_number:03d} | "
            f"linhas lidas: {total_rows:,} | "
            f"eventos sobrepostos: {overlap_rows:,}"
        )

    gc.collect()

if not matched_chunks:
    raise RuntimeError("Nenhum evento vasoativo foi encontrado.")

vasoactive_events = pd.concat(matched_chunks, ignore_index=True)

del matched_chunks
gc.collect()

print("\nLeitura concluída.")
print("Linhas examinadas:", f"{total_rows:,}")
print("Eventos das estadias-alvo:", f"{target_rows:,}")
print("Eventos sobrepostos:", f"{len(vasoactive_events):,}")

# ------------------------------------------------------------
# 8. Conversão e padronização das doses
# ------------------------------------------------------------

vasoactive_events["rateuom_normalized"] = vasoactive_events["rateuom"].apply(
    normalize_unit
)


def convert_rate(row):

    drug = row["drug"]
    rate = row["rate"]
    unit = row["rateuom_normalized"]
    weight = row["patientweight"]

    if pd.isna(rate):
        return (np.nan, "", False, "missing_rate")

    if rate <= 0:
        return (np.nan, "", False, "nonpositive_rate")

    if drug == "vasopressin":

        if unit in ["unit/min", "u/min"]:
            return (rate, "U/min", True, "direct")

        if unit in ["unit/hour", "unit/hr", "u/hour", "u/hr"]:
            return (rate / 60, "U/min", True, "units_hour_to_min")

        return (np.nan, "U/min", False, f"unexpected_unit:{unit}")

    if drug == "angiotensin_ii":

        if unit == "ng/kg/min":
            return (rate / 1000, "mcg/kg/min", True, "ng_to_mcg")

        if unit in ["mcg/kg/min", "ug/kg/min"]:
            return (rate, "mcg/kg/min", True, "direct")

        return (np.nan, "mcg/kg/min", False, f"unexpected_unit:{unit}")

    if unit in ["mcg/kg/min", "ug/kg/min"]:
        return (rate, "mcg/kg/min", True, "direct")

    if unit in ["mcg/min", "ug/min"]:

        if pd.isna(weight) or weight <= 0:
            return (np.nan, "mcg/kg/min", False, "missing_weight")

        return (rate / weight, "mcg/kg/min", True, "divided_by_weight")

    if unit in ["mg/hour", "mg/hr"]:

        if pd.isna(weight) or weight <= 0:
            return (np.nan, "mcg/kg/min", False, "missing_weight")

        return (rate * 1000 / 60 / weight, "mcg/kg/min", True, "mg_hour_conversion")

    return (np.nan, "mcg/kg/min", False, f"unexpected_unit:{unit}")


converted = vasoactive_events.apply(convert_rate, axis=1, result_type="expand")

converted.columns = [
    "rate_standardized",
    "standard_unit",
    "conversion_valid",
    "conversion_method",
]

for column in converted.columns:
    vasoactive_events[column] = converted[column]

# ------------------------------------------------------------
# 9. Limites de plausibilidade
# ------------------------------------------------------------

PLAUSIBILITY_LIMITS = {
    "norepinephrine": (0, 5),
    "epinephrine": (0, 5),
    "phenylephrine": (0, 20),
    "dopamine": (0, 100),
    "vasopressin": (0, 1),
    "angiotensin_ii": (0, 1),
    "dobutamine": (0, 50),
}

vasoactive_events["plausible_rate"] = False

for drug, limits in PLAUSIBILITY_LIMITS.items():

    lower, upper = limits

    valid_mask = (
        vasoactive_events["drug"].eq(drug)
        & vasoactive_events["conversion_valid"]
        & vasoactive_events["rate_standardized"].between(lower, upper, inclusive="both")
    )

    vasoactive_events.loc[valid_mask, "plausible_rate"] = True

vasoactive_events["usable_rate"] = (
    vasoactive_events["conversion_valid"] & vasoactive_events["plausible_rate"]
)

usable_events = vasoactive_events[vasoactive_events["usable_rate"]].copy()

print("\nEventos utilizáveis:", len(usable_events))
print("Eventos excluídos:", len(vasoactive_events) - len(usable_events))

# ------------------------------------------------------------
# 10. QC das unidades
# ------------------------------------------------------------

unit_qc = vasoactive_events.groupby(
    [
        "drug",
        "rateuom",
        "rateuom_normalized",
        "standard_unit",
        "conversion_valid",
        "conversion_method",
    ],
    dropna=False,
    as_index=False,
).agg(
    n_records=("subject_id", "size"),
    n_patients=("subject_id", "nunique"),
    raw_rate_median=("rate", "median"),
    standardized_rate_median=("rate_standardized", "median"),
    standardized_rate_min=("rate_standardized", "min"),
    standardized_rate_max=("rate_standardized", "max"),
)

print("\n" + "=" * 100)
print("AUDITORIA DAS UNIDADES")
print("=" * 100)

display(unit_qc)

# ------------------------------------------------------------
# 11. Reconstrução minuto a minuto e agregação horária
# ------------------------------------------------------------

DRUG_COLUMNS = [
    "norepinephrine",
    "epinephrine",
    "vasopressin",
    "phenylephrine",
    "dopamine",
    "angiotensin_ii",
    "dobutamine",
]

hourly_patient_frames = []

print("\nReconstruindo e agregando por hora...")

for patient_index, patient in cohort.iterrows():

    mb_time = patient["mb_time_preliminary"]
    analysis_start = patient["analysis_start"]
    analysis_end = patient["analysis_end"]
    stay_id = patient["stay_id"]

    # End exclusive para não criar artificialmente a hora 72
    minute_times = pd.date_range(
        start=analysis_start, end=(analysis_end - pd.Timedelta(minutes=1)), freq="1min"
    )

    minute_grid = pd.DataFrame({"time": minute_times})

    for drug in DRUG_COLUMNS:
        minute_grid[drug] = 0.0

    patient_events = usable_events[usable_events["stay_id"].eq(stay_id)]

    for _, event in patient_events.iterrows():

        event_start = max(event["starttime"], analysis_start)

        event_end = min(event["endtime"], analysis_end)

        if event_end <= event_start:
            continue

        active_mask = minute_grid["time"].ge(event_start) & minute_grid["time"].lt(
            event_end
        )

        drug = event["drug"]
        rate = event["rate_standardized"]

        # Máxima dose ativa para impedir dupla contagem
        minute_grid.loc[active_mask, drug] = np.maximum(
            minute_grid.loc[active_mask, drug].to_numpy(), rate
        )

    minute_grid["minutes_from_mb"] = (
        minute_grid["time"] - mb_time
    ).dt.total_seconds() / 60

    minute_grid["nee"] = (
        minute_grid["norepinephrine"]
        + minute_grid["epinephrine"]
        + minute_grid["phenylephrine"] / 10
        + minute_grid["dopamine"] / 100
        + minute_grid["vasopressin"] * 2.5
        + minute_grid["angiotensin_ii"] * 10
    )

    # -1 = baseline; 0 = 0–<1 h; 1 = 1–<2 h...
    minute_grid["hour_index"] = np.floor(minute_grid["minutes_from_mb"] / 60).astype(
        int
    )

    hourly = minute_grid.groupby("hour_index", as_index=False).agg(
        nee_median=("nee", "median"),
        nee_mean=("nee", "mean"),
        nee_min=("nee", "min"),
        nee_max=("nee", "max"),
        norepinephrine_median=("norepinephrine", "median"),
        epinephrine_median=("epinephrine", "median"),
        vasopressin_median=("vasopressin", "median"),
        phenylephrine_median=("phenylephrine", "median"),
        dopamine_median=("dopamine", "median"),
        angiotensin_ii_median=("angiotensin_ii", "median"),
        dobutamine_median=("dobutamine", "median"),
        minutes_in_interval=("time", "size"),
        minutes_with_vasopressor=("nee", lambda values: int((values > 0).sum())),
    )

    hourly["subject_id"] = patient["subject_id"]
    hourly["hadm_id"] = patient["hadm_id"]
    hourly["stay_id"] = patient["stay_id"]
    hourly["hemodynamic_phenotype"] = patient["hemodynamic_phenotype"]
    hourly["mb_time"] = mb_time
    hourly["followup_hours_available"] = patient["followup_hours"]

    hourly["interval_label"] = np.where(
        hourly["hour_index"].eq(-1),
        "Baseline (-1 to 0 h)",
        hourly["hour_index"].apply(lambda value: f"{value} to <{value + 1} h"),
    )

    hourly["any_vasopressor_in_hour"] = hourly["minutes_with_vasopressor"] > 0

    hourly["vasopressor_entire_hour"] = (
        hourly["minutes_with_vasopressor"] == hourly["minutes_in_interval"]
    )

    hourly["nee_zero_entire_hour"] = hourly["minutes_with_vasopressor"] == 0

    hourly_patient_frames.append(hourly)

    if (patient_index + 1) % 10 == 0:
        print(f"Pacientes concluídos: " f"{patient_index + 1}/{len(cohort)}")

hourly_nee = pd.concat(hourly_patient_frames, ignore_index=True)

del hourly_patient_frames
gc.collect()

ordered_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "hemodynamic_phenotype",
    "mb_time",
    "hour_index",
    "interval_label",
    "followup_hours_available",
    "minutes_in_interval",
    "minutes_with_vasopressor",
    "any_vasopressor_in_hour",
    "vasopressor_entire_hour",
    "nee_zero_entire_hour",
    "nee_median",
    "nee_mean",
    "nee_min",
    "nee_max",
    "norepinephrine_median",
    "epinephrine_median",
    "vasopressin_median",
    "phenylephrine_median",
    "dopamine_median",
    "angiotensin_ii_median",
    "dobutamine_median",
]

hourly_nee = (
    hourly_nee[ordered_columns]
    .sort_values(["subject_id", "hour_index"])
    .reset_index(drop=True)
)

print("\nBase horária concluída.")
print("Linhas:", f"{len(hourly_nee):,}")
print("Pacientes:", hourly_nee["subject_id"].nunique())

# ------------------------------------------------------------
# 12. Comparação com a NEE basal congelada
# ------------------------------------------------------------

baseline_hourly = hourly_nee[hourly_nee["hour_index"].eq(-1)][
    ["subject_id", "nee_median", "minutes_in_interval"]
].rename(
    columns={
        "nee_median": "nee_baseline_reconstructed",
        "minutes_in_interval": "baseline_minutes",
    }
)

if "nee_baseline_median" in cohort.columns:

    baseline_comparison = cohort[["subject_id", "nee_baseline_median"]].merge(
        baseline_hourly, on="subject_id", how="left", validate="one_to_one"
    )

    baseline_comparison["absolute_difference"] = (
        baseline_comparison["nee_baseline_reconstructed"]
        - baseline_comparison["nee_baseline_median"]
    ).abs()

    baseline_comparison_summary = pd.DataFrame(
        {
            "metric": [
                "Patients compared",
                "Previous baseline NEE, median",
                "Reconstructed baseline NEE, median",
                "Median absolute difference",
                "Maximum absolute difference",
                "Patients with absolute difference >0.01",
            ],
            "value": [
                len(baseline_comparison),
                baseline_comparison["nee_baseline_median"].median(),
                baseline_comparison["nee_baseline_reconstructed"].median(),
                baseline_comparison["absolute_difference"].median(),
                baseline_comparison["absolute_difference"].max(),
                int((baseline_comparison["absolute_difference"] > 0.01).sum()),
            ],
        }
    )

else:

    baseline_comparison = baseline_hourly.copy()

    baseline_comparison_summary = pd.DataFrame(
        {
            "metric": ["Patients with reconstructed baseline NEE"],
            "value": [baseline_hourly["subject_id"].nunique()],
        }
    )

print("\n" + "=" * 100)
print("COMPARAÇÃO COM A NEE BASAL CONGELADA")
print("=" * 100)

display(baseline_comparison_summary)

# ------------------------------------------------------------
# 13. Completude do seguimento
# ------------------------------------------------------------

followup_rows = []

for threshold in [6, 12, 24, 48, 72]:

    n_available = int((cohort["followup_hours"] >= threshold).sum())

    followup_rows.append(
        {
            "followup_hours": threshold,
            "n_available": n_available,
            "n_total": len(cohort),
            "percent_available": (100 * n_available / len(cohort)),
        }
    )

followup_completeness = pd.DataFrame(followup_rows)

print("\n" + "=" * 100)
print("COMPLETUDE DO SEGUIMENTO")
print("=" * 100)

display(followup_completeness)

# ------------------------------------------------------------
# 14. Resumo por fenótipo e horário
# ------------------------------------------------------------

hourly_phenotype_summary = hourly_nee.groupby(
    ["hemodynamic_phenotype", "hour_index"], as_index=False
).agg(
    n_patients=("subject_id", "nunique"),
    nee_median=("nee_median", "median"),
    nee_p25=("nee_median", lambda values: values.quantile(0.25)),
    nee_p75=("nee_median", lambda values: values.quantile(0.75)),
    patients_zero_entire_hour=("nee_zero_entire_hour", "sum"),
)

# ------------------------------------------------------------
# 15. QC final
# ------------------------------------------------------------

qc_summary = pd.DataFrame(
    {
        "metric": [
            "Patients in frozen cohort",
            "Patients in hourly NEE database",
            "Hourly longitudinal rows",
            "Usable vasoactive records",
            "Excluded vasoactive records",
            "Duplicate patient-hour rows",
            "Rows with negative NEE",
            "Patients with baseline hour",
            "Patients with at least 6 h follow-up",
            "Patients with at least 24 h follow-up",
            "Patients with at least 48 h follow-up",
            "Patients with complete 72 h follow-up",
        ],
        "value": [
            len(cohort),
            hourly_nee["subject_id"].nunique(),
            len(hourly_nee),
            len(usable_events),
            (len(vasoactive_events) - len(usable_events)),
            int(hourly_nee.duplicated(subset=["subject_id", "hour_index"]).sum()),
            int((hourly_nee["nee_median"] < 0).sum()),
            baseline_hourly["subject_id"].nunique(),
            int((cohort["followup_hours"] >= 6).sum()),
            int((cohort["followup_hours"] >= 24).sum()),
            int((cohort["followup_hours"] >= 48).sum()),
            int((cohort["followup_hours"] >= 72).sum()),
        ],
    }
)

print("\n" + "=" * 100)
print("CONTROLE DE QUALIDADE")
print("=" * 100)

display(qc_summary)

# ------------------------------------------------------------
# 16. Textos automáticos para Methods e Results
# ------------------------------------------------------------

methods_text = (
    "Vasopressor exposure was reconstructed from structured "
    "medication administration records from 60 minutes before "
    "to 72 hours after methylene blue administration. Active "
    "infusion intervals were initially mapped to a minute-level "
    "time grid and subsequently summarized in consecutive hourly "
    "intervals using the median dose recorded during each hour. "
    "The baseline interval comprised the 60 minutes preceding "
    "methylene blue administration. Norepinephrine, epinephrine, "
    "phenylephrine, and dopamine doses were standardized to "
    "µg/kg/min, vasopressin to U/min, and angiotensin II to "
    "µg/kg/min. Norepinephrine-equivalent dose was calculated as "
    "norepinephrine + epinephrine + phenylephrine/10 + "
    "dopamine/100 + 2.5 × vasopressin + 10 × angiotensin II. "
    "Dobutamine exposure was reconstructed separately and was not "
    "included in the norepinephrine-equivalent dose. Follow-up "
    "was censored at 72 hours or ICU discharge, whichever occurred "
    "first."
)

n_total = len(cohort)

followup_median = cohort["followup_hours"].median()

followup_q1 = cohort["followup_hours"].quantile(0.25)

followup_q3 = cohort["followup_hours"].quantile(0.75)

n_6h = int((cohort["followup_hours"] >= 6).sum())

n_24h = int((cohort["followup_hours"] >= 24).sum())

n_48h = int((cohort["followup_hours"] >= 48).sum())

n_72h = int((cohort["followup_hours"] >= 72).sum())

results_text = (
    f"Hourly vasopressor exposure was successfully reconstructed "
    f"for all {n_total} patients, generating "
    f"{len(hourly_nee):,} patient-hour observations. Median "
    f"available follow-up after methylene blue administration was "
    f"{followup_median:.1f} hours "
    f"[IQR {followup_q1:.1f}–{followup_q3:.1f}]. At least "
    f"6 hours of observation were available for "
    f"{n_6h}/{n_total} ({100*n_6h/n_total:.1f}%) patients, "
    f"24 hours for {n_24h}/{n_total} "
    f"({100*n_24h/n_total:.1f}%), 48 hours for "
    f"{n_48h}/{n_total} ({100*n_48h/n_total:.1f}%), and the "
    f"complete 72-hour period for {n_72h}/{n_total} "
    f"({100*n_72h/n_total:.1f}%)."
)

methods_document = Document()
methods_document.add_heading(
    "Longitudinal vasopressor analysis — Methods draft", level=1
)

paragraph = methods_document.add_paragraph(methods_text)

for run in paragraph.runs:
    run.font.name = "Arial"
    run.font.size = Pt(10)

results_document = Document()
results_document.add_heading(
    "Longitudinal vasopressor analysis — Results draft", level=1
)

paragraph = results_document.add_paragraph(results_text)

for run in paragraph.runs:
    run.font.name = "Arial"
    run.font.size = Pt(10)

# ------------------------------------------------------------
# 17. Salvar outputs
# ------------------------------------------------------------

timestamp = run_stamp()

hourly_path = COHORT_DIR / f"methylene_blue_hourly_NEE_{timestamp}.csv.gz"

events_path = (
    COHORT_DIR / f"methylene_blue_longitudinal_vasoactive_events_{timestamp}.csv.gz"
)

phenotype_summary_path = (
    TABLE_DIR / f"methylene_blue_hourly_NEE_by_phenotype_{timestamp}.csv"
)

followup_path = (
    TABLE_DIR / f"methylene_blue_hourly_followup_completeness_{timestamp}.csv"
)

unit_qc_path = QC_DIR / f"methylene_blue_hourly_NEE_unit_qc_{timestamp}.csv"

qc_path = QC_DIR / f"methylene_blue_code13A_hourly_NEE_qc_{timestamp}.csv"

baseline_comparison_path = (
    QC_DIR / f"methylene_blue_hourly_baseline_comparison_{timestamp}.csv"
)

methods_docx_path = DOCUMENT_DIR / f"Methods_draft_hourly_NEE_{timestamp}.docx"

methods_txt_path = DOCUMENT_DIR / f"Methods_draft_hourly_NEE_{timestamp}.txt"

results_docx_path = DOCUMENT_DIR / f"Results_draft_hourly_NEE_{timestamp}.docx"

results_txt_path = DOCUMENT_DIR / f"Results_draft_hourly_NEE_{timestamp}.txt"

hourly_nee.to_csv(hourly_path, index=False, compression="gzip")

usable_events.to_csv(events_path, index=False, compression="gzip")

hourly_phenotype_summary.to_csv(phenotype_summary_path, index=False)

followup_completeness.to_csv(followup_path, index=False)

unit_qc.to_csv(unit_qc_path, index=False)

qc_summary.to_csv(qc_path, index=False)

baseline_comparison.to_csv(baseline_comparison_path, index=False)

methods_document.save(methods_docx_path)

results_document.save(results_docx_path)

with open(methods_txt_path, "w", encoding="utf-8") as file:
    file.write(methods_text)

with open(results_txt_path, "w", encoding="utf-8") as file:
    file.write(results_text)

print("\n" + "=" * 100)
print("ARQUIVOS GERADOS")
print("=" * 100)

print("\nBase horária da NEE:")
print(hourly_path)

print("\nResumo por fenótipo:")
print(phenotype_summary_path)

print("\nQC:")
print(qc_path)

print("\nMethods draft:")
print(methods_docx_path)

print("\nResults draft:")
print(results_docx_path)

print("\n" + "=" * 100)
print("TEXTO PRELIMINAR DE RESULTS")
print("=" * 100)

print(results_text)

gc.collect()

print("\nCÓDIGO 13A CONCLUÍDO COM SUCESSO.")
