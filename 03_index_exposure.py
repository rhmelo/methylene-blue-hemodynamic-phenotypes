"""Source: original notebook cell 25. Run through run.py."""

from common import artifact, display, mimic_source, phase_dir, run_stamp

# ============================================================
# CÓDIGO 6 — LIMPEZA DA EXPOSIÇÃO IV E VINCULAÇÃO À UTI
# VERSÃO CORRIGIDA
#
# Objetivos:
# 1. Carregar a coorte IV preliminar
# 2. Inspecionar unidades diferentes de mg
# 3. Converter mL em mg quando a concentração for identificável
# 4. Marcar unidades inválidas ou ambíguas
# 5. Vincular prescrições às estadias de UTI
# 6. Criar uma exposição índice preliminar por paciente
#
# Fonte:

#
# Outputs:
#   HIRID/HiRID_Methylene_Blue_Hemodynamic_Phenotypes/
#   MIMIC_Phase
#
# IMPORTANTE:
# prescriptions.starttime representa o início registrado da
# prescrição/ordem. A administração será validada posteriormente
# com eMAR.
# ============================================================

import pandas as pd
import numpy as np
import re
import gc

# ------------------------------------------------------------
# 1. Caminhos
# ------------------------------------------------------------

SOURCE_DIR = mimic_source()

PROJECT_DIR = phase_dir("mimic_iv")

COHORT_DIR = PROJECT_DIR / "cohorts"
QC_DIR = PROJECT_DIR / "qc"

COHORT_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Pasta-fonte não encontrada: {SOURCE_DIR}")

print("=" * 90)
print("CÓDIGO 6 — LIMPEZA DA EXPOSIÇÃO IV E VINCULAÇÃO À UTI")
print("=" * 90)

print("\nFonte:")
print(SOURCE_DIR)

print("\nPasta de outputs:")
print(PROJECT_DIR)

# ------------------------------------------------------------
# 2. Localizar a coorte IV preliminar mais recente
# ------------------------------------------------------------

IV_PATH = artifact(COHORT_DIR, "methylene_blue_preliminary_iv_candidates_*.csv.gz")

print("\nArquivo IV preliminar selecionado:")
print(IV_PATH)

# ------------------------------------------------------------
# 3. Localizar icustays.csv.gz
# ------------------------------------------------------------

direct_icustays_path = SOURCE_DIR / "icustays.csv.gz"

if direct_icustays_path.exists():
    ICUSTAYS_PATH = direct_icustays_path

else:
    icustays_candidates = list(SOURCE_DIR.rglob("icustays.csv.gz"))

    if len(icustays_candidates) == 1:
        ICUSTAYS_PATH = icustays_candidates[0]

    elif len(icustays_candidates) == 0:
        raise FileNotFoundError(
            "icustays.csv.gz não foi encontrado dentro de MIMIC-IV source."
        )

    else:
        print("\nForam encontrados vários arquivos icustays.csv.gz:")

        for i, path in enumerate(icustays_candidates, start=1):
            print(f"{i}. {path}")

        raise RuntimeError(
            "Há mais de um icustays.csv.gz. " "Defina manualmente ICUSTAYS_PATH."
        )

print("\nArquivo de estadias de UTI:")
print(ICUSTAYS_PATH)

# ------------------------------------------------------------
# 4. Carregar a pequena coorte IV
# ------------------------------------------------------------

iv = pd.read_csv(IV_PATH, compression="gzip", low_memory=False)

if iv.empty:
    raise RuntimeError("A coorte intravenosa preliminar está vazia.")

required_identifier_columns = ["subject_id", "hadm_id"]

missing_identifiers = [
    col for col in required_identifier_columns if col not in iv.columns
]

if missing_identifiers:
    raise RuntimeError(
        "Colunas identificadoras ausentes na coorte IV: " f"{missing_identifiers}"
    )

print("\nRegistros IV carregados:", f"{len(iv):,}")
print("Pacientes IV:", f"{iv['subject_id'].nunique():,}")
print("Internações IV:", f"{iv['hadm_id'].nunique():,}")

print("\nColunas disponíveis:")
print(iv.columns.tolist())

# ------------------------------------------------------------
# 5. Função para encontrar a primeira coluna existente
# ------------------------------------------------------------


def first_existing(columns, candidates):
    """
    Retorna a primeira coluna presente na lista candidates.
    """
    for column in candidates:
        if column in columns:
            return column

    return None


# ------------------------------------------------------------
# 6. Identificar colunas relevantes
# ------------------------------------------------------------

route_col = first_existing(iv.columns, ["route_rx", "route", "route_pharm"])

dose_value_col = first_existing(iv.columns, ["dose_val_rx", "dose_val", "dose"])

dose_unit_col = first_existing(iv.columns, ["dose_unit_rx", "dose_unit"])

drug_col = first_existing(iv.columns, ["drug", "drug_rx", "medication"])

formulary_col = first_existing(
    iv.columns, ["formulary_drug_cd", "formulary_drug_cd_rx"]
)

strength_col = first_existing(iv.columns, ["prod_strength", "prod_strength_rx"])

starttime_col = first_existing(
    iv.columns, ["starttime_rx", "starttime", "starttime_pharm"]
)

stoptime_col = first_existing(iv.columns, ["stoptime_rx", "stoptime", "stoptime_pharm"])

required_mapping = {
    "via": route_col,
    "valor da dose": dose_value_col,
    "unidade da dose": dose_unit_col,
    "horário inicial": starttime_col,
}

missing_mapping = [name for name, column in required_mapping.items() if column is None]

if missing_mapping:
    raise RuntimeError(
        "Não foi possível identificar estas colunas: " f"{missing_mapping}"
    )

print("\nMapeamento de colunas:")
print(f"  Via:                 {route_col}")
print(f"  Valor da dose:       {dose_value_col}")
print(f"  Unidade da dose:     {dose_unit_col}")
print(f"  Medicamento:         {drug_col}")
print(f"  Código farmacêutico: {formulary_col}")
print(f"  Concentração:        {strength_col}")
print(f"  Starttime:           {starttime_col}")
print(f"  Stoptime:            {stoptime_col}")

# ------------------------------------------------------------
# 7. Normalizar os campos
# ------------------------------------------------------------

iv["_route"] = iv[route_col].fillna("").astype(str).str.upper().str.strip()

iv["_dose_value"] = pd.to_numeric(iv[dose_value_col], errors="coerce")

iv["_dose_unit"] = iv[dose_unit_col].fillna("").astype(str).str.strip().str.lower()

if drug_col is not None:
    iv["_drug"] = iv[drug_col].fillna("").astype(str)
else:
    iv["_drug"] = ""

if formulary_col is not None:
    iv["_formulary"] = iv[formulary_col].fillna("").astype(str).str.upper().str.strip()
else:
    iv["_formulary"] = ""

if strength_col is not None:
    iv["_strength"] = iv[strength_col].fillna("").astype(str)
else:
    iv["_strength"] = ""

iv["_starttime"] = pd.to_datetime(iv[starttime_col], errors="coerce")

if stoptime_col is not None:
    iv["_stoptime"] = pd.to_datetime(iv[stoptime_col], errors="coerce")
else:
    iv["_stoptime"] = pd.NaT

# ------------------------------------------------------------
# 8. Extrair concentração em mg/mL da descrição
# ------------------------------------------------------------


def extract_mg_per_ml(text):
    """
    Extrai concentrações descritas como:
      5 mg / mL
      5 mg/mL
      10 mg per mL
    """

    if pd.isna(text):
        return np.nan

    normalized_text = str(text).lower()

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*mg\s*(?:/|per)\s*m[l]", normalized_text, flags=re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return np.nan


iv["_concentration_mg_ml"] = iv["_strength"].apply(extract_mg_per_ml)

# ------------------------------------------------------------
# 9. Mapear concentração pelo código farmacêutico
# ------------------------------------------------------------

# Mapeamento baseado nas formulações observadas no banco.
# Só será usado quando a concentração não tiver sido extraída
# diretamente de prod_strength.

formulary_concentration_map = {
    "MBLU50IV": 5.0,
    "MBLU50I": 5.0,
    "MBLU10I": 10.0,
    "METHB1I": 10.0,
    "MBLU1I": 10.0,
}

mapped_concentration = iv["_formulary"].map(formulary_concentration_map)

iv["_concentration_mg_ml"] = iv["_concentration_mg_ml"].fillna(mapped_concentration)

# ------------------------------------------------------------
# 10. Inicializar variáveis de limpeza
# ------------------------------------------------------------

iv["dose_mg_clean"] = np.nan
iv["dose_conversion_method"] = ""
iv["dose_valid_preliminary"] = False
iv["dose_exclusion_reason"] = ""

# ------------------------------------------------------------
# 11. Doses registradas diretamente em mg
# ------------------------------------------------------------

mask_direct_mg = iv["_dose_unit"].eq("mg") & iv["_dose_value"].notna()

iv.loc[mask_direct_mg, "dose_mg_clean"] = iv.loc[mask_direct_mg, "_dose_value"]

iv.loc[mask_direct_mg, "dose_conversion_method"] = "direct_mg"

# ------------------------------------------------------------
# 12. Converter doses registradas em mL
# ------------------------------------------------------------

mask_ml_convertible = (
    iv["_dose_unit"].eq("ml")
    & iv["_dose_value"].notna()
    & iv["_concentration_mg_ml"].notna()
)

iv.loc[mask_ml_convertible, "dose_mg_clean"] = (
    iv.loc[mask_ml_convertible, "_dose_value"]
    * iv.loc[mask_ml_convertible, "_concentration_mg_ml"]
)

iv.loc[mask_ml_convertible, "dose_conversion_method"] = "mL_times_concentration"

# ------------------------------------------------------------
# 13. Marcar motivos de exclusão
# ------------------------------------------------------------

mask_missing_numeric_dose = iv["_dose_value"].isna()

iv.loc[mask_missing_numeric_dose, "dose_exclusion_reason"] = "missing_numeric_dose"

mask_unit_g = iv["_dose_unit"].eq("g")

iv.loc[mask_unit_g, "dose_exclusion_reason"] = "dose_unit_g_probable_data_error"

mask_unit_mcg = iv["_dose_unit"].isin(["mcg", "ug", "µg"])

iv.loc[mask_unit_mcg, "dose_exclusion_reason"] = "dose_unit_mcg_not_systemic_mb_dose"

mask_unit_ampule = iv["_dose_unit"].isin(["amp", "ampule", "ampoule", "vial"])

iv.loc[mask_unit_ampule, "dose_exclusion_reason"] = "container_unit_ambiguous"

mask_ml_without_concentration = (
    iv["_dose_unit"].eq("ml") & iv["_concentration_mg_ml"].isna()
)

iv.loc[mask_ml_without_concentration, "dose_exclusion_reason"] = (
    "mL_without_known_concentration"
)

recognized_units = {
    "mg",
    "ml",
    "g",
    "mcg",
    "ug",
    "µg",
    "amp",
    "ampule",
    "ampoule",
    "vial",
}

mask_unknown_unit = ~iv["_dose_unit"].isin(recognized_units)

iv.loc[mask_unknown_unit, "dose_exclusion_reason"] = "unknown_dose_unit"

# ------------------------------------------------------------
# 14. Aplicar faixa ampla de plausibilidade
# ------------------------------------------------------------

# Essa faixa é utilizada apenas para controle de qualidade.
# Não é ainda um critério clínico definitivo de inclusão.

mask_plausible_dose = iv["dose_mg_clean"].between(10, 500, inclusive="both")

mask_outside_plausible_range = iv["dose_mg_clean"].notna() & ~mask_plausible_dose

iv.loc[mask_outside_plausible_range, "dose_exclusion_reason"] = (
    "dose_outside_10_to_500_mg"
)

# ------------------------------------------------------------
# 15. Definir dose preliminarmente válida
# ------------------------------------------------------------

iv["dose_valid_preliminary"] = (
    iv["dose_mg_clean"].notna()
    & mask_plausible_dose
    & iv["dose_exclusion_reason"].eq("")
)

# ------------------------------------------------------------
# 16. Inspecionar registros com unidade diferente de mg
# ------------------------------------------------------------

non_mg_records = iv[~iv["_dose_unit"].eq("mg")].copy()

qc_columns = [
    col
    for col in [
        "subject_id",
        "hadm_id",
        "pharmacy_id",
        starttime_col,
        drug_col,
        formulary_col,
        strength_col,
        route_col,
        dose_value_col,
        dose_unit_col,
        "_concentration_mg_ml",
        "dose_mg_clean",
        "dose_conversion_method",
        "dose_valid_preliminary",
        "dose_exclusion_reason",
    ]
    if col is not None and col in iv.columns
]

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 180)
pd.set_option("display.width", 260)

print("\n" + "=" * 90)
print("REGISTROS IV COM UNIDADE DIFERENTE DE mg")
print("=" * 90)

if non_mg_records.empty:
    print("Nenhum registro IV com unidade diferente de mg.")
else:
    # CORREÇÃO:
    # Ordenar antes de selecionar qc_columns.
    display(
        non_mg_records.sort_values(["_dose_unit", "_dose_value"], na_position="last")[
            qc_columns
        ]
    )

# ------------------------------------------------------------
# 17. Resumo da limpeza das doses
# ------------------------------------------------------------

dose_cleaning_summary = (
    iv.groupby(
        [
            "_dose_unit",
            "dose_conversion_method",
            "dose_valid_preliminary",
            "dose_exclusion_reason",
        ],
        dropna=False,
    )
    .agg(
        n_records=("subject_id", "size"),
        n_patients=("subject_id", "nunique"),
        dose_mg_nonmissing=("dose_mg_clean", lambda series: series.notna().sum()),
        dose_mg_median=("dose_mg_clean", "median"),
        dose_mg_p25=("dose_mg_clean", lambda series: series.quantile(0.25)),
        dose_mg_p75=("dose_mg_clean", lambda series: series.quantile(0.75)),
        dose_mg_min=("dose_mg_clean", "min"),
        dose_mg_max=("dose_mg_clean", "max"),
    )
    .reset_index()
    .sort_values(["dose_valid_preliminary", "n_records"], ascending=[False, False])
    .reset_index(drop=True)
)

print("\n" + "=" * 90)
print("RESUMO DA LIMPEZA DAS DOSES")
print("=" * 90)

display(dose_cleaning_summary)

# ------------------------------------------------------------
# 18. Manter doses preliminarmente válidas
# ------------------------------------------------------------

valid_iv = iv[iv["dose_valid_preliminary"]].copy()

print("\n" + "=" * 90)
print("EXPOSIÇÕES IV COM DOSE PRELIMINARMENTE VÁLIDA")
print("=" * 90)

print("Registros:", f"{len(valid_iv):,}")

print("Pacientes:", f"{valid_iv['subject_id'].nunique():,}")

print("Internações:", f"{valid_iv['hadm_id'].nunique():,}")

if not valid_iv.empty:
    print("Dose mediana:", round(valid_iv["dose_mg_clean"].median(), 2), "mg")

    print(
        "Dose P25–P75:",
        round(valid_iv["dose_mg_clean"].quantile(0.25), 2),
        "–",
        round(valid_iv["dose_mg_clean"].quantile(0.75), 2),
        "mg",
    )

# ------------------------------------------------------------
# 19. Carregar icustays
# ------------------------------------------------------------

icu_header = pd.read_csv(ICUSTAYS_PATH, compression="gzip", nrows=0)

icu_available_columns = icu_header.columns.tolist()

icu_preferred_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "first_careunit",
    "last_careunit",
    "intime",
    "outtime",
    "los",
]

icu_usecols = [col for col in icu_preferred_columns if col in icu_available_columns]

required_icu_columns = ["subject_id", "hadm_id", "stay_id", "intime", "outtime"]

missing_icu_columns = [col for col in required_icu_columns if col not in icu_usecols]

if missing_icu_columns:
    raise RuntimeError(
        "Colunas obrigatórias ausentes em icustays: " f"{missing_icu_columns}"
    )

icustays = pd.read_csv(
    ICUSTAYS_PATH, compression="gzip", usecols=icu_usecols, low_memory=False
)

icustays["intime"] = pd.to_datetime(icustays["intime"], errors="coerce")

icustays["outtime"] = pd.to_datetime(icustays["outtime"], errors="coerce")

print("\nEstadias de UTI carregadas:", f"{len(icustays):,}")
print("Pacientes com estadia de UTI:", f"{icustays['subject_id'].nunique():,}")

# ------------------------------------------------------------
# 20. Vincular prescrições à internação e estadia de UTI
# ------------------------------------------------------------

candidate_links = valid_iv.merge(
    icustays, on=["subject_id", "hadm_id"], how="left", suffixes=("", "_icu")
)

candidate_links["hours_from_icu_admission"] = (
    candidate_links["_starttime"] - candidate_links["intime"]
).dt.total_seconds() / 3600

candidate_links["hours_to_icu_discharge"] = (
    candidate_links["outtime"] - candidate_links["_starttime"]
).dt.total_seconds() / 3600

candidate_links["prescription_start_during_icu"] = (
    candidate_links["_starttime"].notna()
    & candidate_links["intime"].notna()
    & candidate_links["outtime"].notna()
    & (candidate_links["_starttime"] >= candidate_links["intime"])
    & (candidate_links["_starttime"] <= candidate_links["outtime"])
)

# ------------------------------------------------------------
# 21. Manter apenas prescrições iniciadas durante a UTI
# ------------------------------------------------------------

during_icu = candidate_links[candidate_links["prescription_start_during_icu"]].copy()

# ------------------------------------------------------------
# 22. Criar identificador único da prescrição
# ------------------------------------------------------------

if "pharmacy_id" in during_icu.columns:
    during_icu["_prescription_key"] = during_icu["pharmacy_id"].astype(str)

elif "poe_id" in during_icu.columns:
    during_icu["_prescription_key"] = during_icu["poe_id"].astype(str)

else:
    during_icu["_prescription_key"] = during_icu.index.astype(str)

# ------------------------------------------------------------
# 23. Resolver possíveis múltiplos vínculos de uma ordem
# ------------------------------------------------------------

# Em uma internação com múltiplas estadias de UTI, o horário
# deve pertencer somente à estadia temporalmente correspondente.
# Se houver rara sobreposição, escolhemos a admissão de UTI
# mais recente antes da prescrição.

during_icu = (
    during_icu.sort_values(["_prescription_key", "intime"], ascending=[True, False])
    .drop_duplicates(subset=["_prescription_key"], keep="first")
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# 24. Resumo da vinculação à UTI
# ------------------------------------------------------------

icu_link_summary = pd.DataFrame(
    {
        "metric": [
            "Valid IV prescription records",
            "Valid IV patients",
            "Valid IV admissions",
            "Valid IV prescriptions starting during ICU",
            "Patients with valid IV prescription during ICU",
            "Admissions with valid IV prescription during ICU",
            "ICU stays with valid IV prescription",
        ],
        "value": [
            len(valid_iv),
            valid_iv["subject_id"].nunique(),
            valid_iv["hadm_id"].nunique(),
            len(during_icu),
            during_icu["subject_id"].nunique(),
            during_icu["hadm_id"].nunique(),
            during_icu["stay_id"].nunique(),
        ],
    }
)

print("\n" + "=" * 90)
print("VINCULAÇÃO PRELIMINAR À UTI")
print("=" * 90)

display(icu_link_summary)

# ------------------------------------------------------------
# 25. Criar exposição índice preliminar por paciente
# ------------------------------------------------------------

if during_icu.empty:
    index_exposure = during_icu.copy()

else:
    sort_columns = ["subject_id", "_starttime"]

    if "pharmacy_id" in during_icu.columns:
        sort_columns.append("pharmacy_id")

    index_exposure = (
        during_icu.sort_values(sort_columns, na_position="last")
        .drop_duplicates(subset=["subject_id"], keep="first")
        .copy()
    )

index_exposure["mb_time_preliminary"] = index_exposure["_starttime"]

# ------------------------------------------------------------
# 26. Resumo da exposição índice
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("EXPOSIÇÃO ÍNDICE PRELIMINAR")
print("=" * 90)

print("Pacientes:", f"{len(index_exposure):,}")

if not index_exposure.empty:
    print("Internações:", f"{index_exposure['hadm_id'].nunique():,}")

    print("Estadias de UTI:", f"{index_exposure['stay_id'].nunique():,}")

    print("Dose mediana:", round(index_exposure["dose_mg_clean"].median(), 2), "mg")

    print(
        "Dose P25–P75:",
        round(index_exposure["dose_mg_clean"].quantile(0.25), 2),
        "–",
        round(index_exposure["dose_mg_clean"].quantile(0.75), 2),
        "mg",
    )

    print(
        "Horas após admissão na UTI — mediana:",
        round(index_exposure["hours_from_icu_admission"].median(), 2),
    )

    print(
        "Horas após admissão na UTI — P25–P75:",
        round(index_exposure["hours_from_icu_admission"].quantile(0.25), 2),
        "–",
        round(index_exposure["hours_from_icu_admission"].quantile(0.75), 2),
    )

# ------------------------------------------------------------
# 27. Número de prescrições por paciente
# ------------------------------------------------------------

if during_icu.empty:
    patient_exposure_counts = pd.DataFrame(
        columns=[
            "subject_id",
            "n_iv_mb_prescriptions",
            "n_icu_stays",
            "first_mb_time",
            "last_mb_time",
            "first_dose_mg",
            "total_prescribed_dose_mg",
            "exposure_span_hours",
        ]
    )

    prescriptions_per_patient = pd.DataFrame(columns=["n_prescriptions", "n_patients"])

else:
    patient_exposure_counts = (
        during_icu.sort_values(["subject_id", "_starttime"])
        .groupby("subject_id", as_index=False)
        .agg(
            n_iv_mb_prescriptions=("subject_id", "size"),
            n_icu_stays=("stay_id", "nunique"),
            first_mb_time=("_starttime", "min"),
            last_mb_time=("_starttime", "max"),
            first_dose_mg=("dose_mg_clean", "first"),
            total_prescribed_dose_mg=("dose_mg_clean", "sum"),
        )
    )

    patient_exposure_counts["exposure_span_hours"] = (
        patient_exposure_counts["last_mb_time"]
        - patient_exposure_counts["first_mb_time"]
    ).dt.total_seconds() / 3600

    prescriptions_per_patient = (
        patient_exposure_counts["n_iv_mb_prescriptions"]
        .value_counts()
        .sort_index()
        .rename_axis("n_prescriptions")
        .reset_index(name="n_patients")
    )

print("\n" + "=" * 90)
print("NÚMERO DE PRESCRIÇÕES IV POR PACIENTE")
print("=" * 90)

display(prescriptions_per_patient)

# ------------------------------------------------------------
# 28. Criar resumo da dose índice
# ------------------------------------------------------------

if index_exposure.empty:
    index_dose_summary = pd.DataFrame()

else:
    index_dose_summary = pd.DataFrame(
        {
            "metric": [
                "n_patients",
                "dose_mg_nonmissing",
                "dose_mg_mean",
                "dose_mg_sd",
                "dose_mg_median",
                "dose_mg_p25",
                "dose_mg_p75",
                "dose_mg_min",
                "dose_mg_max",
                "hours_from_icu_admission_median",
                "hours_from_icu_admission_p25",
                "hours_from_icu_admission_p75",
            ],
            "value": [
                len(index_exposure),
                index_exposure["dose_mg_clean"].notna().sum(),
                index_exposure["dose_mg_clean"].mean(),
                index_exposure["dose_mg_clean"].std(),
                index_exposure["dose_mg_clean"].median(),
                index_exposure["dose_mg_clean"].quantile(0.25),
                index_exposure["dose_mg_clean"].quantile(0.75),
                index_exposure["dose_mg_clean"].min(),
                index_exposure["dose_mg_clean"].max(),
                index_exposure["hours_from_icu_admission"].median(),
                index_exposure["hours_from_icu_admission"].quantile(0.25),
                index_exposure["hours_from_icu_admission"].quantile(0.75),
            ],
        }
    )

# ------------------------------------------------------------
# 29. Salvar outputs
# ------------------------------------------------------------

timestamp = run_stamp()

clean_all_path = COHORT_DIR / f"methylene_blue_iv_dose_cleaning_{timestamp}.csv.gz"

valid_iv_path = COHORT_DIR / f"methylene_blue_valid_iv_prescriptions_{timestamp}.csv.gz"

candidate_links_path = (
    COHORT_DIR / f"methylene_blue_iv_icu_candidate_links_{timestamp}.csv.gz"
)

during_icu_path = COHORT_DIR / f"methylene_blue_valid_iv_during_icu_{timestamp}.csv.gz"

index_path = (
    COHORT_DIR / f"methylene_blue_preliminary_index_exposure_{timestamp}.csv.gz"
)

non_mg_path = QC_DIR / f"methylene_blue_non_mg_iv_records_{timestamp}.csv"

dose_summary_path = QC_DIR / f"methylene_blue_dose_cleaning_summary_{timestamp}.csv"

icu_summary_path = QC_DIR / f"methylene_blue_icu_linkage_summary_{timestamp}.csv"

patient_counts_path = QC_DIR / f"methylene_blue_patient_exposure_counts_{timestamp}.csv"

prescriptions_per_patient_path = (
    QC_DIR / f"methylene_blue_prescriptions_per_patient_{timestamp}.csv"
)

index_dose_summary_path = QC_DIR / f"methylene_blue_index_dose_summary_{timestamp}.csv"

report_path = QC_DIR / f"methylene_blue_code6_report_{timestamp}.txt"

iv.to_csv(clean_all_path, index=False, compression="gzip")

valid_iv.to_csv(valid_iv_path, index=False, compression="gzip")

candidate_links.to_csv(candidate_links_path, index=False, compression="gzip")

during_icu.to_csv(during_icu_path, index=False, compression="gzip")

index_exposure.to_csv(index_path, index=False, compression="gzip")

non_mg_records.to_csv(non_mg_path, index=False)

dose_cleaning_summary.to_csv(dose_summary_path, index=False)

icu_link_summary.to_csv(icu_summary_path, index=False)

patient_exposure_counts.to_csv(patient_counts_path, index=False)

prescriptions_per_patient.to_csv(prescriptions_per_patient_path, index=False)

index_dose_summary.to_csv(index_dose_summary_path, index=False)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("METHYLENE BLUE — CODE 6 REPORT\n")

    file.write("=" * 60 + "\n\n")

    file.write(f"IV source file: {IV_PATH}\n")

    file.write(f"ICU stays source file: {ICUSTAYS_PATH}\n\n")

    file.write(f"Initial IV records: {len(iv)}\n")

    file.write(f"Initial IV patients: " f"{iv['subject_id'].nunique()}\n")

    file.write(f"Valid IV dose records: {len(valid_iv)}\n")

    file.write(f"Patients with valid IV dose: " f"{valid_iv['subject_id'].nunique()}\n")

    file.write(f"Valid prescriptions starting during ICU: " f"{len(during_icu)}\n")

    file.write(
        f"Patients with valid prescription during ICU: "
        f"{during_icu['subject_id'].nunique()}\n"
    )

    file.write(f"Preliminary index-exposure patients: " f"{len(index_exposure)}\n")

    if not index_exposure.empty:
        file.write(
            f"Index dose median: " f"{index_exposure['dose_mg_clean'].median()}\n"
        )

        file.write(
            f"Index dose P25: " f"{index_exposure['dose_mg_clean'].quantile(0.25)}\n"
        )

        file.write(
            f"Index dose P75: " f"{index_exposure['dose_mg_clean'].quantile(0.75)}\n"
        )

print("\n" + "=" * 90)
print("OUTPUTS SALVOS")
print("=" * 90)

print(clean_all_path)
print(valid_iv_path)
print(candidate_links_path)
print(during_icu_path)
print(index_path)
print(non_mg_path)
print(dose_summary_path)
print(icu_summary_path)
print(patient_counts_path)
print(prescriptions_per_patient_path)
print(index_dose_summary_path)
print(report_path)

gc.collect()

print("\nCÓDIGO 6 CONCLUÍDO COM SUCESSO.")
