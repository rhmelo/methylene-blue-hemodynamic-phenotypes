"""Source: original notebook cell 51. Run through run.py."""

from common import artifact, display, phase_dir, run_stamp

# ============================================================
# ADICIONAR DOSE DE AZUL DE METILENO EM mg/kg
#
# Peso:
#   mediana do patientweight registrado nos eventos vasoativos
#   basais do paciente
#
# Faixa plausível:
#   30 a 250 kg
# ============================================================

import pandas as pd
import numpy as np

PROJECT_DIR = phase_dir("mimic_iv")

COHORT_DIR = PROJECT_DIR / "cohorts"
QC_DIR = PROJECT_DIR / "qc"

# ------------------------------------------------------------
# 1. Localizar a base final e os eventos vasoativos
# ------------------------------------------------------------

base_path = artifact(COHORT_DIR, "TABLE1_BASE_FINAL_COMPLETE_[0-9]*.csv.gz")
vasoactive_path = artifact(
    COHORT_DIR, "methylene_blue_baseline_vasoactive_events_*.csv.gz"
)

print("Base:")
print(base_path)

print("\nEventos vasoativos:")
print(vasoactive_path)

# ------------------------------------------------------------
# 2. Carregar apenas as colunas necessárias
# ------------------------------------------------------------

base = pd.read_csv(base_path, compression="gzip", low_memory=False)

vaso = pd.read_csv(
    vasoactive_path,
    compression="gzip",
    usecols=["subject_id", "hadm_id", "stay_id", "patientweight"],
    low_memory=False,
)

vaso["patientweight"] = pd.to_numeric(vaso["patientweight"], errors="coerce")

# Manter somente pesos plausíveis
vaso.loc[~vaso["patientweight"].between(30, 250, inclusive="both"), "patientweight"] = (
    np.nan
)

# ------------------------------------------------------------
# 3. Peso mediano por paciente
# ------------------------------------------------------------

weight_summary = vaso.groupby(["subject_id", "hadm_id", "stay_id"], as_index=False).agg(
    baseline_weight_kg=("patientweight", "median"),
    n_weight_records=("patientweight", "count"),
)

# ------------------------------------------------------------
# 4. Integrar e calcular mg/kg
# ------------------------------------------------------------

for column in ["subject_id", "hadm_id", "stay_id"]:
    base[column] = pd.to_numeric(base[column], errors="coerce").astype("Int64")

    weight_summary[column] = pd.to_numeric(
        weight_summary[column], errors="coerce"
    ).astype("Int64")

base = base.drop(
    columns=["baseline_weight_kg", "n_weight_records", "mb_dose_mg_per_kg"],
    errors="ignore",
)

base_with_weight = base.merge(
    weight_summary,
    on=["subject_id", "hadm_id", "stay_id"],
    how="left",
    validate="one_to_one",
)

base_with_weight["dose_mg_clean"] = pd.to_numeric(
    base_with_weight["dose_mg_clean"], errors="coerce"
)

base_with_weight["mb_dose_mg_per_kg"] = (
    base_with_weight["dose_mg_clean"] / base_with_weight["baseline_weight_kg"]
)

# ------------------------------------------------------------
# 5. QC
# ------------------------------------------------------------

qc = pd.DataFrame(
    {
        "metric": [
            "Patients in cohort",
            "Weight available",
            "Methylene blue dose available",
            "Methylene blue dose in mg/kg available",
            "Median weight, kg",
            "Median methylene blue dose, mg/kg",
            "P25 methylene blue dose, mg/kg",
            "P75 methylene blue dose, mg/kg",
            "Minimum methylene blue dose, mg/kg",
            "Maximum methylene blue dose, mg/kg",
        ],
        "value": [
            len(base_with_weight),
            base_with_weight["baseline_weight_kg"].notna().sum(),
            base_with_weight["dose_mg_clean"].notna().sum(),
            base_with_weight["mb_dose_mg_per_kg"].notna().sum(),
            base_with_weight["baseline_weight_kg"].median(),
            base_with_weight["mb_dose_mg_per_kg"].median(),
            base_with_weight["mb_dose_mg_per_kg"].quantile(0.25),
            base_with_weight["mb_dose_mg_per_kg"].quantile(0.75),
            base_with_weight["mb_dose_mg_per_kg"].min(),
            base_with_weight["mb_dose_mg_per_kg"].max(),
        ],
    }
)

print("\n" + "=" * 100)
print("QC DA DOSE EM mg/kg")
print("=" * 100)

display(qc)

# ------------------------------------------------------------
# 6. Salvar nova base
# ------------------------------------------------------------

timestamp = run_stamp()

updated_base_path = (
    COHORT_DIR / f"TABLE1_BASE_FINAL_COMPLETE_WITH_MB_MGKG_{timestamp}.csv.gz"
)

qc_path = QC_DIR / f"methylene_blue_dose_mgkg_qc_{timestamp}.csv"

base_with_weight.to_csv(updated_base_path, index=False, compression="gzip")

qc.to_csv(qc_path, index=False)

print("\nNova base salva em:")
print(updated_base_path)

print("\nQC salvo em:")
print(qc_path)
