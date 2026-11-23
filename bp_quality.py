"""Source: revision notebook cells 78–79. Run through run.py."""

from common import artifact, display, phase_dir

# ============================================================
# CÓDIGO 16A
# AUDITORIA DA CLASSIFICAÇÃO PRÉ-MB
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# localizar diretório do projeto
# ------------------------------------------------------------

ROOT = phase_dir("mimic_iv")

COHORT = artifact(
    phase_dir("mimic_iv"), "TABLE1_BASE_FINAL_combined_bp_phenotype_*.csv.gz"
)

HEMO = artifact(
    phase_dir("mimic_iv"), "methylene_blue_final_pre_mb_hemodynamics_*.csv.gz"
)

BP_EVENTS = artifact(phase_dir("mimic_iv"), "methylene_blue_pre_mb_bp_events_*.csv.gz")

assert COHORT.exists()
assert HEMO.exists()
assert BP_EVENTS.exists()

cohort = pd.read_csv(COHORT, compression="gzip")
hemo = pd.read_csv(HEMO, compression="gzip")
bp = pd.read_csv(BP_EVENTS, compression="gzip")

print("=" * 70)
print("AUDITORIA DA CLASSIFICAÇÃO")
print("=" * 70)

print("\nPacientes da coorte:")
print(len(cohort))

print("\nPacientes classificáveis:")
print(cohort["hemodynamic_phenotype"].value_counts(dropna=False))

print("\n")

# ------------------------------------------------------------
# modalidade
# ------------------------------------------------------------

if "bp_modality_selected" in cohort.columns:

    print("Modalidade utilizada")

    display(
        cohort["bp_modality_selected"]
        .value_counts(dropna=False)
        .rename_axis("Modality")
        .reset_index(name="Patients")
    )

# ------------------------------------------------------------
# disponibilidade
# ------------------------------------------------------------

cols = [c for c in hemo.columns if "paired" in c.lower()]

if len(cols):

    print("\nNúmero de medidas pareadas")

    display(hemo[cols].describe().T)

# ------------------------------------------------------------
# resumo fisiológico
# ------------------------------------------------------------

# ------------------------------------------------------------
# Resumo fisiológico por fenótipo
# ------------------------------------------------------------

physiology_summary = (
    cohort.loc[cohort["hemodynamic_phenotype"].ne("Unavailable")]
    .groupby("hemodynamic_phenotype", observed=True)
    .agg(
        n=("subject_id", "nunique"),
        median_dbp_pre_mb=("dbp_pre_mb_median", "median"),
        q1_dbp_pre_mb=("dbp_pre_mb_median", lambda x: x.quantile(0.25)),
        q3_dbp_pre_mb=("dbp_pre_mb_median", lambda x: x.quantile(0.75)),
        median_pp_pre_mb=("pp_pre_mb_median", "median"),
        q1_pp_pre_mb=("pp_pre_mb_median", lambda x: x.quantile(0.25)),
        q3_pp_pre_mb=("pp_pre_mb_median", lambda x: x.quantile(0.75)),
        median_paired_measurements=("n_selected_paired_bp", "median"),
        min_paired_measurements=("n_selected_paired_bp", "min"),
        max_paired_measurements=("n_selected_paired_bp", "max"),
    )
    .reset_index()
)

display(physiology_summary)

OUT = ROOT / "revision_analyses"
OUT.mkdir(parents=True, exist_ok=True)

physiology_summary.to_csv(
    OUT / "audit_pre_MB_physiology_by_phenotype.csv",
    index=False,
)

print("\nArquivo salvo em:")
print(OUT / "audit_pre_MB_physiology_by_phenotype.csv")

# ------------------------------------------------------------
# salvar auditoria
# ------------------------------------------------------------

OUT = ROOT / "revision_analyses"

OUT.mkdir(exist_ok=True)

cohort.to_csv(
    OUT / "audit_final_classification.csv",
    index=False,
)

print("\nArquivo salvo em:")
print(OUT / "audit_final_classification.csv")
# ------------------------------------------------------------
# Qualidade das medidas nos pacientes classificáveis
# ------------------------------------------------------------

classifiable = cohort[cohort["hemodynamic_phenotype"].ne("Unavailable")].copy()

measurement_qc = pd.DataFrame(
    {
        "metric": [
            "Classifiable patients",
            "Invasive BP",
            "Non-invasive fallback",
            "Median paired measurements",
            "IQR paired measurements",
            "Minimum paired measurements",
            "Maximum paired measurements",
            "Patients with 1 paired measurement",
            "Patients with >=2 paired measurements",
        ],
        "value": [
            classifiable["subject_id"].nunique(),
            classifiable["bp_modality_selected"].eq("Invasive").sum(),
            classifiable["bp_modality_selected"].eq("Non-invasive").sum(),
            classifiable["n_selected_paired_bp"].median(),
            (
                f"{classifiable['n_selected_paired_bp'].quantile(0.25):.0f}–"
                f"{classifiable['n_selected_paired_bp'].quantile(0.75):.0f}"
            ),
            classifiable["n_selected_paired_bp"].min(),
            classifiable["n_selected_paired_bp"].max(),
            classifiable["n_selected_paired_bp"].eq(1).sum(),
            classifiable["n_selected_paired_bp"].ge(2).sum(),
        ],
    }
)

display(measurement_qc)

measurement_qc.to_csv(
    OUT / "audit_pre_MB_measurement_quality.csv",
    index=False,
)

print("\nAuditoria 16A principal concluída.")
