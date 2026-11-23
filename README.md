# Macrohemodynamic phenotypes and methylene blue response in septic shock

This two-phase retrospective study characterized prespecified physiological phenotypes in HiRID and applied the same thresholds in a MIMIC-IV cohort with coded sepsis or septic shock receiving intravenous methylene blue (MB) while on norepinephrine. The external phase evaluated early norepinephrine-equivalent dose (NEE) trajectories after MB.

The phenotypes are vasoplegia-compatible (diastolic blood pressure [DBP] <50 and pulse pressure [PP] ≥40 mmHg), mixed/low-pulsatility (DBP <50 and PP <40 mmHg), and preserved diastolic pressure (DBP ≥50 mmHg). HiRID sets time zero at the first qualifying norepinephrine episode and uses the ensuing first hour. MIMIC-IV uses the first plausible IV MB exposure linked to the index ICU stay and paired blood pressure in the preceding hour. Invasive measurements take priority over oscillometric measurements.

This package is for author review. See the [reproducibility audit](REPRODUCIBILITY_AUDIT.md) and [notebook cell inventory](NOTEBOOK_CELL_INVENTORY.md).

## Database access and files

Obtain individual credentialed PhysioNet access for each database and complete the required training and data-use agreement. The pipeline reads local extracts. [HiRID on PhysioNet](https://physionet.org/content/hirid/1.1.1/) · [MIMIC-IV on PhysioNet](https://physionet.org/content/mimiciv/) · [MIMIC access guide](https://mimic.mit.edu/docs/faq/how-to-get-access.html)

| Input root | Expected files |
|---|---|
| HiRID | `reference_data.tar.gz` containing `hirid_variable_reference.csv` and `general_table.csv`, `observation_tables_parquet.tar.gz`, `pharma_records_parquet.tar.gz` |
| MIMIC-IV hospital | `prescriptions.csv.gz`, `pharmacy.csv.gz`, `patients.csv.gz`, `admissions.csv.gz`, `diagnoses_icd.csv.gz`, `d_icd_diagnoses.csv.gz`, `d_labitems.csv.gz`, `labevents.csv.gz` |
| MIMIC-IV ICU | `icustays.csv.gz`, `d_items.csv.gz`, `inputevents.csv.gz`, `chartevents.csv.gz`, `procedureevents.csv.gz` |

MIMIC-IV files may be in the standard `hosp/` and `icu/` directories or a flat directory. Use one database snapshot per input root. The original MIMIC-IV release was not recorded. The HiRID archives match the 1.1.1 distribution layout, but the original snapshot was not independently confirmed. Record release numbers, input checksums and the Python environment in your authorized workspace. Large files are processed by partition or chunk, with cohort-level intermediate data held in memory. Allocate disk space for raw sources, intermediate data and logs.

## Setup and execution

Use Python 3.12. The archived environment reports Python 3.12.13. `requirements.txt` pins the direct analytical dependencies to recorded versions. The Patsy pin is a compatible version selected for this package and was absent from the Colab report.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py list
python run.py all \
  --hirid-dir /secure/raw/hirid \
  --mimic-dir /secure/raw/mimic-iv \
  --work-dir /secure/study-run
python verify.py --work-dir /secure/study-run
```

The equivalent environment variables are `HIRID_SOURCE_DIR`, `MIMIC_IV_SOURCE_DIR` and `MB_WORK_DIR`. You can run the groups `hirid`, `mimic` and `analyses` separately in that order, or select an individual stage from `python run.py list`. Run one process per work directory. On rerun, successful stages are skipped and failed stages can be retried with the same command. Use a fresh work directory for a different snapshot or a clean rerun. To start from authorized intermediates, place exactly one file matching each stage’s input pattern under the corresponding subdirectory of a private work directory. Review the scripts’ input paths and schemas before substituting intermediates.

## Directories, execution order and outputs

| Stage | Operation | Selected outputs relative to the work directory |
|---|---|---|
| `hirid/` 01–05 | Map variables, identify a qualifying norepinephrine and IV antibiotic episode with invasive BP coverage, reconstruct hemodynamics and normalize dose by weight. | `hirid/derived_data/step02_*` through `step04_*`, with flow and QC tables under `hirid/outputs/` |
| `hirid/` 06–10 | Document the exploratory GMM sensitivity, apply the prespecified DBP/PP classification, evaluate persistent hypotension and subsequent norepinephrine burden, and freeze the operational rules. | `hirid/derived_data/step07_*` and `step09_*`, Figure 2 scatterplot under `hirid/outputs/figures/`, and S1 source definitions under `hirid/HiRID_Final_Physiological_Package/` |
| `mimic_iv/` 01–07 | Extract MB from prescriptions and pharmacy, link index plausible IV exposure to ICU, screen coded sepsis and active norepinephrine, exclude methemoglobinemia, and classify pre-MB BP. | `mimic_iv/cohorts/TABLE1_BASE_FINAL_combined_bp_phenotype_*.csv.gz` plus eligibility flow and QC tables |
| `mimic_iv/` 08–12 and `analyses/table_1.py` | Extract labs, vasoactives, ventilation and renal replacement, derive MB dose per kg and freeze the analytic cohort. | `mimic_iv/tables/Table_1_FINAL_corrected_*.csv`, DOCX under `documents/`, test log under `qc/`, and `cohorts/METHYLENE_BLUE_ANALYTIC_COHORT_FROZEN_FINAL_*.csv.gz` |
| `mimic_iv/` 13–14 | Reconstruct minute-level vasopressors and aggregate hourly NEE. Audit death, ICU discharge and subsequent ICU stays. | `mimic_iv/cohorts/methylene_blue_hourly_NEE_*.csv.gz`; follow-up and short follow-up CSVs under `mimic_iv/qc/` |
| `analyses/` BP, AUC, primary and sensitivities | Evaluate BP quality (S2), complete-case AUC, corrected time-point counts and deaths (S3–S4), revised GEE marginal means and contrasts (S5–S6), and sensitivities (S7). | `mimic_iv/revision_analyses/` tables, model summaries, `robustness_analyses_FINAL_CORRECTED.csv` and Figure 3 PNG/PDF/TIFF; AUC CSVs under `mimic_iv/tables/` and `cohorts/` |
| `analyses/` tables and figure | Create complete-case Table 2, descriptive Table S8, the Figure 1 study diagram, and the software environment report. | Table 2 CSV and DOCX, `mimic_iv/tables/Table_S8_descriptive_outcomes.csv`, Figure 1 PNG/PDF/SVG under `figures/`, and `qc/software_environment.csv` |

## Outcome conventions

- MIMIC-IV baseline labs use the last valid measurement in the 24 hours preceding MB, using the source item dictionaries, value rules and time boundaries in the extraction script. Baseline and longitudinal vasopressor conversion and plausibility rules are retained in their respective extraction stages. The longitudinal NEE formula is norepinephrine + epinephrine + phenylephrine/10 + dopamine/100 + 2.5 × vasopressin + 10 × angiotensin II after normalization. Dobutamine is tracked separately.
- The final longitudinal model uses hourly median NEE with a `log1p` transformation. It excludes intervals beginning at or after recorded death and retains the interval in which death occurs if it began before death. Available zero NEE values remain zero. Hourly extraction stops at the earlier of 72 h or ICU discharge.
- The primary GEE uses Gaussian errors, identity link, exchangeable correlation, categorical interval and phenotype, and robust Wald inference for the interaction. The primary references are preserved DBP and baseline. Sensitivities restrict to invasive pressure, restrict to all seven valid time points, or fit the six post-MB intervals with standardized transformed baseline NEE, pre-MB lactate and transformed ICU-to-MB hours.
- AUC sums the hourly means weighted by observed minutes and subtracts six times baseline median NEE. Inference and Table 2 require a complete baseline, six full post-MB hours and survival through 6 h. The archived Figure 3B plots every nonmissing AUC and displays the complete-case p-value. See the audit for the corresponding sample sizes.
- S8 uses the existing hospital mortality flag and index ICU mortality indicator. ICU death is based on the recorded death timestamp and ICU stay bounds. Hospital LOS is admission to discharge and index ICU LOS is ICU admission to ICU discharge. The S8 output reports available denominators, missing values, mortality counts and percentages, and LOS medians with quartiles, overall and by phenotype.

Keep every intermediate and output in the authorized environment. Diagnostics can contain patient identifiers, and some figures display individual observations. The repository itself contains only code and documentation.
