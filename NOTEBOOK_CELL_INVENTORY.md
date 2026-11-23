# Notebook cell inventory

Cell numbers are one-based. The supplied MB_HIRID(3).ipynb and the requested MB_HIRID(2).ipynb are byte-for-byte identical. Their SHA-256 is `505f12f62ff1e9262ee77323465ac582e1e41fa0b1e0e8073c4703c30b2a77a2`. The first 74 source cells also match the companion revision notebooks. Additional entries refer to MB_HIRID_revision_robustness (1).ipynb.

## Original notebook, cells 1–74

| Cell | Role | Disposition |
|---:|---|---|
| 1 | HiRID archive inventory and schema preview | Excluded, README lists the raw archives. |
| 2 | HiRID reference inspection and MB search | Excluded, exploratory. |
| 3 | HiRID candidate MB IDs | Excluded, no HiRID MB response analysis. |
| 4 | HiRID MB medication audit | Excluded, feasibility only. |
| 5 | Initial HiRID variable lookup | Replaced by the frozen IDs in cell 7. |
| 6 | Refined HiRID variable and IV antibiotic lookup | Replaced by cell 7. |
| 7 | Frozen HiRID variable IDs | hirid/01_variable_map.py |
| 8 | Early HiRID cohort definition | Replaced by cell 9. |
| 9 | Final HiRID cohort with >24 h antibiotic course | hirid/02_cohort.py |
| 10 | HiRID hemodynamics and norepinephrine extraction | hirid/03_hemodynamic_extraction.py |
| 11 | HiRID norepinephrine cleanup | hirid/04_norepinephrine_cleaning.py |
| 12 | HiRID weight selection and dose normalization | hirid/05_weight_normalization.py |
| 13 | GMM exploration and weighted follow-up derivation | hirid/06_gmm_exploration.py, VNERi removed. |
| 14 | Three-cluster GMM covariance sensitivity | hirid/07_gmm_sensitivity.py, VNERi removed. |
| 15 | Prespecified DBP/PP phenotypes, threshold sensitivity and scatterplot | hirid/08_prespecified_phenotypes.py, Figure 2 scientific content, VNERi removed. |
| 16 | Earlier HiRID supplement builder | Replaced by cell 18. |
| 17 | HiRID clinical extraction and construct validation | hirid/09_construct_validation.py, VNERi model removed. |
| 18 | Final HiRID physiological summary and frozen rules | hirid/10_physiological_summary.py, S1 sources, VNERi removed. |
| 19 | Earlier two-phase flowchart | Replaced by the study diagram from cell 69. |
| 20 | MIMIC-IV file inventory | Excluded, configurable paths and README replace it. |
| 21 | MB discovery attempt in ICU inputevents | Excluded, final exposure comes from prescriptions and pharmacy. |
| 22 | Search for hospital medication source files | Excluded, README lists source files. |
| 23 | Extract MB from prescriptions | mimic_iv/01_prescriptions.py |
| 24 | Pharmacy linkage and IV candidates | mimic_iv/02_pharmacy_validation.py |
| 25 | Dose cleanup and index exposure linked to ICU | mimic_iv/03_index_exposure.py |
| 26 | Initial characterization and ICD extraction | mimic_iv/05_extract_index_diagnoses.py retains ICD extraction. Cell 27 supersedes characterization. |
| 27 | Corrected index characterization | mimic_iv/04_characterize_index_cohort.py, source of S8 variables. |
| 28 | Coded sepsis, active norepinephrine and methemoglobinemia exclusions | mimic_iv/06_sepsis_and_active_norepinephrine.py |
| 29 | Invasive-only BP draft | Replaced by cell 35. |
| 30 | Early BP fallback draft | Replaced by cell 35. |
| 31 | Intermediate correction to BP draft | Replaced by cell 35. |
| 32 | Colab Drive remount | Excluded. |
| 33 | Colab Drive restart | Excluded. |
| 34 | Alternative Google Drive mount | Excluded. |
| 35 | Final pre-MB BP pairing, fallback and phenotype assignment | mimic_iv/07_pre_mb_blood_pressure.py |
| 36 | Individual QC for unclassifiable patients | Excluded, cell 35 keeps cohort QC. |
| 37 | Final baseline laboratories | mimic_iv/08_baseline_laboratories.py |
| 38 | Vasoactive item discovery | IDs retained by cell 39. |
| 39 | Baseline vasoactives and NEE | mimic_iv/09_baseline_vasoactives.py |
| 40 | QC of coincident same-agent infusion records | Excluded as a diagnostic view. Infusion handling remains in cells 39 and 54. |
| 41 | Ventilation event extraction and preliminary classifier | mimic_iv/10_mechanical_ventilation.py retains extraction. |
| 42 | Inspect unclassified ventilation events | Excluded. |
| 43 | Extended unclassified-event preview | Excluded. |
| 44 | Corrected device and mode hierarchy | mimic_iv/10_mechanical_ventilation.py retains hierarchy. |
| 45 | Final ventilation mode update | mimic_iv/10_mechanical_ventilation.py retains final inference and write. |
| 46 | RRT item discovery | IDs retained by cell 47. |
| 47 | Index-time RRT extraction | mimic_iv/11_renal_replacement.py |
| 48 | Install DOCX support in Colab | requirements.txt |
| 49 | First Table 1 draft | Replaced by cell 52. |
| 50 | Streamlined Table 1 draft | Replaced by cell 52. |
| 51 | MB dose per kg using extracted weight | mimic_iv/12_mb_weight_normalization.py |
| 52 | Final corrected Table 1 and frozen cohort | analyses/table_1.py |
| 53 | Colab Drive mount | Excluded. |
| 54 | Minute-to-hour NEE trajectory through 72 h | mimic_iv/13_hourly_vasopressor_trajectory.py |
| 55 | Read back hourly NEE QC | Excluded, QC exports remain in cell 54. |
| 56 | Follow-up, death and ICU discharge audit | mimic_iv/14_followup_audit.py |
| 57 | Early response and plot draft | Replaced by cell 60 and revision cell 85. |
| 58 | Display response and inspect outliers | Excluded. |
| 59 | Individual extreme-NEE diagnostic | Excluded. |
| 60 | AUC, early longitudinal input and preliminary mixed model | analyses/auc.py retains AUC and early longitudinal input. Superseded fit and plot removed. |
| 61 | Earlier categorical GEE | Replaced by revision cell 83 and sensitivity cells 91–94. |
| 62 | Colab Drive mount | Excluded. |
| 63 | Early NEE/AUC figure | Replaced by revision cell 85. |
| 64 | Figure using earlier GEE output | Replaced by revision cell 85. |
| 65 | Earlier risk-count formatting patch | Replaced by revision cell 85. |
| 66 | Raw-median trajectory figure | Replaced by revision cell 85. |
| 67 | Exploratory sustained vasopressor liberation endpoint | Excluded, outside the revised manuscript. |
| 68 | Exploratory competing risks in R | Excluded, outside the revised manuscript. |
| 69 | Study design diagram | analyses/figure_1.py, corrected annotation to the DBP/PP rule. |
| 70 | Colab Drive mount | Excluded. |
| 71 | Inspect saved cohort files | Excluded. |
| 72 | Display saved summaries | Excluded. |
| 73 | Corrected Table 2 | analyses/table_2.py, original numeric method with the revised primary GEE result in its footnote. |
| 74 | Record environment versions | analyses/environment.py and requirements.txt |

## Revision notebook additions, cells 75–100

| Cell | Role | Disposition |
|---:|---|---|
| 75 | Find Colab mount | Excluded. |
| 76 | Earlier BP quality audit | Replaced by cells 78–79. |
| 77 | Locate and preview candidate artifacts | Excluded. |
| 78 | BP classification and physiology audit | analyses/bp_quality.py |
| 79 | Measurement QC in the classifiable cohort | analyses/bp_quality.py, S2. |
| 80 | Inspect longitudinal schema | Excluded. |
| 81 | Early NEE counts and zero audit | analyses/primary_gee_and_figure_3.py. Final counts follow death correction. |
| 82 | Post-death row audit and filter | analyses/primary_gee_and_figure_3.py, S4. |
| 83 | Final death-corrected primary GEE | analyses/primary_gee_and_figure_3.py, S3 and primary model. |
| 84 | Preliminary marginal means and contrasts | Replaced by cell 85. |
| 85 | Final covariance-aware marginal means, contrasts and Figure 3 | analyses/primary_gee_and_figure_3.py, S5–S6. |
| 86 | Colab Drive mount | Excluded. |
| 87 | Interactive project path check | Excluded. |
| 88 | Interactive HiRID path check | Excluded. |
| 89 | Enumerate project files | Excluded. |
| 90 | Reload sources for sensitivities | Relevant file reads in analyses/sensitivity.py. |
| 91 | Invasive-only sensitivity | analyses/sensitivity.py, S7. |
| 92 | Complete seven-timepoint sensitivity | analyses/sensitivity.py, S7. |
| 93 | Adjusted sensitivity covariate and complete-case preparation | analyses/sensitivity.py retains preparation. Superseded fit used untransformed ICU-to-MB hours. |
| 94 | Final adjusted post-MB sensitivity | analyses/sensitivity.py, S7, uses transformed ICU-to-MB hours. |
| 95 | HiRID count audit with missing paths | Superseded by verify.py on generated output. |
| 96 | Interactive source path check | Excluded. |
| 97 | Search for HiRID artifacts | Excluded. |
| 98 | Colab Drive mount | Excluded. |
| 99 | Interactive HiRID path check | Excluded. |
| 100 | Final HiRID sample size audit | Historical evidence for n=1,050. verify.py checks the named cohort. |

New files are `analyses/table_s8.py` for the requested descriptive analysis and `run.py`, `common.py`, and `verify.py` for execution and verification.
