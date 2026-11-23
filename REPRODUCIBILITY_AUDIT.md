# Reproducibility audit

## Source and benchmark reconciliation

Reviewed all 74 cells in MB_HIRID(2).ipynb. The supplied MB_HIRID(3).ipynb is byte-for-byte identical. The companion revision notebooks include the post-death correction, Figure 3, and the robustness analyses that the 74-cell copy lacks. The larger companion, MB_HIRID_revision_robustness (1).ipynb, provides the sensitivity models. The benchmarks were checked against archived execution outputs and compared with the revised manuscript, its Supplementary Appendix, and Tables 1–2. These comparisons establish notebook provenance. Exact reproduction on credentialed database snapshots remains unverified.

| Manuscript benchmark | Matching archived execution evidence | Ported source |
|---|---|---|
| HiRID n=1,050 | Original cell 9 cohort flow and cell 15 classification, with 355 vasoplegia-compatible, 215 mixed/low-pulsatility and 480 preserved DBP. Revision cell 100 independently records 1,050 unique patients in the Stage 07–09 extracts. | `hirid/02_cohort.py`, `hirid/08_prespecified_phenotypes.py` |
| MIMIC-IV n=59 | Original cell 35 starts with 66 screened patients, assigns 53 invasive and 6 non-invasive phenotypes, and leaves 7 unclassifiable. Original cell 52 freezes the 59-patient cohort. | `mimic_iv/07_pre_mb_blood_pressure.py`, `analyses/table_1.py` |
| Primary longitudinal GEE, 59 patients and 387 observations, p=0.019 | Revision cell 83 reports 387 observations, Wald χ²=24.216186 with 12 df and p=0.019006. Cell 82 excludes 19 intervals entirely after death from 8 patients, including 16 zero-NEE intervals. | `analyses/primary_gee_and_figure_3.py` |
| Baseline-adjusted 6-h NEE AUC, p=0.691 | Original cells 60 and 73 report p=0.691000 in 50 eligible survivors with full observed intervals. | `analyses/auc.py`, `analyses/table_2.py` |
| Invasive-only GEE, p=0.002 | Revision cell 91 reports 53 patients, 349 observations, p=0.002320. | `analyses/sensitivity.py` |
| Complete longitudinal follow-up GEE, p=0.010 | Revision cell 92 reports 51 patients, 357 observations, p=0.010240. Follow-up means a valid observation at each of the seven model time points. | `analyses/sensitivity.py` |
| Adjusted post-MB GEE, p=0.018 | Revision cell 94 reports 58 patients, 322 observations, Wald χ²=21.400042 with 10 df and p=0.018470. | `analyses/sensitivity.py` |

The older cell 61 GEE used 406 observations from 59 patients (p=0.041764) or 350 observations from 50 complete survivors (p=0.019358). The latter rounds to the same p-value as the revised primary GEE. The sample and death-exclusion rules distinguish the analyses. Revision cell 93 provides adjusted-model inputs. Its provisional untransformed-time model reported p=0.019321 and was superseded by revision cell 94 using `log1p` ICU-to-MB hours.

## Reproduction by deliverable

| Output | What the repository provides | What remains to verify |
|---|---|---|
| HiRID cohort and phenotype derivation | Raw archive extraction, final item map, infusion and antibiotic linkage, BP coverage, original rate cleaning, weight normalization, GMM exploration, and the prespecified rules and construct validation. | Rerun against the original authorized archives and reconcile time-zero and phenotype counts. |
| MIMIC-IV cohort, Table 1 and S2 | Raw prescription and pharmacy extraction, dose and ICU linkage, diagnosis and norepinephrine filters, original BP/lab/vasoactive/ventilation/RRT logic, Table 1 and BP-quality summaries. | Confirm the MIMIC-IV release and any preprocessing of the original local CSV collection. Then compare aggregate values, missingness and unit-conversion QC. |
| Table 2 and secondary AUC | Original complete-case selection, NEE summary and rank-test code. The table footnote now reads the revised primary GEE result. | Confirm numeric equivalence after extraction on the original inputs. |
| Primary model and S3–S7 | Original death-filtered longitudinal model, time counts, death QC, marginal means with back-transformed limits, full-covariance pairwise contrasts with Holm adjustment within interval, and all three final sensitivity specifications. S7 benchmark rows and figure annotations now use fitted values instead of hard-coded results. | Confirm counts, coefficients, model convergence, confidence limits, and contrast p-values in a credentialed end-to-end run. |
| Figures 1–2 | The cell 69 study-design renderer and the cell 15 phenotype scatterplot. Figure 1 text was aligned with the final two-variable rule. | Final graphical layout, export cropping and typography were not verified pixel-for-pixel. Figure 2 source reproduces the scatterplot design and cutoffs. |
| Figure 3 | The revision cell 85 renderer using fitted GEE quantities and the original AUC plot membership. | Panel B uses 59 nonmissing displayed values (17, 11, 31) but shows the p-value from 50 AUC-eligible complete cases. The submitted PNG and saved code agree on the displayed denominators. Decide whether the publication figure should retain that display with a population label or restrict the plot to the inferential cohort. The release leaves the plotted membership unchanged. |
| Supplementary Table S8 | New, descriptive CSV code using the 59-patient frozen MIMIC-IV cohort and the original mortality and LOS variables. | Generate the actual descriptive values with the credentialed cohort. ICU mortality is in-index-stay mortality by recorded death timestamp. ICU LOS is the full index stay. |
| STROBE and typeset appendix | Code regenerates statistical source tables and definitions. | The checklist and manuscript typesetting are editorial artifacts. They are outside the analytical regeneration scripts. |

## Execution changes and data availability

- Replaced Colab mounts and absolute paths with separate configured source roots and a private work directory. Resolved historical filenames to their explicitly identified upstream stages. The run timestamp and success markers support retrying a failed stage in the same work directory. The workflow uses the original extracted timestamps and study time windows.
- Consolidated duplicate ventilation passes into the final device/mode classifier while retaining the required event extraction. Retained the older ICD block solely because it produces the diagnosis input used after corrected index characterization. Removed exploratory VNERi calculations and the competing-risk draft outside the revised manuscript. Retained the prespecified physiological comparisons and the GMM work described by the manuscript.
- The controlled database archives, MIMIC-IV extracts and published-analysis patient-level intermediates were unavailable for this review. None are needed in the repository itself. Where the original workstation run cannot be recreated, the decisive inputs for rerunning inference are the frozen 59-patient cohort, hourly NEE, and death/follow-up extracts listed in the README. The original MIMIC-IV release and source checksums remain unknown. The exact Colab Patsy version was also unrecorded.
- The run verification script reads generated aggregate checks or locally aggregates the two cohorts. It compares counts exactly and p-values to three decimals. Archived evidence and synthetic execution cannot establish numeric agreement on clinical data.

## Checks completed for this package

The pinned Python environment installed successfully and passed dependency checks. Python source compilation and undefined-name checks passed. Tests on synthetic inputs ran Table 1, BP quality, AUC, the death-corrected GEE and Figure 3, all three sensitivity models, Table 2, S8, Figure 1 and the environment report. Separate synthetic tests exercised the consolidated ICD extraction and the final ventilation mode classifier. The benchmark command correctly flagged the synthetic counts and p-values as different from the manuscript. These tests verify that the selected stages can execute and exchange files. Complete raw-to-results runs on HiRID and MIMIC-IV still require the authorized database snapshots.
