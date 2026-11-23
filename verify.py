"""Compare a completed private run with the aggregate manuscript benchmarks."""

import argparse
from pathlib import Path
import sys

import pandas as pd

from common import artifact

GEE_TARGETS = {
    "Primary GEE": (59, 387, 0.019),
    "Invasive-only GEE": (53, 349, 0.002),
    "Complete 6-h follow-up GEE": (51, 357, 0.010),
    "Adjusted post-MB GEE": (58, 322, 0.018),
}


def compare(work_dir):
    hirid = work_dir / "hirid"
    mimic = work_dir / "mimic_iv"
    revision = mimic / "revision_analyses"
    results = []

    phenotypes = pd.read_parquet(
        hirid
        / "derived_data"
        / "step07_analysis_dataset_with_prespecified_physiological_phenotypes.parquet",
        columns=["patientid", "physiological_phenotype"],
    )
    counts = phenotypes.groupby("physiological_phenotype")["patientid"].nunique()
    results.extend(
        [
            ("HiRID cohort", phenotypes["patientid"].nunique(), 1050),
            ("HiRID vasoplegia-compatible", counts.get(1, 0), 355),
            ("HiRID mixed/low-pulsatility", counts.get(2, 0), 215),
            ("HiRID preserved DBP", counts.get(3, 0), 480),
        ]
    )

    cohort = pd.read_csv(
        artifact(mimic, "METHYLENE_BLUE_ANALYTIC_COHORT_FROZEN_FINAL_*.csv.gz")
    )
    results.append(("MIMIC-IV cohort", cohort["subject_id"].nunique(), 59))
    auc = pd.read_csv(
        artifact(mimic, "methylene_blue_code13C_AUC_mixed_model_qc_*.csv")
    )
    auc = auc.set_index("metric")["value"]
    results.extend(
        [
            (
                "Complete-case AUC cohort",
                int(auc["Patients eligible for complete 6 h AUC"]),
                50,
            ),
            (
                "Baseline-adjusted AUC P",
                round(auc["Global p value for baseline-adjusted AUC"], 3),
                0.691,
            ),
        ]
    )

    model_summary = pd.read_csv(
        revision / "robustness_analyses_FINAL_CORRECTED.csv"
    ).set_index("analysis")
    for model, (patients, observations, p_value) in GEE_TARGETS.items():
        row = model_summary.loc[model]
        results.extend(
            [
                (f"{model} patients", int(row["n_patients"]), patients),
                (f"{model} observations", int(row["n_observations"]), observations),
                (
                    f"{model} interaction P",
                    round(row["phenotype_time_interaction_p"], 3),
                    p_value,
                ),
            ]
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        results = compare(args.work_dir.expanduser().resolve())
    except (FileNotFoundError, KeyError, RuntimeError) as error:
        parser.exit(2, f"Benchmark comparison incomplete. {error}\n")

    matched = True
    for label, observed, expected in results:
        ok = observed == expected
        matched &= ok
        status = "PASS" if ok else "DIFFER"
        print(f"{status:<7} {label}: observed {observed}, expected {expected}")
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
