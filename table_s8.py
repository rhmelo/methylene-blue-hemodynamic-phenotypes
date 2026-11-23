"""Generate the new descriptive Supplementary Table S8 from the frozen cohort."""

import pandas as pd

from common import artifact, phase_dir

PHENOTYPES = (
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
    "Preserved diastolic pressure",
)
OUTCOMES = (
    ("Hospital mortality", "hospital_mortality", "binary"),
    ("Index ICU mortality", "icu_mortality", "binary"),
    ("Hospital length of stay, days", "hospital_los_days", "continuous"),
    ("Index ICU length of stay, days", "icu_los_days_calculated", "continuous"),
)
KEYS = ["subject_id", "stay_id"]
PHENOTYPE_COLUMN = "hemodynamic_phenotype"


def describe_outcomes(cohort):
    needed = {*KEYS, PHENOTYPE_COLUMN, *(column for _, column, _ in OUTCOMES)}
    missing = sorted(needed - set(cohort.columns))
    if missing:
        raise KeyError(f"Frozen cohort is missing S8 source columns: {missing}")
    if (
        cohort.empty
        or cohort[KEYS].isna().any().any()
        or cohort["subject_id"].duplicated().any()
    ):
        raise ValueError(
            "S8 requires one index ICU stay for each unique cohort patient."
        )
    if not cohort[PHENOTYPE_COLUMN].isin(PHENOTYPES).all():
        raise ValueError("S8 requires the classifiable MIMIC-IV cohort.")
    groups = [("Overall", cohort)] + [
        (label, cohort.loc[cohort[PHENOTYPE_COLUMN].eq(label)]) for label in PHENOTYPES
    ]

    rows = []
    for label, column, kind in OUTCOMES:
        for group, frame in groups:
            values = pd.to_numeric(frame[column], errors="raise").dropna()
            row = {
                "outcome": label,
                "group": group,
                "n_total": len(frame),
                "n_available": len(values),
                "n_missing": len(frame) - len(values),
            }
            if kind == "binary":
                if not values.isin([0, 1]).all():
                    raise ValueError(f"{column} must contain 0, 1 or missing values.")
                events = int(values.sum())
                row.update(
                    events=events,
                    percent=100 * events / len(values) if len(values) else None,
                )
            else:
                quartiles = values.quantile([0.25, 0.50, 0.75])
                row.update(
                    q1=quartiles.loc[0.25],
                    median=quartiles.loc[0.50],
                    q3=quartiles.loc[0.75],
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    root = phase_dir("mimic_iv")
    source = artifact(root, "METHYLENE_BLUE_ANALYTIC_COHORT_FROZEN_FINAL_*.csv.gz")
    cohort = pd.read_csv(source)
    summary = describe_outcomes(cohort)
    out = root / "tables" / "Table_S8_descriptive_outcomes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
