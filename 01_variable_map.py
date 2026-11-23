"""Source: original notebook cell 7. Run through run.py."""

from common import phase_dir

# ============================================================
# STEP 01C — FINAL MANUAL CORRECTION OF VARIABLE IDs
# ============================================================

import json

PROJECT_DIR = phase_dir("hirid")

LOG_DIR = PROJECT_DIR / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

final_id_map = {
    # Vasopressors
    "norepinephrine_ids": [
        1000462,  # Noradrenalin 1 mg/ml
        1000656,  # Noradrenalin 100 µg/ml Perfusor
        1000657,  # Noradrenalin 20 µg/ml Perfusor
        1000658,  # Noradrenalin 10 µg/ml Bolus
    ],
    # Systemic antibiotic candidates
    # Route will still be checked in pharma_records.
    "antibiotic_ids": [
        181,  # Metronidazole infusion
        1000233,  # Penicillin
        1000274,  # Augmentin injection
        1000284,  # Penicillin G
        1000549,  # Augmentin injection
        1000812,  # Augmentin infusion
        1000825,  # Cefepime
        1000837,  # Piperacillin-tazobactam
        1001084,  # Meropenem 500 mg
        1001086,  # Meropenem 1 g
        1001168,  # Clindamycin phosphate 300
        1001169,  # Clindamycin phosphate 600
        1001173,  # Rifampicin infusion
    ],
    # Blood lactate only
    "lactate_ids": [
        24000485,
        24000524,
        24000732,
    ],
    # Invasive arterial pressure only
    "sbp_ids": [100],
    "dbp_ids": [120],
    "map_ids": [110],
    "heart_rate_ids": [200],
    # Blood creatinine only
    "creatinine_ids": [20000600],
    "urine_output_ids": [10020000],
}

final_map_path = LOG_DIR / "step01c_final_variable_id_map.json"

with open(
    final_map_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        final_id_map,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("=" * 80)
print("FINAL VARIABLE ID MAP")
print("=" * 80)

print(
    json.dumps(
        final_id_map,
        indent=2,
        ensure_ascii=False,
    )
)

print("\nSaved to:\n" f"{final_map_path}")
