"""Run HiRID, MIMIC-IV and analysis stages in a private work directory."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parent

HIRID = tuple(sorted((REPO / "hirid").glob("[0-9]*.py")))
MIMIC_EXTRACTION = tuple(sorted((REPO / "mimic_iv").glob("[0-9]*.py")))
ANALYSES = tuple(
    REPO / "analyses" / f"{name}.py"
    for name in (
        "bp_quality",
        "auc",
        "primary_gee_and_figure_3",
        "sensitivity",
        "table_2",
        "table_s8",
        "figure_1",
        "environment",
    )
)
MIMIC = (
    MIMIC_EXTRACTION[:12] + (REPO / "analyses" / "table_1.py",) + MIMIC_EXTRACTION[12:]
)
STAGES = {path.stem: (path,) for path in HIRID + MIMIC + ANALYSES}
GROUPS = {
    "hirid": HIRID,
    "mimic": MIMIC,
    "analyses": ANALYSES,
    "all": HIRID + MIMIC + ANALYSES,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("list", *GROUPS, *STAGES))
    parser.add_argument(
        "--work-dir", type=Path, help="Private workspace for all generated outputs"
    )
    parser.add_argument("--hirid-dir", type=Path, help="HiRID raw archive directory")
    parser.add_argument(
        "--mimic-dir", type=Path, help="MIMIC-IV directory containing hosp/ and icu/"
    )
    args = parser.parse_args()

    if args.target == "list":
        for group in ("hirid", "mimic", "analyses"):
            print(f"{group}:")
            for path in GROUPS[group]:
                print(f"  {path.stem}")
        return

    settings = {
        "MB_WORK_DIR": args.work_dir,
        "HIRID_SOURCE_DIR": args.hirid_dir,
        "MIMIC_IV_SOURCE_DIR": args.mimic_dir,
    }
    env = os.environ.copy()
    for key, option in settings.items():
        if option:
            env[key] = str(option.expanduser().resolve())
    if not env.get("MB_WORK_DIR"):
        parser.error("provide --work-dir or set MB_WORK_DIR")
    work_dir = Path(env["MB_WORK_DIR"]).expanduser().resolve()
    env["MB_WORK_DIR"] = str(work_dir)
    if work_dir.is_relative_to(REPO) or REPO.is_relative_to(work_dir):
        parser.error(
            "choose a work directory outside the repository tree and its parents"
        )
    for key in ("HIRID_SOURCE_DIR", "MIMIC_IV_SOURCE_DIR"):
        if env.get(key):
            source_dir = Path(env[key]).expanduser().resolve()
            env[key] = str(source_dir)
            if work_dir.is_relative_to(source_dir) or source_dir.is_relative_to(
                work_dir
            ):
                parser.error(f"keep the work directory separate from {key}")

    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO), env.get("PYTHONPATH")])
    )
    os.umask(0o077)
    work_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = work_dir / "run_stamp.txt"
    if not stamp_path.exists():
        stamp_path.write_text(
            datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "\n"
        )
    env["MB_RUN_STAMP"] = stamp_path.read_text().strip()

    targets = GROUPS[args.target] if args.target in GROUPS else STAGES[args.target]
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for path in targets:
        # Keep full QC logs local. The notebook diagnostics can include patient rows.
        log_path = log_dir / f"{path.stem}.log"
        done_path = log_path.with_suffix(".ok")
        if done_path.exists():
            print(f"Completed previously {path.relative_to(REPO)}", flush=True)
            continue
        print(f"Running {path.relative_to(REPO)}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-u", str(path)],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            parser.exit(
                result.returncode, f"{path.stem} failed. See private log {log_path}.\n"
            )
        done_path.touch()
    print(f"Completed {args.target}. Outputs are under {work_dir}.")


if __name__ == "__main__":
    main()
