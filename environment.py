"""Source: original notebook cell 74. Run through run.py."""

from common import display, phase_dir

import platform
import importlib.metadata as metadata
import pandas as pd

packages = [
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "matplotlib",
    "python-docx",
    "pyarrow",
    "patsy",
    "joblib",
]

rows = [
    {"component": "Python", "version": platform.python_version()},
    {"component": "Platform", "version": platform.platform()},
]

for package in packages:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "Not installed"

    rows.append({"component": package, "version": version})

environment = pd.DataFrame(rows)

display(environment)

output_dir = phase_dir("mimic_iv") / "qc"
output_dir.mkdir(parents=True, exist_ok=True)
environment.to_csv(output_dir / "software_environment.csv", index=False)
