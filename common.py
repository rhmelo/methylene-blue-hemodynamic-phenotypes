"""Paths and console output shared by the ported notebook stages."""

import os
from pathlib import Path


def configured_path(variable):
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f"Set {variable} or provide the corresponding run.py option.")
    return Path(value).expanduser().resolve()


def phase_dir(name):
    return configured_path("MB_WORK_DIR") / name


def hirid_source():
    return configured_path("HIRID_SOURCE_DIR")


def mimic_source():
    return configured_path("MIMIC_IV_SOURCE_DIR")


def run_stamp():
    return os.environ["MB_RUN_STAMP"]


def source_file(root, filename):
    """Resolve an exact raw filename in a flat or nested database layout."""
    direct = root / filename
    if direct.is_file():
        return direct
    return artifact(root, filename)


def artifact(root, patterns):
    """Resolve a unique input, in priority order when given multiple patterns."""
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        matches = sorted(path for path in root.rglob(pattern) if path.is_file())
        if len(matches) == 1:
            return matches[0]
        if matches:
            matching_paths = "\n".join(str(path) for path in matches)
            raise RuntimeError(
                f"Ambiguous input {pattern!r} under {root}:\n{matching_paths}"
            )
    raise FileNotFoundError(f"No matching input under {root}: {', '.join(patterns)}")


def display(value):
    """Replace IPython display with a private console preview."""
    print(value)
