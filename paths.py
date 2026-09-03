"""Repository-wide filesystem paths.

Every script resolves its input CSVs through :func:`data_path` so that all runs
read the same physical files, regardless of the working directory the job was
launched from. Previously each stage directory carried its own copy of ``Data/``
and scripts disagreed about whether ``"Data/..."`` was relative to the current
working directory or to the script file, so two stages of the same experiment
could silently read different copies.

Usage from any script under ``<repo>/<Family>/<stage>/``::

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paths import data_path

    data_processor = DataProcessor(data_file_path=data_path("classification_NIR_Data_raw_VFA_TA.csv"), ...)

Set the ``NIR_DATA_DIR`` environment variable to read the dataset from
somewhere other than ``<repo>/Data`` (for example a cluster scratch filesystem)
without editing any script.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("NIR_DATA_DIR", REPO_ROOT / "Data")).resolve()

CALIBRATION_FILE = "calibration_NIR_Data.csv"


def data_path(filename):
    """Return the absolute path to ``filename`` inside the dataset directory.

    Args:
        filename (str): Basename of a CSV in the dataset directory, such as
            ``"classification_NIR_Data_raw_VFA_TA.csv"``.

    Returns:
        pathlib.Path: Absolute path to the requested file.

    Raises:
        FileNotFoundError: If the dataset directory or the file is missing, with
            a message naming the directory that was searched.
    """
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}. "
            "Expected <repo>/Data; set NIR_DATA_DIR to point elsewhere."
        )

    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def calibration_path():
    """Return the absolute path to the absorbance calibration CSV."""
    return data_path(CALIBRATION_FILE)
