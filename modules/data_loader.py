"""
Dataset loading for TaskAware.

Supports CSV and XLSX files. Returns a pandas DataFrame
after basic readability checks.
"""

import pandas as pd


ALLOWED_EXTENSIONS = {
    "csv",
    "xlsx",
}


def allowed_file(filename):
    """
    Check whether the uploaded file is CSV or XLSX.
    """
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def load_dataset(filepath):
    """
    Load a CSV or XLSX dataset into a pandas DataFrame.

    Raises ValueError for unsupported extensions.
    """
    extension = filepath.rsplit(".", 1)[1].lower()

    if extension == "csv":
        return pd.read_csv(filepath)

    if extension == "xlsx":
        return pd.read_excel(filepath)

    raise ValueError(
        "Unsupported file type. Only CSV and XLSX "
        "are supported."
    )