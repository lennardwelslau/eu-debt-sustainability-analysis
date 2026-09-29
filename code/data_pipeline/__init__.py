"""
Data pipeline for the DSA model: builds, reads and documents the Excel input workbook.
"""
from .schema import EU27, PARAMETERS, SERIES, REPO_ROOT, INPUT_DIR, RAW_DIR, resolve_input_path, latest_input_file
from .workbook import write_workbook, read_workbook, read_country, read_shocks, set_overrides, apply_overrides
from .legacy import convert_legacy_csv
