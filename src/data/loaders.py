"""
NHANES Data Loading Utilities

Functions to load XPT files from downloaded NHANES data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import pyreadstat
import yaml


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_config() -> dict:
    """Load the project configuration."""
    config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_file_mappings() -> dict:
    """Load file mappings configuration."""
    mappings_path = get_project_root() / "config" / "file_mappings.yaml"
    with open(mappings_path) as f:
        return yaml.safe_load(f)


def load_xpt_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single XPT file using pyreadstat.

    Parameters
    ----------
    filepath : str or Path
        Path to the XPT file.

    Returns
    -------
    pd.DataFrame
        Loaded data. (SEQN is preserved as a regular column.)
    """
    df, meta = pyreadstat.read_xport(str(filepath))
    return df


def get_file_path(year: str, file_id: str, raw_dir: Optional[Path] = None) -> Path:
    """
    Get the full path to a NHANES data file.

    Parameters
    ----------
    year : str
        Survey year (e.g., "2015-2016").
    file_id : str
        File identifier (e.g., "DEMO", "BMX").
    raw_dir : Path, optional
        Path to raw data directory. Defaults to data/raw.

    Returns
    -------
    Path
        Full path to the XPT file.
    """
    if raw_dir is None:
        raw_dir = get_project_root() / "data" / "raw"

    mappings = load_file_mappings()
    suffix = mappings["year_suffixes"].get(year, "")

    # Check for year-specific overrides
    overrides = mappings.get("year_overrides", {}).get(year, {})
    if file_id in overrides:
        filename = overrides[file_id]
        if filename is None:
            raise ValueError(f"File {file_id} not available for year {year}")
    else:
        filename = f"{file_id}{suffix}"

    return raw_dir / year / f"{filename}.xpt"


def load_file_for_year(year: str, file_id: str, raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load a specific NHANES file for a given year.

    Parameters
    ----------
    year : str
        Survey year (e.g., "2015-2016").
    file_id : str
        File identifier (e.g., "DEMO", "BMX").
    raw_dir : Path, optional
        Path to raw data directory.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    filepath = get_file_path(year, file_id, raw_dir)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return load_xpt_file(filepath)


def load_all_files_for_year(
    year: str,
    raw_dir: Optional[Path] = None,
    merge: bool = True
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load all NHANES files for a given year.

    Parameters
    ----------
    year : str
        Survey year (e.g., "2015-2016").
    raw_dir : Path, optional
        Path to raw data directory.
    merge : bool, default True
        If True, merge all files on SEQN. If False, return dict of DataFrames.

    Returns
    -------
    pd.DataFrame or dict
        Merged DataFrame or dict of DataFrames by file_id.
    """
    if raw_dir is None:
        raw_dir = get_project_root() / "data" / "raw"

    mappings = load_file_mappings()
    files_config = mappings["files"]

    dataframes = {}

    # Load each file category
    for category, file_list in files_config.items():
        for file_info in file_list:
            file_id = file_info["id"]
            try:
                df = load_file_for_year(year, file_id, raw_dir)
                dataframes[file_id] = df
            except (FileNotFoundError, ValueError) as e:
                print(f"  Warning: Could not load {file_id} for {year}: {e}")

    if not merge:
        return dataframes

    # Merge all dataframes on SEQN
    if not dataframes:
        raise ValueError(f"No files loaded for year {year}")

    # Start with DEMO as base (contains all participants)
    if "DEMO" not in dataframes:
        raise ValueError("DEMO file required but not found")

    merged = dataframes["DEMO"].copy()

    for file_id, df in dataframes.items():
        if file_id == "DEMO":
            continue
        # Left join to keep all DEMO participants
        merged = merged.merge(df, on="SEQN", how="left", suffixes=("", f"_{file_id}"))

    return merged


def load_multiple_years(
    years: List[str],
    raw_dir: Optional[Path] = None,
    add_year_column: bool = True
) -> pd.DataFrame:
    """
    Load and combine data from multiple survey years.

    Parameters
    ----------
    years : list of str
        Survey years to load (e.g., ["2015-2016", "2017-2018"]).
    raw_dir : Path, optional
        Path to raw data directory.
    add_year_column : bool, default True
        If True, add a column indicating the survey year.

    Returns
    -------
    pd.DataFrame
        Combined data from all years.
    """
    all_data = []

    for year in years:
        print(f"Loading data for {year}...")
        df = load_all_files_for_year(year, raw_dir, merge=True)
        if add_year_column:
            df["SURVEY_YEAR"] = year
        all_data.append(df)
        print(f"  Loaded {len(df):,} participants")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal combined: {len(combined):,} participants")

    return combined


def get_variable_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary information about all variables in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary with dtype, non-null count, null count, unique values, etc.
    """
    info = []
    for col in df.columns:
        info.append({
            "variable": col,
            "dtype": str(df[col].dtype),
            "non_null": df[col].notna().sum(),
            "null_count": df[col].isna().sum(),
            "null_pct": round(df[col].isna().sum() / len(df) * 100, 2),
            "unique": df[col].nunique(),
            "sample_values": str(df[col].dropna().head(3).tolist())[:50]
        })
    return pd.DataFrame(info)


def get_file_summary(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Get a summary of all downloaded files.

    Parameters
    ----------
    raw_dir : Path, optional
        Path to raw data directory.

    Returns
    -------
    pd.DataFrame
        Summary of files with year, file_id, rows, columns.
    """
    if raw_dir is None:
        raw_dir = get_project_root() / "data" / "raw"

    config = load_config()
    years = config["data"]["initial_years"]

    summary = []
    for year in years:
        year_dir = raw_dir / year
        if not year_dir.exists():
            continue

        for xpt_file in year_dir.glob("*.xpt"):
            try:
                df = load_xpt_file(xpt_file)
                summary.append({
                    "year": year,
                    "file": xpt_file.stem,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "size_mb": round(xpt_file.stat().st_size / 1024 / 1024, 2)
                })
            except Exception as e:
                summary.append({
                    "year": year,
                    "file": xpt_file.stem,
                    "rows": None,
                    "columns": None,
                    "size_mb": round(xpt_file.stat().st_size / 1024 / 1024, 2),
                    "error": str(e)
                })

    return pd.DataFrame(summary)
