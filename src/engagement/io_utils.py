from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def read_table_with_fallback(
    parquet_path: Path,
    *,
    default_columns: list[str] | None = None,
) -> pd.DataFrame:
    csv_path = parquet_path.with_suffix(".csv")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except (ImportError, ModuleNotFoundError, ValueError):
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if default_columns is None:
        return pd.DataFrame()
    return pd.DataFrame(columns=default_columns)


def write_table_with_fallback(
    df: pd.DataFrame,
    parquet_path: Path,
    *,
    logger: logging.Logger | None = None,
) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:  # noqa: BLE001
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        if logger is not None:
            logger.warning(
                "Parquet write failed (%s: %s). Wrote CSV fallback: %s",
                type(exc).__name__,
                exc,
                csv_path,
            )
        return csv_path
