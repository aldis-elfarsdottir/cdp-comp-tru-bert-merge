import os

import polars as pl


def write_csv(df: pl.DataFrame, path: str, makedirs: bool = True):
    # Create necessary directories
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    df.write_csv(path)


def write_parquet(df: pl.DataFrame, path: str, makedirs: bool = True):
    # Create necessary directories
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df.write_parquet(path, compression='zstd', compression_level=1, statistics=False, use_pyarrow=False)


def read_csv(path: str, *args, **kwargs) -> pl.DataFrame:
    return pl.scan_csv(path, *args, **kwargs).collect()


def read_excel(path: str) -> dict[str, pl.DataFrame]:
    return pl.read_excel(path, sheet_id=0, raise_if_empty=False, engine='calamine')


def read_parquet(path: str) -> pl.DataFrame:
    return pl.scan_parquet(path).collect()