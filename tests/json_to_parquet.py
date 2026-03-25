from pathlib import Path

import pandas as pd


def json_to_parquet(
    json_path: str | Path,
    parquet_path: str | Path,
    orient: str = "records",
    lines: bool | None = None,
) -> None:
    """Convert a JSON file to Parquet.

    Parameters
    ----------
    json_path : str | Path
        Input JSON file.
    parquet_path : str | Path
        Output Parquet file.
    orient : str
        JSON orientation (default: "records").
    lines : bool | None
        Set to True for NDJSON (one JSON object per line).
        If None, inferred from file suffix (.jsonl → True).
    """
    json_path = Path(json_path)
    # parquet_path = Path(parquet_path)

    if lines is None:
        lines = json_path.suffix in {".jsonl", ".ndjson"}

    df = pd.read_json(json_path, orient=orient, lines=lines)

    df.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    json_to_parquet(
        "mapping_aldershierarki.json",
        "gs://ssb-dapla-felles-data-produkt-prod/kostra/eksempeldata/mapping_aldershierarki.parquet",
    )
