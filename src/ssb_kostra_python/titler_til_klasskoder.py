# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: kostra-fellesfunksjoner
#     language: python
#     name: kostra-fellesfunksjoner
# ---

# %%

from typing import Literal

import pandas as pd
from klass import KlassClassification

# %%
"""Fest navn/tittel til klassifikasjonskoder basert på KLASS.

Denne modulen lar deg knytte lesbare navn (tittel) til koder i klassifikasjonsvariabler
ved hjelp av KLASS. Du velger hvilke klassifikasjonsvariabler som skal få navn
koplet til verdiene i datasettet, og kjører funksjonen på det aktuelle datasettet.

Oppsett av mapping::

    mapping_klassifikasjonsvariable = [
        {"code_col": "kommuneregion", "klass_id": 231},
        {"code_col": "funksjon",       "klass_id": 277},
        {"code_col": "avtaleform",     "klass_id": 252},
    ]

Kjøring::

    df_aug, diag = mapping_hierarki.kodelister_navn(
        df_befolkningsdata,
        mappings=mapping_klassifikasjonsvariable,
        language="nb",
        include_future=True,
        verbose=True,
    )
    display(df_aug)

Merk:

- Dersom du kjører regionshierarki-aggregering på et datasett etter at du har festet
  navn på klassifikasjonsvariablene, kan det bli inkonsistens. Aggregeringsfunksjonen
  aggregerer koder, men ikke navnene.

- Fjern i så fall navnekolonnene før aggregering. Etter aggregering kan du legge dem til igjen.
"""
# ---------- internals ----------


from typing import Any


def _pick_level_columns(
    pivot_df: pd.DataFrame, level: int | None
) -> tuple[int, str, str]:
    code_cols = [c for c in pivot_df.columns if str(c).startswith("code_")]
    name_cols = [c for c in pivot_df.columns if str(c).startswith("name_")]
    if not code_cols or not name_cols:
        raise RuntimeError(
            "KLASS mapping is missing expected 'code_*'/'name_*' columns."
        )

    if level is None:
        # choose the smallest available level number
        def _lvl(c: str) -> int:
            try:
                return int(str(c).split("_", 1)[1])
            except Exception:
                return 10**9  # put non-conforming columns at the end

        level = min(_lvl(c) for c in code_cols)

    mcode = f"code_{level}"
    mname = f"name_{level}"
    if mcode not in pivot_df.columns or mname not in pivot_df.columns:
        raise RuntimeError(
            f"Expected columns '{mcode}' and '{mname}' not found in mapping."
        )
    return level, mcode, mname


def _fetch_mapping_for_year(
    klass_id: int,
    year: int,
    *,
    language: Literal["nb", "nn", "en"] = "nb",
    include_future: bool = True,
    select_level: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Return a 2-col DF: ['_map_code','_map_name'] and the level used."""
    from_date = f"{year}-01-01"
    to_date = f"{year}-12-31"

    klass = KlassClassification(
        int(klass_id), language=language, include_future=include_future
    )
    codes = klass.get_codes(
        from_date=from_date,
        to_date=to_date,
        language=language,
        include_future=include_future,
        select_level=select_level,
    )
    pivot = codes.pivot_level()
    level, mcode, mname = _pick_level_columns(pivot, select_level)

    mapping = pivot[[mcode, mname]].rename(
        columns={mcode: "_map_code", mname: "_map_name"}
    )
    # No zero-padding per requirement; compare as plain strings
    mapping["_map_code"] = mapping["_map_code"].astype(str).str.strip()
    mapping["_map_name"] = mapping["_map_name"].astype(str)
    return mapping, level


def _attach_one_mapping(
    df_in: pd.DataFrame,
    *,
    year: int,
    code_col: str,
    klass_id: int,
    name_col_out: str | None = None,
    language: Literal["nb", "nn", "en"] = "nb",
    include_future: bool = True,
    select_level: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach names for one (code_col, klass_id) pair; returns new df and diagnostics."""
    if code_col not in df_in.columns:
        raise ValueError(f"Column '{code_col}' not found in DataFrame.")

    mapping, level_used = _fetch_mapping_for_year(
        klass_id=klass_id,
        year=year,
        language=language,
        include_future=include_future,
        select_level=select_level,
    )

    df = df_in.copy()
    df[code_col] = df[code_col].astype(str).str.strip()
    merged = df.merge(mapping, how="left", left_on=code_col, right_on="_map_code")

    if name_col_out is None:
        name_col_out = f"{code_col}_navn"

    # Insert the name column immediately after the code column
    insert_at_raw = merged.columns.get_loc(code_col)
    if not isinstance(insert_at_raw, int):
        raise TypeError(
            f"Expected int from get_loc, got {type(insert_at_raw)}: {insert_at_raw}"
        )
    insert_at = insert_at_raw
    merged.insert(insert_at + 1, name_col_out, merged["_map_name"])

    # Drop helper columns
    merged = merged.drop(columns=["_map_code", "_map_name"])

    # --- validation: data codes NOT present in mapping ---
    data_codes = set(df[code_col].dropna().astype(str).str.strip())
    map_codes = set(mapping["_map_code"])
    invalid_in_data = sorted(data_codes - map_codes)

    diagnostics = {
        "code_col": code_col,
        "klass_id": int(klass_id),
        "level": level_used,
        "year": int(year),
        "invalid_count": len(invalid_in_data),
        "invalid_sample": invalid_in_data[:20],
        "all_invalid": invalid_in_data,  # keep full list in case you need it
    }
    return merged, diagnostics


# ---------- public API ----------


# def attach_multiple_classification_names(
def kodelister_navn(
    df: pd.DataFrame,
    mappings: list[dict[str, Any]],
    *,
    language: Literal["nb", "nn", "en"] = "nb",
    include_future: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply multiple (code_col, klass_id) mappings for the year in ``df['periode']``.

    Args:
        df: Must contain ``'periode'`` with exactly one unique year.
        mappings: List of dictionaries. Each dict has the following keys::
            {
                "code_col": "kommunenr",          # required
                "klass_id": 131,                    # required
                "name_col_out": "kommunenr_navn", # optional; default <code_col>_navn
                "select_level": 1,                  # optional
            }
        language: Language code passed to KLASS. {"nb", "nn", "en"}, default "nb".
        include_future: Whether to include future codes in KLASS. default True
        verbose: Whether to print diagnostic messages. default True

    Returns:
        A tuple containing: ``df_out``: Original DF with each name column inserted right after its code column.
            ``diag``: Per-pair diagnostics keyed by ``code_col`` (or ``code_col|klass_id`` if duplicates).

    Raises:
        ValueError: If ``'periode'`` is missing or contains multiple unique years.
    """
    # Validate 'periode' once
    if "periode" not in df.columns:
        raise ValueError("DataFrame must contain a 'periode' column (year).")
    unique_years = pd.Series(df["periode"]).dropna().unique()
    if len(unique_years) != 1:
        raise ValueError(
            f"'periode' must have exactly one unique value; found {len(unique_years)}: {unique_years!r}"
        )
    year = int(unique_years[0])

    out = df.copy()
    diagnostics: dict[str, Any] = {}

    for item in mappings:
        code_col = item["code_col"]
        klass_id = item["klass_id"]
        name_col_out = item.get("name_col_out")
        select_level = item.get("select_level")

        out, diag = _attach_one_mapping(
            out,
            year=year,
            code_col=code_col,
            klass_id=klass_id,
            name_col_out=name_col_out,
            language=language,
            include_future=include_future,
            select_level=select_level,
        )

        key = code_col if code_col not in diagnostics else f"{code_col}|{klass_id}"
        diagnostics[key] = diag

        if verbose:
            msg = (
                f"[{code_col}] klass_id={klass_id}, level={diag['level']}, year={year} — "
                f"invalid data codes: {diag['invalid_count']}"
            )
            if diag["invalid_count"]:
                sample = ", ".join(diag["invalid_sample"])
                extra = (
                    ""
                    if diag["invalid_count"] <= 20
                    else f" …(+{diag['invalid_count']-20} more)"
                )
                msg += f" | sample: {sample}{extra}"
            print(msg)

    return out, diagnostics
