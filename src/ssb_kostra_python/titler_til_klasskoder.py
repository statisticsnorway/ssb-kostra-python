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

import pandas as pd
from klass import KlassClassification

# %%
"""
Med denne funksjonen kan du feste navn/tittel til klassifikasjonsvariablenes koder hvis de svarer til en kodeliste i KLASS.

Du velger ut de klassifikasjonsvariablene i datasettet ditt som skal få kodeNAVN fra KLASS koplet på kodeVERDIEN i datasettet.
Deretter kjører du funksjonen på datasettet som skal behandles.

Slik skriver du koden:

Du starter med mappingen, altså angir du variabelen i datasettet ditt og klass_id i KLASS. Du kan nøye deg med ett par, eller du kan ha flere. Men merk deg formatet.
Her er det snakk om en liste som inneholder dictionary (én eller flere).

mapping_klassifikasjonsvariable = [
    {"code_col": "kommuneregion", "klass_id": 231}, <--- regionsvariabelen i datasettet heter "kommuneregion" og den er koplet til klass-id '231' (https://www.ssb.no/klass/klassifikasjoner/231)
    {"code_col": "funksjon",  "klass_id": 277},     <--- en annen klassifikasjonsvariabel i datasettet er "funksjon" og den er koplet til klass-id '277' (https://www.ssb.no/klass/klassifikasjoner/277)
    {"code_col": "avtaleform", "klass_id": 252},    <--- regionsvariabelen i datasettet heter "avtaleform" og den er koplet til klass-id '252' (https://www.ssb.no/klass/klassifikasjoner/252)
]

Under kjører du funksjonen med de instillingene du bestemte i "mapping_klassifikasjonsvariable". Hovedfunksjonen som skal kjøres heter "kodelister_navn".
De øvrige funksjonene du ser under brukes i hovedfunksjonen.

df_aug, diag = mapping_hierarki.kodelister_navn(    <--- funksjonen spytter ut to resultater: "df_aug" er det nye datasettet med navn på klassifikasjonsvariablene. "diag" (diagnose) gir beskjed om datasettet inneholder verdier for klassifikasjonsvariabelen som ikke finnes i kodelisten.
    df_befolkningsdata,                             <--- dette er datasettet som skal behandles, i dette eksempelet heter det "df_befolkningsdata".
    mappings=mapping_klassifikasjonsvariable,       <--- dette er mappingen du definerte i forkant. Husk at det er nødvendig å lage mappingen så funksjonen vet hvor den skal lete.
    language="nb",
    include_future=True,
    verbose=True,   # prints a short line per mapping with invalid-code count
)

display(df_aug)                                                         <--- med denne koden kan du se det endelige resultatet.

OBS!!
Om du kjører regionshierarkiaggregering (du finner denne funksjonen lenger opp) på et datasett etter at du har festet navn på klassifikasjonsvariablene dine, vil det gå i ball.
Det er fordi aggregeringsfunksjonen aggregerer regionskodene med en hardkodet mapping, men ikke navnene.

Så om du har gjort dette før en nødvendig aggregering, så bør du fjerne disse kolonnene i forkant, for så å utføre aggregeringen. Etter aggregeringen kan legge dem til igjen.

"""
# ---------- internals ----------


def _pick_level_columns(pivot_df: pd.DataFrame, level: int | None):
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
    language: str = "nb",
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
    language: str = "nb",
    include_future: bool = True,
    select_level: int | None = None,
) -> tuple[pd.DataFrame, dict]:
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
    insert_at = merged.columns.get_loc(code_col) + 1
    merged.insert(insert_at, name_col_out, merged["_map_name"])

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
    mappings: list[dict],
    *,
    language: str = "nb",
    include_future: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Apply multiple (code_col, klass_id) mappings to a DF for the same year from df['periode'].

    Args:
        df: Must contain 'periode' with exactly one unique year.
        mappings: List of dictionaries. Each dict:
            {
                "code_col": "kommunenr",     # required
                "klass_id": 131,              # required
                "name_col_out": "kommunenr_navn",   # optional; default <code_col>_navn
                "select_level": 1,            # optional
            }
        language: Language code passed to KLASS.
        include_future: Whether to include future codes in KLASS.
        verbose: Whether to print diagnostic messages.

    Returns:
        tuple: A tuple containing:
            - df_out (pd.DataFrame): Original DF with each name column inserted right after its code column.
            - diag (dict): Per-pair diagnostics keyed by code_col (or code_col|klass_id if duplicates).

    Raises:
        ValueError: If 'periode' is missing or contains multiple unique years.
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
    diagnostics: dict = {}

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
