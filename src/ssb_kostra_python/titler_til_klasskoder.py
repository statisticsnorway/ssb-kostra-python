import re
from typing import Any
from typing import Literal

import pandas as pd
from klass import KlassClassification

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
    select_level: int,
) -> tuple[pd.DataFrame, int]:
    """Return a 2-col DF: ['_map_code','_map_name'] and the level used."""
    from_date = f"{year}-01-01"
    to_date = f"{year}-12-31"

    klass = KlassClassification(
        str(klass_id), language=language, include_future=include_future
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
    select_level: int,
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


def kodelister_navn(
    df: pd.DataFrame,
    mappings: list[dict[str, Any]],
    *,
    language: Literal["nb", "nn", "en"] = "nb",
    include_future: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fester kodenavn til klassifikasjonsvariabler ved hjelp av KLASS.

    Funksjonen brukes på et datasett som gjelder for ett enkelt år.
    Den legger til nye kolonner med kodenavn for valgte
    klassifikasjonsvariabler i henhold til tilhørende KLASS-kodelister.

    Datasettet må inneholde en periodevariabel kalt ``periode`` med
    nøyaktig én unik verdi. Datasettet kan altså ikke inneholde flere
    årganger samtidig.

    For hver klassifikasjonsvariabel som skal få kodenavn, må det oppgis
    informasjon om:

    - hvilken kolonne som inneholder kodeverdiene
    - hvilken KLASS-kodeliste som skal brukes
    - hvilket navn den nye kolonnen med kodenavn skal ha
    - eventuelt hvilket nivå som skal velges

    Denne informasjonen gis gjennom argumentet ``mappings``.

    Hvis ``name_col_out`` ikke oppgis, blir navnet på den nye kolonnen
    automatisk satt til ``<code_col>_navn``.

    Eksempel
    --------
    Anta at datasettet inneholder klassifikasjonsvariablene ``periode``,
    ``bydelsregion`` og ``alder``, samt statistikkvariabelen ``personer``.

    Vi ønsker å feste kodenavn til ``bydelsregion`` og ``alder``.
    Bydeler er knyttet til KLASS-liste 241, og alder er knyttet til
    KLASS-liste 248.

    Først lager vi mappingen::

        mapping_klassifikasjonsvariable = [
            {
                "code_col": "bydelsregion",
                "klass_id": 241,
                "name_col_out": "bydelsregion_navn",
                "select_level": 1,
            },
            {
                "code_col": "alder",
                "klass_id": 248,
                "name_col_out": "alder_navn",
                "select_level": 1,
            },
        ]

    Deretter kjøres funksjonen slik::

        df_med_kodenavn, sammendrag = (
            titler_til_klasskoder.kodelister_navn(
                df_uten_kodenavn,
                mappings=mapping_klassifikasjonsvariable,
                language="nb",
                include_future=True,
                verbose=True,
            )
        )

        display(df_med_kodenavn)

    Funksjonen returnerer to objekter:

    - ``df_med_kodenavn`` er datasettet med nye kolonner for kodenavn.
    - ``sammendrag`` inneholder diagnostisk informasjon om mappingene.

    Parameters
    ----------
    df : pd.DataFrame
        Datasettet som skal få kodenavn lagt til.

        Datasettet må inneholde kolonnen ``periode`` med nøyaktig én
        unik årgang.

    mappings : list[dict[str, Any]]
        Liste med dictionaries som beskriver hvilke
        klassifikasjonsvariabler som skal kobles mot KLASS.

        Hver dictionary kan inneholde følgende nøkler:

        - ``code_col``:
          Navnet på kolonnen som inneholder kodeverdiene.
          Påkrevd.

        - ``klass_id``:
          ID-en til KLASS-kodelisten som skal brukes.
          Påkrevd.

        - ``name_col_out``:
          Navnet på den nye kolonnen med kodenavn.
          Valgfritt. Dersom den ikke oppgis, brukes
          ``<code_col>_navn``.

        - ``select_level``:
          Nivå som skal velges fra KLASS.
          Valgfritt.

    language : {"nb", "nn", "en"}, default "nb"
        Språket som skal brukes ved oppslag i KLASS.

        ``"nb"`` betyr bokmål, ``"nn"`` betyr nynorsk og ``"en"``
        betyr engelsk.

    include_future : bool, default True
        Angir om framtidige koder skal inkluderes i oppslaget mot KLASS.

    verbose : bool, default True
        Angir om funksjonen skal skrive ut diagnostiske meldinger under
        kjøringen.

    Returns:
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        En tuple som inneholder:

        - ``df_out``:
          Det opprinnelige datasettet med nye navnekolonner lagt inn
          rett etter tilhørende kodekolonne.

        - ``diag``:
          Diagnostisk informasjon per mapping, indeksert etter
          ``code_col`` eller ``code_col|klass_id`` dersom det finnes
          duplikater.

    Raises:
    ------
    ValueError
        Hvis kolonnen ``periode`` mangler, eller dersom datasettet
        inneholder mer enn én unik årgang.
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
        if select_level is None:
            raise ValueError("Undefined select_level.")

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


KLASS_IDS = {
    "kommuneregion": "231",
    "fylkesregion": "232",
    "bydelsregion": "241",
}
TOKENS = {"nan", "<na>", "none", "nul", "null", "na", "n/a", ""}
ZFILLS = {"kommuneregion": 4, "fylkesregion": 4, "bydelsregion": 6}


def mapping_regionsnavn(
    inputfil: pd.DataFrame,
    *,
    language: Literal["nb", "nn", "en"] = "nb",
    region_col: str | None = None,  # allow explicit override; else auto-detect
    name_suffix: str = "_navn",
) -> pd.DataFrame:
    """Denne funksjonen kan du bruke til å feste regionsnavn på regionskodene dine.

    Så lenge regionsvariabelen heter "bydelsregion", "kommuneregion" eller "fylkesregion", vil funksjonen feste
    riktig regionsnavn fra KLASS for det året datasettet gjelder. Dersom regionsvariabelen heter noe annet enn dette, må den omdøpes til riktig regionsnivå.

    Funksjonen henter regionsnavn fra følgende KLASS-kodelister:

    "kommuneregion": 231
    "fylkesregion": 232
    "bydelsregion": 241

    Denne funksjonen bør du bruke ETTER at du har utført hierarkiaggregeringen, og IKKE før.
    Grunnen til dette er at hierarkiaggregeringsfunksjonen fungerer til å aggregere regionskodene, men ikke regionsnavnene.
    Om du trenger å se regionsnavn både før og etter regionsaggregering, kan du da feste regionsnavn i første omgang, fjerne dem i forkant av regionsaggregeringen, og feste dem igjen etter at regionsaggregeringen er utført.

    Slik bruker du funksjonen:

    df_regionsnavn = titler_til_klasskoder.mapping_regionsnavn(df_uten_regionsnavn)

    Datasettet til venstre for likhetstegnet er datasettet som genereres med regionsnavn tilhørende regionskoden. I parentesen ligger datasettet du ønsker å føre regionsnavn på.
    """
    if "periode" not in inputfil.columns:
        raise ValueError("Column 'periode' is required in inputfil.")
    # 1) region column: explicit or auto-detect (must be exactly one)
    regionsvariable = ["kommuneregion", "fylkesregion", "bydelsregion"]
    if region_col is None:
        present = [c for c in regionsvariable if c in inputfil.columns]
        if not present:
            raise ValueError(
                "No region column found (expected one of kommuneregion/fylkesregion/bydelsregion)."
            )
        if len(present) > 1:
            raise ValueError(
                f"Multiple region columns present: {present}. Please specify region_col=..."
            )
        region_col = present[0]
    elif region_col not in inputfil.columns:
        raise ValueError(f"Specified region_col '{region_col}' not in dataframe.")

    # 2) determine a single valid year from 'periode'
    s = inputfil["periode"].astype("string")
    uniq = pd.Series(s.unique())
    valid_years: list[str] = []
    for v in uniq:
        if pd.isna(v):
            continue  # type: ignore[unreachable]
        sv = str(v).strip()  # type: ignore[unreachable]
        if not sv:
            continue
        core = sv.lstrip("0").lower()
        if core in TOKENS:
            continue
        if re.fullmatch(r"\d{4}", sv):
            valid_years.append(sv)
    valid_years = sorted(set(valid_years))
    if len(valid_years) != 1:
        raise ValueError(
            f"Need exactly one valid 4-digit 'periode'; found {valid_years or 'none'}."
        )
    year = valid_years[0]

    # 3) fetch KLASS mapping table for (region_col, year)
    klass_id = KLASS_IDS[region_col]
    klass = KlassClassification(klass_id, language=language, include_future=True)
    codes = klass.get_codes(from_date=f"{year}-01-01", to_date=f"{year}-12-31")

    mapping = codes.pivot_level()
    try:
        map_code_col = next(c for c in mapping.columns if c.lower().startswith("code"))
        map_name_col = next(c for c in mapping.columns if c.lower().startswith("name"))
    except StopIteration as err:
        raise ValueError(
            "Mapping must have columns starting with 'code' and 'name' (e.g. 'code_1', 'name_1')."
        ) from err

    # 4) normalize both sides (strings + trim)
    out = inputfil.copy()
    out[region_col] = out[region_col].astype("string").str.strip()
    mapping[map_code_col] = mapping[map_code_col].astype("string").str.strip()
    mapping[map_name_col] = mapping[map_name_col].astype("string").str.strip()

    # --- CONDITIONAL padding: only if digit-only AND length < target width ---
    width = ZFILLS[region_col]

    # Left side (data)
    mask_left_digits_short = out[region_col].str.fullmatch(r"\d+") & (
        out[region_col].str.len() < width
    )
    out[region_col] = out[region_col].where(
        ~mask_left_digits_short, other=out[region_col].str.zfill(width)
    )

    # Right side (mapping)
    mask_right_digits_short = mapping[map_code_col].str.fullmatch(r"\d+") & (
        mapping[map_code_col].str.len() < width
    )
    mapping[map_code_col] = mapping[map_code_col].where(
        ~mask_right_digits_short, other=mapping[map_code_col].str.zfill(width)
    )
    # ------------------------------------------------------------------------

    # ensure one name per code in mapping
    mapping = mapping.drop_duplicates(subset=[map_code_col])

    merged = out.merge(
        mapping[[map_code_col, map_name_col]],
        how="left",
        left_on=region_col,
        right_on=map_code_col,
        validate="m:1",
        suffixes=("", "_map"),
    )

    new_name_col = f"{region_col}{name_suffix}"
    merged.rename(columns={map_name_col: new_name_col}, inplace=True)
    if map_code_col != region_col:
        merged.drop(columns=[map_code_col], inplace=True)

    return merged
