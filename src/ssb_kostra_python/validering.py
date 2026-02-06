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
import re
from typing import Any

import ipywidgets as widgets
import pandas as pd
from fagfunksjoner import logger
from functions.funksjoner.hjelpefunksjoner import definere_klassifikasjonsvariable
from IPython.display import display  # for nice tables in notebooks
from klass import KlassClassification

SEPARATOR = "-" * 230  # or whatever length you need


# %%
# def show_toggle(df, mask, title, *, preview_rows: int = 15):
def show_toggle(
    df: pd.DataFrame,
    mask: pd.Series,
    title: str,
    *,
    preview_rows: int = 15,
) -> None:
    """Renders a compact toggle to reveal offending rows on demand.

    - Always shows a title
    - Shows a ToggleButton if there are rows; clicking displays a preview table
    """
    count = int(mask.sum())
    header = widgets.HTML(
        f"<b>{title}</b> &nbsp; <span style='color:#666'>({count} rows)</span>"
    )
    if count == 0:
        # Nothing to show — just the header
        display(header)
        return

    btn = widgets.ToggleButton(description="Show/hide rows", icon="table", value=False)
    out = widgets.Output()

    def _on_toggle(change: dict[str, Any]) -> None:
        if change["name"] == "value":
            out.clear_output()
            if btn.value:
                with out:
                    display(df.loc[mask].head(preview_rows))

    btn.observe(_on_toggle, names="value")
    display(widgets.VBox([header, btn, out]))


# %%
def _missing_cols(inputfil: pd.DataFrame, klassifikasjonsvariable: list[str]):
    """Denne funksjonen sjekker om datasettet inneholder de klassifikasjonsvariablene som brukeren angir. Funksjonen inngår i valideringen."""
    # Check for missing columns
    missing_cols = [c for c in klassifikasjonsvariable if c not in inputfil.columns]
    logger.info(f"Checking if the {klassifikasjonsvariable} columns are present...")
    if missing_cols:
        logger.error(f"❌ Missing required column(s): {missing_cols}!\n")
    else:
        logger.info(
            f"✅ No missing columns. All {klassifikasjonsvariable} are present.\n"
        )


# %%
def _missing_values(
    df: pd.DataFrame,
    klassifikasjonsvariable: list[str],
    preview_rows: int = 10,
    zeros_valid_for: set[str] | None = None,
) -> None:
    """Sjekker om klassifikasjonsvariablene i datasettet mangler verdier.

    Denne funksjonen sjekker om klassifikasjonsvariablene i datasettet mangler verdier (koder). Om koder mangler, vil en feilmelding skrives ut. Inngår i valideringen.
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    Check that klassifikasjonsvariable columns have no missing values.

    Missing includes:
      - native NA (pd.NA/NaN/None)
      - empty/whitespace-only strings
      - common NA tokens (nan, <na>, none, nul, null, na, n/a)
      - *padded-missing* like '0nan', '000<NA>', '000null' (leading zeros + NA token)

    BUT: if a column is in `zeros_valid_for`, strings that are *only zeros* ('0', '00', '000', '0000', ...)
    are treated as **valid**, not missing.

    Parameters
    ----------
    zeros_valid_for : set[str] | None
        Columns for which all-zero strings should NOT be treated as missing.
        Example: {"noekkelkode"}  # where '000' is a valid code
    """
    logger.info("ℹ️ Checking for missing values in klassifikasjonsvariable...\n")

    TOKENS = {"nan", "<na>", "none", "nul", "null", "na", "n/a", ""}

    zeros_valid_for = set(zeros_valid_for or [])

    missing_cols = [c for c in klassifikasjonsvariable if c not in df.columns]
    if missing_cols:
        logger.error(
            f"❌ Columns not found (cannot check for missing): {missing_cols}\n"
        )

    any_issues = False

    for col in (c for c in klassifikasjonsvariable if c in df.columns):
        s = df[col].astype("string")

        # native NA
        mask_native = s.isna()

        # normalized strings
        s_norm = s.fillna("").str.strip()

        # empty/whitespace
        mask_empty = s_norm.eq("")

        # detect all-zero strings like '0', '00', '000', ...
        mask_all_zeros = s_norm.str.fullmatch(r"0+")  # True for ONLY zeros

        # padded-missing heuristic: strip leading zeros then compare to NA tokens
        core = s_norm.str.lstrip("0").str.lower()
        mask_token_base = core.isin(TOKENS)

        # If this column allows zero codes, don't flag all-zero strings as missing
        if col in zeros_valid_for:
            mask_token = mask_token_base & ~mask_all_zeros
        else:
            mask_token = mask_token_base

        # combined missing mask
        mask_missing = mask_native | mask_empty | mask_token

        if mask_missing.any():
            any_issues = True
            count = int(mask_missing.sum())
            logger.error(f"❌ Missing values detected in '{col}' ({count} rows).\n")
            # display(df.loc[mask_missing].head(preview_rows))
            show_toggle(
                df,
                mask_missing,
                f"Missing values in '{col}' — click to preview",
                preview_rows=preview_rows,
            )

    if not any_issues and not missing_cols:
        logger.info(
            f"✅ No missing values in any of the klassifikasjonsvariable {klassifikasjonsvariable}.\n"
        )


# %%
def _valid_periode_region(
    df: pd.DataFrame, klassifikasjonsvariable: list[str], preview_rows: int = 10
):
    """Denne funksjonen sjekker om verdiene (kodene) til periode- og regionsvariabelen er på riktig format. Inngår i valideringen."""
    logger.info("ℹ️ Checking if periode and region are in the valid format...\n")

    PADDED_TOKENS = {"nan", "<na>", "none", "nul", "null", "na", ""}

    for col in klassifikasjonsvariable:
        # ----- PERIODE: must be 4 digits -----
        if col == "periode":
            s = df[col].astype("string")
            mask_missing = s.isna() | s.str.strip().eq("")
            s_norm = s.fillna("").str.strip()
            core = s_norm.str.lstrip("0").str.lower()
            mask_padded_missing = ~mask_missing & core.isin(PADDED_TOKENS)

            # only true-format errors (exclude missing + padded-missing)
            mask_fmt_bad = (
                ~mask_missing & ~mask_padded_missing & ~s_norm.str.fullmatch(r"\d{4}")
            )

            if mask_padded_missing.any():
                logger.warning(
                    f"⚠️ Suspected zero-padded missing in '{col}'. Defer to missing-values check.\n"
                )
                # display(df.loc[mask_padded_missing].head(preview_rows))
                show_toggle(
                    df,
                    mask_padded_missing,
                    "Padded-missing 'periode' — click to preview",
                    preview_rows=15,
                )

            if mask_fmt_bad.any():
                logger.error(f"❌ Check: {col} is not four digits.\n")
                # display(df.loc[mask_fmt_bad].head(preview_rows))
                show_toggle(
                    df,
                    mask_fmt_bad,
                    "Format-invalid 'periode' — click to preview",
                    preview_rows=15,
                )
            elif not mask_padded_missing.any():
                logger.info("✅ 'periode' is formatted correctly.\n")

        # ----- KOMMUNE/FYLKESREGION: digits-only must be exactly 4; non-digits allowed -----
        if col in ["kommuneregion", "fylkesregion"]:
            s = df[col].astype("string")
            mask_missing = s.isna() | s.str.strip().eq("")
            s_norm = s.fillna("").str.strip()
            core = s_norm.str.lstrip("0").str.lower()
            mask_padded_missing = ~mask_missing & core.isin(PADDED_TOKENS)

            mask_numeric = (
                ~mask_missing & ~mask_padded_missing & s_norm.str.fullmatch(r"\d+")
            )
            mask_fmt_bad = mask_numeric & (s_norm.str.len() != 4)

            if mask_padded_missing.any():
                logger.warning(
                    f"⚠️ Suspected zero-padded missing in '{col}'. Defer to missing-values check.\n"
                )
                # display(df.loc[mask_padded_missing].head(preview_rows))
                show_toggle(
                    df,
                    mask_padded_missing,
                    f"Padded-missing '{col}' — click to preview",
                    preview_rows=15,
                )

            if mask_fmt_bad.any():
                logger.error(f"❌ Check: {col} is not four digits.\n")
                # display(df.loc[mask_fmt_bad].head(preview_rows))
                show_toggle(
                    df,
                    mask_fmt_bad,
                    f"Format-invalid '{col}' — click to preview",
                    preview_rows=15,
                )
            elif not mask_padded_missing.any():
                if col == "kommuneregion":
                    logger.info("✅ 'kommuneregion' is formatted correctly.\n")
                else:
                    logger.info("✅ 'fylkesregion' is formatted correctly.\n")

        # ----- BYDELSREGION: digits-only must be 6 and in 030101-039999 -----
        if col == "bydelsregion":
            s = df[col].astype("string")
            mask_missing = s.isna() | s.str.strip().eq("")
            s_norm = s.fillna("").str.strip()
            core = s_norm.str.lstrip("0").str.lower()
            mask_padded_missing = ~mask_missing & core.isin(PADDED_TOKENS)

            mask_numeric = (
                ~mask_missing & ~mask_padded_missing & s_norm.str.fullmatch(r"\d+")
            )
            # valid if 6 digits and integer between 30101 and 39999 (leading 0 kept in string)
            numeric_vals = pd.to_numeric(s_norm.where(mask_numeric), errors="coerce")
            mask_range_ok = (
                mask_numeric
                & (s_norm.str.len() == 6)
                & numeric_vals.between(30101, 39999)
            )

            mask_fmt_bad = mask_numeric & ~mask_range_ok

            if mask_padded_missing.any():
                logger.warning(
                    f"⚠️ Suspected zero-padded missing in '{col}'. Defer to missing-values check.\n"
                )
                # display(df.loc[mask_padded_missing].head(preview_rows))
                show_toggle(
                    df,
                    mask_padded_missing,
                    "Padded-missing 'bydelsregion' — click to preview",
                    preview_rows=15,
                )

            if mask_fmt_bad.any():
                logger.error(
                    f"❌ Column '{col}' must be 6-digit numeric in 030101-039999.\n"
                )
                # display(df.loc[mask_fmt_bad].head(preview_rows))
                show_toggle(
                    df,
                    mask_fmt_bad,
                    "Format-invalid 'bydelsregion' — click to preview",
                    preview_rows=15,
                )
            elif not mask_padded_missing.any():
                logger.info("✅ 'bydelsregion' is formatted correctly.\n")


# %%
def _number_of_periods_in_df(inputfil: pd.DataFrame, preview_rows: int = 10):
    """Sjekker antall perioder i datasettet. Inngår i valideringen.

    Sjekker hvor mange forskjellige perioder datasettet inneholder. Helt konkret sjekker den hvor mange forskjellige unike verdier som finnes i periode-kolonnen.
    Om kolonnen inneholder meningsløse verdier, som for eksempel 202P, eller 5025, vil sjekken markere disse som egne verdier og kategorisere feilen de antas å tilhøre.
    """
    logger.info("ℹ️ Inspecting distinct 'periode' values...\n")

    TOKENS = {"nan", "<na>", "none", "nul", "null", "na", "n/a", ""}

    s = inputfil["periode"].astype("string")
    uniq = pd.Series(s.unique())

    valid, padded_missing, true_missing, fmt_invalid = [], [], [], []

    for v in uniq:
        if pd.isna(v) or (isinstance(v, str) and v.strip() == ""):
            true_missing.append(v)
            continue
        sv = str(v).strip()
        core = sv.lstrip("0").lower()
        if core in TOKENS:
            padded_missing.append(v)
            continue
        if re.fullmatch(r"\d{4}", sv):
            valid.append(sv)
        else:
            fmt_invalid.append(v)

    logger.info(f"ℹ️ Found {len(uniq)} distinct 'periode' value(s).\n")
    if valid:
        logger.info(f"✅ Valid periods ({len(valid)}): {sorted(valid)}\n")

    # Build masks (without mutating df)
    mask_padded = s.isin(padded_missing)
    mask_missing = s.isna() | s.fillna("").str.strip().eq("")
    mask_fmt = s.isin(fmt_invalid)

    # 1) Padded-missing → warn + display rows
    if mask_padded.any():
        logger.warning(
            f"⚠️ Suspected padded-missing periods ({int(mask_padded.sum())} rows) → see results from the missing-values check.\n"
        )
        # display(df.loc[mask_padded].head(preview_rows))
        show_toggle(
            inputfil,
            mask_padded,
            "Suspected padded-missing periods — click to preview",
            preview_rows=preview_rows,
        )

    # 2) True missing → error + display rows
    if mask_missing.any():
        logger.error(
            f"❌ Missing 'periode' values ({int(mask_missing.sum())} rows) → handled by the missing-values check.\n"
        )
        # display(df.loc[mask_missing].head(preview_rows))
        show_toggle(
            inputfil,
            mask_missing,
            "Missing 'periode' values — click to preview",
            preview_rows=preview_rows,
        )

    # 3) Format-invalid → error + display rows
    if mask_fmt.any():
        logger.error(
            f"❌ Format-invalid 'periode' tokens ({int(mask_fmt.sum())} rows) → handled by the format check.\n"
        )
        show_toggle(
            inputfil,
            mask_fmt,
            "Format-invalid 'periode' tokens — click to preview",
            preview_rows=preview_rows,
        )

    if not (mask_padded.any() or mask_missing.any() or mask_fmt.any()):
        logger.info("✅ All distinct 'periode' values look valid.\n")

    print(valid)
    return valid


# %%
# assumes `show_toggle(...)` exists and `logger` is configured
# assumes KlassClassification is available in scope


def _klass_check(
    df: pd.DataFrame,
    klassifikasjonsvariable: list[str],
    preview_rows: int = 15,
    interactive: bool = True,
):
    """Sjekker at klassifikasjonsvariablene i datasettet har koder som finnes i KLASS for det aktuelle året.

    Denne funksjonen sjekker om kodene til klassifikasjonsvariablene i datasettet er gyldige og i tråd med KLASS-kodelisten for det aktuelle året. Feilmelding skrives ut dersom
    sjekken finner koder som ikke finnes i kodelisten for det året. Funksjonen vil automatisk sjekke den regionsvariabelen den finner i datasettet mot riktig KLASS-liste.
    Datasettet inneholder gjerne flere klassifikasjonsvariable en periode- og regionsvariabelen. Funksjonen sørger for å spørre om kodelistenummeret i KLASS til denne eller disse variablene.
    For hver slik variabel må du skrive inn nummeret. Du kan også hoppe over steget ved å trykke "Enter", men da blir ikke variabelens koder sjekket mot KLASS.

    Denne funksjonen inngår i valideringen, men du kan også kjøre den separat. Den trenger å vite datasettet som skal sjekkes og klassifikasjonsvariablene som skal sjekkes.
    Om klassifikasjonsvariablene ikke er forhåndsdefinert, kan du skrive koden slik:

    _klass_check(df_datasett, ['periode', 'kommuneregion', 'funksjon', 'avtaleform'])

    eller slik, hvor du først definerer en liste du kan kalle f. eks. "klassifikasjonsvariable":

    klassifikasjonsvariable = ['periode', 'kommuneregion', 'funksjon', 'avtaleform']
    _klass_check(df_datasett, klassifikasjonsvariable)

    I dette eksemplet er viser variabelen 'funksjon' til "Kodeliste for KOSTRA regnskapsfunksjoner" med lenke "https://www.ssb.no/klass/klassifikasjoner/277". I promptet må brukeren angi
    277 i tekstfeltet. Variabelen 'avtaleform' viser til "Kodeliste for avtaleform kostra" med lenke "https://www.ssb.no/klass/klassifikasjoner/252". I det neste promptet må brukeren angi
    252 for den neste variabelen.

    """
    logger.info("ℹ️ Inspecting your data before performing KLASS check...\n")

    # ---------- Phase A: determine the single valid periode (if any) ----------
    periode = None
    antall_perioder = 0

    if "periode" in klassifikasjonsvariable and "periode" in df.columns:
        s = df["periode"].astype("string")
        uniq = pd.Series(s.unique())

        TOKENS = {"nan", "<na>", "none", "nul", "null", "na", "n/a", ""}

        def _is_valid_year(v: Any) -> bool:
            if pd.isna(v):
                return False
            sv = str(v).strip()
            if sv == "":
                return False
            core = sv.lstrip("0").lower()
            if core in TOKENS:
                return False
            return bool(re.fullmatch(r"\d{4}", sv))

        valid_years = sorted([str(v).strip() for v in uniq if _is_valid_year(v)])
        antall_perioder = len(valid_years)
        if antall_perioder == 1:
            periode = valid_years[0]
            logger.info(
                f"ℹ️ Checking if all region values are found in KLASS for the year {periode}...\n"
            )
        elif antall_perioder == 0:
            logger.warning(
                "⚠️ No valid 4-digit 'periode' detected; KLASS check skipped.\n"
            )
        else:
            logger.warning(
                f"⚠️ Your data set contains {antall_perioder} valid periods {valid_years}; \n"
                "KLASS check runs only when exactly one period is present."
            )
    else:
        logger.warning("⚠️ Column 'periode' not provided; KLASS check skipped.\n")

    # If we don't have exactly one periode, stop here
    if antall_perioder != 1 or not periode:
        print("-" * 190)
        print("-" * 190 + "\n")
        return

    # ---------- Phase B: build the active mapping (defaults + prompted extras) ----------
    defaults = {
        "kommuneregion": 231,
        "fylkesregion": 232,
        "bydelsregion": 241,
    }

    # figure out which requested columns are not already mapped
    known_keys = {"periode", *defaults.keys()}
    unknown_cols = [
        c for c in klassifikasjonsvariable if c not in known_keys and c in df.columns
    ]

    # Helper: prompt for a KLASS id (empty input => skip). Validate integer and verify via quick fetch.
    def _prompt_klass_id(colname: str) -> int | None:
        if not interactive:
            logger.warning(
                f"⚠️ No mapping provided for '{colname}' (non-interactive run). Skipping.\n"
            )
            return None
        tries = 3
        while tries > 0:
            try:
                raw = input(
                    f"Enter KLASS ID for '{colname}' (press Enter to skip): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                logger.warning(f"⚠️ Input not available; skipping '{colname}'.\n")
                return None
            if raw == "":
                logger.warning(
                    f"⚠️ User skipped '{colname}'; KLASS check disabled for this column.\n"
                )
                return None
            if not raw.isdigit():
                logger.warning(
                    "⚠️ Invalid input; please enter a numeric KLASS ID or press Enter to skip.\n"
                )
                tries -= 1
                continue
            klass_id = int(raw)
            # quick verify: attempt to fetch codes for this (klass_id, periode)
            try:
                test = KlassClassification(klass_id, language="en", include_future=True)
                codes = test.get_codes(
                    from_date=f"{periode}-01-01", to_date=f"{periode}-12-31"
                )
                # some clients expose .data, others return a df; handle both
                df_codes = getattr(codes, "data", codes)
                if df_codes is None or len(df_codes) == 0:
                    logger.warning(
                        f"⚠️ KLASS ID {klass_id} returned no codes for {periode}. Try again or press Enter to skip.\n"
                    )
                    tries -= 1
                    continue
                logger.info(
                    f"ℹ️ Verified KLASS ID {klass_id} for '{colname}' and {periode}.\n"
                )
                return klass_id
            except Exception as e:
                logger.warning(
                    f"⚠️ KLASS lookup failed for ID {klass_id}: {e}. Try again or press Enter to skip.\n"
                )
                tries -= 1
        logger.warning(
            f"⚠️ Giving up on '{colname}' after multiple attempts; skipping.\n"
        )
        return None

    # Collect user-provided mappings (no persistence)
    extra_map: dict[str, int] = {}
    if unknown_cols:
        logger.info(f"ℹ️ Prompting for KLASS IDs: {unknown_cols}\n")
        for ucol in unknown_cols:
            klass_id = _prompt_klass_id(ucol)
            if klass_id is not None:
                extra_map[ucol] = klass_id

    # Merge active mapping
    klass_ids = {**defaults, **extra_map}

    # ---------- Phase C: validate each mapped column against KLASS ----------
    # simple in-run cache to avoid refetching the same (klass_id, periode)
    _cache: dict[tuple[int, str], list[str]] = {}

    def _get_klass_codes(klass_id: int, year: str) -> list[str]:
        key = (klass_id, year)
        if key in _cache:
            return _cache[key]
        k = KlassClassification(klass_id, language="en", include_future=True)
        result = k.get_codes(from_date=f"{year}-01-01", to_date=f"{year}-12-31")
        df_codes = getattr(result, "data", result)
        codes = (
            df_codes[["code"]]
            .assign(code=lambda s: s["code"].astype(str).str.strip())["code"]
            .tolist()
        )
        _cache[key] = codes
        return codes

    for col in klassifikasjonsvariable:
        if col == "periode":
            continue
        if col not in klass_ids:
            # unmapped column (either skipped by user or not prompted due to absence)
            logger.info(
                f"ℹ️ No KLASS mapping for '{col}'; skipping KLASS validation for this column.\n"
            )
            continue
        if col not in df.columns:
            logger.warning(f"⚠️ Column '{col}' not found in data; skipping.\n")
            continue

        series = df[col].astype("string").dropna().str.strip()
        if series.empty:
            logger.warning(f"⚠️ Column '{col}' has no non-missing values; skipping.\n")
            continue
        dataset_codes = series.tolist()

        klass_id = klass_ids[col]
        try:
            klass_codes = _get_klass_codes(klass_id, periode)
        except Exception as e:
            logger.error(
                f"❌ KLASS lookup failed for '{col}' (ID={klass_id}) and periode {periode}: {e}\n"
            )
            continue

        missing = sorted(set(dataset_codes) - set(klass_codes))
        if missing:
            logger.error(
                f"❌ Column '{col}' contains codes not present in KLASS for {periode} \n"
                f"({len(missing)} distinct code(s))."
            )
            mask_invalid = df[col].astype("string").str.strip().isin(missing)
            show_toggle(
                df,
                mask_invalid,
                f"Invalid code(s) for classification '{col}'  — click to preview",
                preview_rows=preview_rows,
            )
        else:
            logger.info(f"✅ All '{col}' codes are present in KLASS for {periode}.\n")


# %%
def validering(
    inputfil: pd.DataFrame, klassifikasjonsvariable: list | None = None
) -> None:
    """Alle enkeltsjekkene samlet i én funksjon. Denne funksjonen er den som skal kjøres for å validere datasettet ditt.

    Denne funksjonen kjører flere sjekker etter hverandre. Funksjonen trenger å vite filen som skal undersøkes (inputfil) og klassifikasjonsvariablene
    som inngår i datasettet (klassifikasjonsvariable). Om du ikke har definert klassifikasjonsvariablene i tidligere i løpet, kan du skrive koden for eksempel slik:

    klassifikasjonsvariable = ['periode', 'bydelsregion', 'helsekontroller']
    mapping_hierarki.validering(navn_på_din_fil, klassifikasjonsvariable)

    """
    if klassifikasjonsvariable == []:
        logger.info("Genererer klassifikasjonsvariable fra inputfilen...")
        klassifikasjonsvariable, _ = definere_klassifikasjonsvariable(inputfil)
    else:
        logger.info("Klassifikasjonsvariable allerede definert.")
    _missing_cols(inputfil, klassifikasjonsvariable)
    print("\n" + SEPARATOR)
    # _missing_values(inputfil, klassifikasjonsvariable)
    _missing_values(
        inputfil, klassifikasjonsvariable, zeros_valid_for=set(klassifikasjonsvariable)
    )
    print("\n" + SEPARATOR)
    _valid_periode_region(inputfil, klassifikasjonsvariable)
    print("\n" + SEPARATOR)
    _number_of_periods_in_df(inputfil)
    print("\n" + SEPARATOR)
    _klass_check(inputfil, klassifikasjonsvariable)
    print("\n" + SEPARATOR)
