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
import logging

import numpy as np
import pandas as pd
from IPython.display import display  # for nice tables in notebooks

logger = logging.getLogger(__name__)


# %%
def _round_half_up(values: pd.Series, decimals: int = 0) -> pd.Series:
    """Kommersiell avrunding (0.5 -> 1, -0.5 -> -1), også for desimaler.

    Fungerer på numpy-arrays eller pandas-Serier av tall.
    """
    factor = 10**decimals
    # Cast to float to avoid issues with Int64 etc.
    v = pd.Series(pd.to_numeric(values, errors="coerce").astype(float))
    arr = v.to_numpy(dtype=float)
    # round half away from zero
    rounded = np.sign(arr) * np.floor(np.abs(arr) * factor + 0.5) / factor

    return pd.Series(rounded, index=v.index, name=v.name)


def print_instruks_konverter_dtypes() -> str:
    """Lager instruks for å lage mapping."""
    instruks = """ℹ️Bruk malen under for dtype_mapping. Du må angi denne mappingen i forkant for at funksjonen skal kunne konvertere variablene slik du ønsker.

    dtype_mapping = {
        "klassifikasjonsvariabel":  ["var1", "var2"]         ℹ️Legg inn variablene du vil klassifikasjonsverdier
        "heltall":                  ["var3", "var4"],        ℹ️Legg inn variablene du vil runde av til heltall (kommersiell avrunding)
        "desimaltall_1_des":        ["var5", "var6"],        ℹ️Legg inn variablene du vil runde til 1 desimal
        "desimaltall_2_des":        ["var7", "var8"],        ℹ️Legg inn variablene du vil runde til 2 desimaler
        "stringvar":                ["var9", "var10"],       ℹ️Legg inn variablene du vil konvertere til tekst
        "bool_var":                 ["var11", "var11"],      ℹ️Legg inn variablene du vil konvertere til boolske verdier
    }

    NB:
    ℹ️ Variabler som ikke legges inn her, blir IKKE endret.
    ℹ️ Hvis du angir en variabel som ikke finnes i dataframen, får du en advarsel.
    ℹ️ Du kan la lister stå tomme hvis ingen variabler skal konverteres i en gitt gruppe."""

    print(instruks)
    return instruks


def konverter_dtypes(
    df: pd.DataFrame, dtype_mapping: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.Series]:
    """ℹ️Bruk malen under for dtype_mapping. Du må angi denne mappingen i forkant for at funksjonen skal kunne konvertere variablene slik du ønsker.

    dtype_mapping = {
        "klassifikasjonsvariabel":  ["var1", "var2"]         ℹ️Legg inn variablene du vil klassifikasjonsverdier
        "heltall":                  ["var3", "var4"],        ℹ️Legg inn variablene du vil runde av til heltall (kommersiell avrunding)
        "desimaltall_1_des":        ["var5", "var6"],        ℹ️Legg inn variablene du vil runde til 1 desimal
        "desimaltall_2_des":        ["var7", "var8"],        ℹ️Legg inn variablene du vil runde til 2 desimaler
        "stringvar":                ["var9", "var10"],       ℹ️Legg inn variablene du vil konvertere til tekst
        "bool_var":                 ["var11", "var12"],      ℹ️Legg inn variablene du vil konvertere til boolske verdier
    }

    NB:
    ℹ️ Variabler som ikke legges inn her, blir IKKE endret.
    ℹ️ Hvis du angir en variabel som ikke finnes i dataframen, får du en advarsel.
    ℹ️ Du kan la lister stå tomme hvis ingen variabler skal konverteres i en gitt gruppe.
    """
    df = df.copy()
    warnings = []

    for gruppe, kolonner in dtype_mapping.items():
        for kol in kolonner:
            if kol not in df.columns:
                warnings.append(f"Advarsel: Kolonnen '{kol}' finnes ikke i dataframen.")
                continue

            if gruppe == "klassifikasjonsvariabel":
                df[kol] = df[kol].astype("category")

            elif gruppe == "heltall":
                avrundet = _round_half_up(df[kol], decimals=0)
                # Nullable int-type for å tillate NaN
                df[kol] = avrundet.astype("Int64")

            elif gruppe == "desimaltall_1_des":
                df[kol] = _round_half_up(df[kol], decimals=1)

            elif gruppe == "desimaltall_2_des":
                df[kol] = _round_half_up(df[kol], decimals=2)

            elif gruppe == "stringvar":
                df[kol] = df[kol].astype("string")

            elif gruppe == "bool_var":
                # Enkel variant: anta at verdiene allerede er 0/1 eller bool
                df[kol] = df[kol].astype("boolean")

            else:
                warnings.append(
                    f"Advarsel: Ukjent gruppe '{gruppe}' for kolonnen '{kol}'. Ingen konvertering utført."
                )

    # Her kan du velge:
    # - returnere warnings,
    # - eller printe dem,
    # - eller begge deler.
    if warnings:
        for w in warnings:
            print(w)

    df_dtypes = df.dtypes

    display(df)

    logger.info("ℹ️ Under ser du en oversikt over variabeltypene:\n")
    logger.info("ℹ️ category -> klassifikasjonsvariabel")
    logger.info("ℹ️ string[python] -> stringvariabel")
    logger.info("ℹ️ Int64 -> heltall")
    logger.info("ℹ️ float64 -> desimaltall")
    logger.info("ℹ️ boolean -> booleansk variabel (1/0, ja/nei)\n")

    logger.info(
        "ℹ️ Under ser du dtypene som kjennetegner variablene dine etter prosedyren:\n"
    )
    display(df_dtypes)

    return df, df_dtypes


# %%
