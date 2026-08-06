# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: ssb-kostra-python
#     language: python
#     name: ssb-kostra-python
# ---

# %%
from decimal import ROUND_HALF_UP
from decimal import Decimal
from decimal import InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
from fagfunksjoner.fagfunksjoner_logger import logger
from IPython.display import display


# %%
def _round_half_up(values: pd.Series, decimals: int = 0) -> pd.Series:
    """Runder kommersielt til valgt antall desimaler.

    Verdier som ligger nøyaktig midt mellom to mulige resultater,
    rundes bort fra null:

    1.005  -> 1.01
    -1.005 -> -1.01

    Manglende eller ugyldige verdier returneres som NaN.
    """
    if decimals < 0:
        raise ValueError("'decimals' kan ikke være negativ.")

    numeric_values = pd.to_numeric(values, errors="coerce")

    quantizer = Decimal("1").scaleb(-decimals)

    def round_value(value: Any) -> float:
        if pd.isna(value):
            return np.nan

        try:
            decimal_value = Decimal(str(value))
            rounded_value = decimal_value.quantize(
                quantizer,
                rounding=ROUND_HALF_UP,
            )
            return float(rounded_value)

        except (InvalidOperation, ValueError, TypeError):
            return np.nan

    rounded = numeric_values.map(round_value)

    return pd.Series(
        rounded,
        index=values.index,
        name=values.name,
        dtype="float64",
    )


def print_instruks_konverter_dtypes() -> str:
    """Lager instruks for å lage mapping."""
    instruks = """ℹ️Bruk malen under for dtype_mapping. Du må angi denne mappingen i forkant for at funksjonen skal kunne konvertere variablene slik du ønsker.

    dtype_mapping = {
        "klassifikasjonsvariabel":  ["var1", "var2"],        ℹ️Legg inn variablene du vil klassifikasjonsverdier
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
    """Bruk malen under for ``dtype_mapping``.

    Du må angi denne mappingen i forkant for at funksjonen skal kunne konvertere variablene slik du ønsker.

    Eksempel::

        dtype_mapping = {
            "klassifikasjonsvariabel": ["var1", "var2"],  # Legg inn variablene du vil klassifikasjonsverdier
            "heltall": ["var3", "var4"],  # Legg inn variablene du vil runde av til heltall (kommersiell avrunding)
            "desimaltall_1_des": ["var5", "var6"],  # Legg inn variablene du vil runde til 1 desimal
            "desimaltall_2_des": ["var7", "var8"],  # Legg inn variablene du vil runde til 2 desimaler
            "stringvar": ["var9", "var10"],  # Legg inn variablene du vil konvertere til tekst
            "bool_var": ["var11", "var12"],  # Legg inn variablene du vil konvertere til boolske verdier
        }

    Merknader:

    - Variabler som ikke legges inn her, blir ikke endret.
    - Hvis du angir en variabel som ikke finnes i dataframen, får du en advarsel.
    - Du kan la lister stå tomme hvis ingen variabler skal konverteres i en gitt gruppe.

    Du har for eksempel et datasett "df" med klassifikasjonsvariablene "periode", "bydelsregion" og "alder", og i tillegg tellevariabelen "personer".
    Da lager du mappingen slik:

    dtype_mapping = {
        "klassifikasjonsvariabel":  ["periode", "bydelsregion", "alder"],
        "heltall":                  ["personer"],
        "desimaltall_1_des":        [],
        "desimaltall_2_des":        [],
        "stringvar":                [],
        "bool_var":                 []}

    Mappingen er ikke selve funksjonen, men info til funksjonen. Funksjonen skrives slik:

    df_konvertert, dtypes = avrunding.konverter_dtypes(df, dtype_mapping)

    Til venstre for likhetstegnet ser du to objekter. Det første, i dette tilfellet "df" er alltid det konverterte datasettet. Den andre, i dette tilfellet "dtypes"
    er typekartleggingen etter konverteringen. I parentesen til høyre for likhetstegnet ser du argumentene, altså inputen/info til funksjonen. Det første er
    datasettet som skal konverteres, i dette tilfellet "df". Det andre er mappingen, i dette tilfellet "dtype_mapping" der du har lagt inn variablene som skal
    konverteres til de ulike typene.
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
