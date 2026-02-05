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
from klass import KlassClassification
from klass import KlassCorrespondence
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype, is_bool_dtype
from fagfunksjoner import logger
import re
from IPython.display import display  # for nice tables in notebooks
import ipywidgets as widgets
from IPython.display import display
import dapla as dp
import sys
import time
from unittest.mock import patch
INPUT_PATCH_TARGET = "builtins.input"
from functions.funksjoner import hjelpefunksjoner


# %%
def summere_til_aldersgrupperinger(inputfil: pd.DataFrame, hierarki_path: str):
    """
    Aggregerer individbaserte aldersverdier til forhåndsdefinerte aldersgrupper
    ved hjelp av et aldershierarki, og slår de aggregerte verdiene sammen med
    originaldatasettet.

    Funksjonen:
    - Leser inn et aldershierarki fra fil (parquet)
    - Tilpasser datatyper og formatering for korrekt kobling
    - Mapper individuelle aldersverdier til aldersgrupper ("from" → "to")
    - Summerer statistikkvariabler (f.eks. antall personer) over aldersgrupper
    - Bevarer øvrige klassifikasjonsvariabler (f.eks. periode, kjønn, region)
    - Returnerer et datasett som inneholder både originale aldre og aggregerte
      aldersgrupper

    Parametere
    ----------
    inputfil : pd.DataFrame
        Inndatafil med individbaserte eller finmaskede aldersverdier.
        Forutsetter minst følgende kolonner:
        - 'periode' (år)
        - 'alder' (3-sifret alderskode)
        - én eller flere statistikkvariabler (f.eks. 'personer')

    hierarki_path : str
        Filsti til parquet-fil som inneholder aldershierarki.
        Forutsetter følgende kolonner:
        - 'periode' : år
        - 'from'    : alder (finmaskert nivå)
        - 'to'      : aldersgruppe

    Returverdier
    ------------
    rename_variabel : list[str]
        Liste med variabler som erstattes i aggregeringen
        (for tiden ['alder']).

    groupby_variable : list[str]
        Klassifikasjonsvariabler som brukes i gruppering ved aggregering
        (alle klassifikasjonsvariabler unntatt 'alder').

    df_combined : pd.DataFrame
        Datasett som inneholder både:
        - opprinnelige aldersnivåer
        - aggregerte aldersgrupper
        med identiske klassifikasjons- og statistikkvariabler.

    Merknader
    ---------
    - Kun perioder som finnes i `inputfil` blir brukt fra hierarkifilen.
    - Aldershierarkiet forventes å være entydig per periode og alder.
    - Funksjonen forutsetter at hjelpefunksjoner håndterer korrekt
      identifikasjon av klassifikasjons- og statistikkvariabler.
    """

    aldershierarki = pd.read_parquet(hierarki_path)
    inputfil_copy = inputfil.copy()
    # Ensure correct types and formatting for merging
    # Sørger for at formatet på klassifikasjonsvariablene er "string"
    inputfil_copy_formatted = hjelpefunksjoner.format_fil(inputfil_copy)
    # Filter mapping to only include the year(s) present in the main dataset
    # Henter ut det ene året som ligger i folketallsfilen
    available_years = inputfil_copy["periode"].unique()
    # Skiller ut det året i folketallsfilen i aldershierarkifilen
    # aldershierarki["periode"] = aldershierarki["periode"].astype(str)
    aldershierarki_filtered = aldershierarki[aldershierarki["periode"].isin(available_years)].copy()
    aldershierarki_filtered = aldershierarki_filtered[['periode', 'from', 'to']]
    # Convert 'from' to 3-digit strings for joining
    # Setter "from" i hierarkifilen som klassifikasjonsvariabel
    aldershierarki_filtered["from"] = aldershierarki_filtered["from"].astype(str).str.zfill(3)
    # Merge the main data with the mapping on periode and alder ('from')
    # Slår sammen folketallsfilen og hierarkifilen for det aktuelle året
    df_merged = inputfil_copy_formatted.merge(
        aldershierarki_filtered,
        how="inner",
        left_on=["periode", "alder"],
        right_on=["periode", "from"]
    )
    df_merged = df_merged.drop(columns=["from"])
    
    
    klassifikasjonsvariable, statistikkvariable = hjelpefunksjoner.definere_klassifikasjonsvariable(df_merged)
    
    print(df_merged.dtypes)
 
    # Group by cohort and region, summing the persons
    # Genererer datasett kun med antall summert på aldersgrupperinger
    rename_variabel = ["alder"]
    groupby_variable = [x for x in klassifikasjonsvariable if x not in rename_variabel]
    print(f"Aggregerer statistikkvariablen(e) {statistikkvariable} til aldersgrupperinger.")
    print("groupby_variable:")
    print(groupby_variable)
    
    df_cohorts = (
        df_merged.groupby(groupby_variable, as_index=False)[statistikkvariable]
        .sum()
        .rename(columns={"to": "alder"})
    )

    # Concatenate original and cohort-aggregated data
    # Slår sammen den opprinnelige folketallsfilen med folketallsfilen med aggregerte aldersgrupper
    df_combined = pd.concat([inputfil_copy_formatted, df_cohorts], ignore_index=True)
        
    display(df_combined)
    return rename_variabel, groupby_variable, df_combined

# %%
