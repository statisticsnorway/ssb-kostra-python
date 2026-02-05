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
from functions.funksjoner import hjelpefunksjoner


# %%
def summere_over_kjonn(inputfil: pd.DataFrame):
    if "kjonn" not in inputfil.columns:
        print("Kjønn er ikke en klassifikasjonsvariabel i datasettet. Ingen summering utføres.")
        return inputfil  # or return None, depending on your pipeline design

    inputfil_copy = inputfil.copy()
    print("Kjønn er en klassifikasjonsvariabel i datasettet.")

    summeringsvariabel = ['kjonn']
    alle_variable = inputfil.columns.tolist()
    klassifikasjonsvariable, statistikkvariable = hjelpefunksjoner.definere_klassifikasjonsvariable(inputfil)
    groupby_variable = [x for x in alle_variable if x not in summeringsvariabel and x not in statistikkvariable]

    print(f"Summerer statistikkvariablen(e) {statistikkvariable} over variablene {summeringsvariabel}.")

    summert_over_kjonn = (
        inputfil_copy
        .groupby(groupby_variable, as_index=False, observed=True)[statistikkvariable]
        .sum()
    )

    print(f"Datasettet har blitt summert over {summeringsvariabel}.")
    print(f"Statistikkvariabelen(e) som har blitt summert er {statistikkvariable}.")

    return summert_over_kjonn

# %%
