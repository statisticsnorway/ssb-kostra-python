# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: ssb-kostra-python
#     language: python
#     name: ssb-kostra-python
# ---

# %% [markdown]
# ## I dette eksempelarket ser vi på hvordan vi aggregerer opp ettårige aldersgrupper til sammensatte KOSTRA-aldersgrupperinger.
# ## Funksjonen vi bruker heter "summere_til_aldersgrupperinger". Denne ligger på ssb-kostra-python/src/funksjoner
# ## Vi laster den inn med from ssb_kostra_python.summere_til_aldersgrupperinger import (summere_til_aldersgrupperinger,)

# %% [markdown]
# ### Laster ned pakker

# %%
import pandas as pd
from fagfunksjoner import latest_version_path

INPUT_PATCH_TARGET = "builtins.input"
from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python.summere_til_aldersgrupperinger import (
    summere_til_aldersgrupperinger,
)

# %% [markdown]
# ### Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.

# %%
# Definerer en filsti. "latest_version_path" (pakke lastet ned over) sørger for å identifisere siste versjon av datasettet.
filsti_folkemengde_bydeler_2024 = latest_version_path(
    "/buckets/delt-kostra-befolkning-delt/bydeler/2024/folkmengde_bydeler_p2024-12-31"
)
# Leser selve filen. Denne er lagret som en parquet-fil.
folketall_bydeler = pd.read_parquet(filsti_folkemengde_bydeler_2024)
# Viser datasettet.
display(folketall_bydeler)

# %% [markdown]
# ### Summerer opp datasettet "folketall_bydeler" til KOSTRA-aldersgrupperinger

# %%
# Summerer til KOSTRA-aldersgrupperinger
folketall_bydeler_alder = summere_til_aldersgrupperinger(folketall_bydeler)
