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

# %% [markdown]
# # I dette eksempelarket ser vi på hvordan vi aggregerer opp ettårige aldersgrupper til sammensatte KOSTRA-aldersgrupperinger.

# %% [markdown]
# Funksjonen vi bruker heter "summere_til_aldersgrupperinger". Denne ligger på **ssb-kostra-python/src/funksjoner**.
# Vi laster den inn med `from ssb_kostra_python.summere_til_aldersgrupperinger import (summere_til_aldersgrupperinger,)`

# %% [markdown]
# ### Laster ned pakker

# %%
import pandas as pd
from fagfunksjoner import latest_version_path

INPUT_PATCH_TARGET = "builtins.input"
from unittest.mock import patch

from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python.summere_til_aldersgrupperinger import (
    summere_til_aldersgrupperinger,
)

# %% [markdown]
# ### Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.

# %% [markdown]
# Du kan selv bestemme hvilket statistikkår tabellen skal gjelde ved å endre på **statistikkaar = 20XX**.

# %%
# Bestemmer først statistikkår
statistikkaar = 2024
# Definerer en filsti. "latest_version_path" (pakke lastet ned over) sørger for å identifisere siste versjon av datasettet.
filsti_folkemengde_bydeler = latest_version_path(
    f"/buckets/delt-kostra-befolkning-delt/bydeler/{statistikkaar}/folkmengde_bydeler_p{statistikkaar}-12-31"
)
# Leser selve filen. Denne er lagret som en parquet-fil.
folketall_bydeler = pd.read_parquet(filsti_folkemengde_bydeler)
# Viser datasettet.
display(folketall_bydeler)

# %% [markdown]
# ### Summerer opp datasettet "folketall_bydeler" til KOSTRA-aldersgrupperinger med manuell inntasting av klassifikasjonsvariable.

# %% [markdown]
# Funksjonen trenger å vite klassifikasjonsvariablene for å aggregere riktig.
# Når du aggregerer til KOSTRA-aldersgrupperinger, vil mappingfilen inneholde en klassifikasjonsvariabel **to**. Ikke glem å skrive inn denne også.

# %%
# Summerer til KOSTRA-aldersgrupperinger
folketall_bydeler_alder = summere_til_aldersgrupperinger(folketall_bydeler)

# %% [markdown]
# ### Summerer opp datasettet "folketall_bydeler" til KOSTRA-aldersgrupperinger med forhåndsdefinering av klassifikasjonsvariable.

# %% [markdown]
# Det kan være slitsomt å måtte taste inn klassifikasjonsvariablene hver gang funksjonen kjøres.
# Når du setter opp et produksjonsløp og du vet hvilke klassifikasjonsvariable som inngår når funksjonen kjøres, kan du sette opp koden som vist under for å unngå dette.
# Du forhåndsdefinerer klassifikasjonsvariablene i **predefined_input**. Deretter kopler du den opprinnelige funksjonen til de forhåndsdefinerte inputene som vist under. Funksjonen vil ta i bruk de forhåndsdefinerte inputene og kjøre uten å be deg taste dem inn.

# %%
predefined_input = "kjonn, alder, to"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    folketall_bydeler_alder = summere_til_aldersgrupperinger(folketall_bydeler)

# Viser datasettet
display(folketall_bydeler_alder)

# %%
