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
# ### I dette eksempelarket ser vi på hvordan vi aggregerer opp antallet mennesker på kjønn i et datasett som fordeler på mann og kvinne.
# ### Funksjonen vi bruker heter "summere_kjonn". Denne ligger på ssb-kostra-python/src/funksjoner.
# ### Vi laster den inn med "from ssb_kostra_python import summere_kjonn".

# %%
import pandas as pd
from fagfunksjoner import latest_version_path

INPUT_PATCH_TARGET = "builtins.input"
from unittest.mock import patch

from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python import summere_kjonn

# %% [markdown]
# ### Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.

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

# %%
# Kjører funksjonen. folketall_bydeler_sum_kjonn er det endelige datasettet som genereres.
folketall_bydeler_sum_kjonn = summere_kjonn.summere_over_kjonn(folketall_bydeler)
# Viser det genererte datasettet. Du vil se at kolonnen for kjønn er borte, for nå er kjønnene summert opp.
display(folketall_bydeler_sum_kjonn)

# %% [markdown]
# ### Det kan være slitsomt å måtte taste inn klassifikasjonsvariablene hver gang funksjonen kjøres.
# ### Når du setter opp et produksjonsløp og du vet hvilke klassifikasjonsvariable som inngår når funksjonen kjøres, kan du sette opp koden som vist under for å unngå dette.
# ### Du forhåndsdefinerer klassifikasjonsvariablene i “predefined_input”. Deretter kopler du den opprinnelige funksjonen til de forhåndsdefinerte inputene som vist under. Funksjonen vil ta i bruk de forhåndsdefinerte inputene og kjøre uten å be deg taste dem inn.

# %%
predefined_input = "kjonn, alder"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    folketall_bydeler_sum_kjonn = summere_kjonn.summere_over_kjonn(folketall_bydeler)

# Viser datasettet
display(folketall_bydeler_sum_kjonn)
