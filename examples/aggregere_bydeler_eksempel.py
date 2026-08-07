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
# # I dette eksempelarket ser vi på hvordan vi aggregerer opp Oslo-bydeler til samlegrupperingen EAB for alle bydelene.
# ## I dette eksemplet brukes befolkningsdata, men det du kan anvende funksjonen på dine egne bydelsdata som du trenger å aggregere opp til KOSTRA-regionsgrupperinger.

# %% [markdown]
# Funksjonen vi bruker heter “regionshierarki”. Denne ligger på **ssb_kostra_python/src/funksjoner**.
# Vi laster den inn med `from ssb_kostra_python.regionshierarki import hierarki`.

# %%
import pandas as pd
from IPython.display import display  # for nice tables in notebooks

INPUT_PATCH_TARGET = "builtins.input"
from unittest.mock import patch

from fagfunksjoner import latest_version_path

from ssb_kostra_python.regionshierarki import hierarki

# %% [markdown]
# ## Først henter vi ned en folketallsfil som fordeler Oslo-befolkningen på kjønn, bydel og alder.

# %% [markdown]
# Om du ønsker å hente data for et annet år, kan du bare sette statistikkaar til noe annet, slik: `statistikkaar=20XX`.

# %%
# Definerer en filsti. "latest_version_path" (pakke lastet ned over) sørger for å identifisere siste versjon av datasettet.
statistikkaar = 2023
filsti_folkemengde_bydeler = latest_version_path(
    f"/buckets/delt-kostra-befolkning-delt/bydeler/{statistikkaar}/folkmengde_bydeler_p{statistikkaar}-12-31"
)
# Leser selve filen. Denne er lagret som en parquet-fil.
folketall_bydeler = pd.read_parquet(filsti_folkemengde_bydeler)
# Viser datasettet.
display(folketall_bydeler)

# %% [markdown]
# ## Utfører aggregering til KOSTRA-regionsgrupperinger (EAB) for bydelene

# %% [markdown]
# `folketall_bydeler_EAB = hierarki(folketall_bydeler, , add_region_names=True)` vil si at funksjonen **“hierarki”** utføres på datasettet **“folketall_bydeler”** og lagres i **folketall_bydeler_EAB**. Om du presiserer at **add_region_names=True**, sørger du for at regionsnavnene inkluderes i datasettet etter hierarkioperasjonen. Om du ønsker regionsnavn i datasettet etter hierarki, må du presisere dette også selv om datasettet inneholder regionsnavn før hierarkioperasjonen. Dette er fordi hierarkioperasjonen ikke aggregerer regionsnavnene i tråd med en mapping, men fjerner regionsnavnene midlertidig før kodene aggregeres, og deretter legger dem på igjen. Om du setter **add_region_names=False**, vil det endelige datasettet ikke beholde regionsnavn, uansett om det inneholdt regionsnavn før hierarkioperasjonen eller ei. Du kan veksle mellom **True** og **False** i koden under for å se hvordan datasettet genereres på de to ulike måtene.
#
# Funksjonen trenger å vite om alle klassifikasjonsvariablene i datasettet. Den identifiserer alltid **periode- og regionsvariabelen**. De øvrige, i dette tilfellet **kjonn** og **alder**, må du føre inn selv, skilt fra hverandre med komma.

# %%
folketall_bydeler_EAB_navn = hierarki(folketall_bydeler, add_region_names=True)
display(folketall_bydeler_EAB_navn)

# %% [markdown]
# ## Det kan være slitsomt å måtte taste inn klassifikasjonsvariablene hver gang funksjonen kjøres.

# %% [markdown]
# Når du setter opp et produksjonsløp og du vet hvilke klassifikasjonsvariable som inngår når funksjonen kjøres, kan du sette opp koden som vist under for å unngå dette.
# Du forhåndsdefinerer klassifikasjonsvariablene i “predefined_input”. Deretter kopler du den opprinnelige funksjonen til de forhåndsdefinerte inputene som vist under. Funksjonen vil ta i bruk de forhåndsdefinerte inputene og kjøre uten å be deg taste dem inn.

# %%
predefined_input = "kjonn, alder"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    folketall_bydeler_EAB = hierarki(folketall_bydeler, add_region_names=True)

display(folketall_bydeler_EAB)

# %%
