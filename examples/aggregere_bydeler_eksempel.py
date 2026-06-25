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
# ### I dette eksempelarket ser vi på hvordan vi:
# ##### aggregerer opp Oslo-bydeler til samlegrupperingen EAB for alle bydelene.
# ### Funksjonen vi bruker heter “regionshierarki”. Denne ligger på kssb_kostra_python/src/funksjoner.
# ### Vi laster den inn med from ssb_kostra_python.regionshierarki import (hierarki).

# %%

import pandas as pd
from IPython.display import display  # for nice tables in notebooks

INPUT_PATCH_TARGET = "builtins.input"
from fagfunksjoner import latest_version_path

from ssb_kostra_python.regionshierarki import hierarki

# %% [markdown]
# ### Først henter vi ned en folketallsfil som fordeler Oslo-befolkningen på kjønn, bydel og alder.
# ### Om du ønsker å hente data for et annet år, kan du bare sette statistikkaar til noe annet, slik statistikkaar=20XX.

# %%
# Definerer en filsti. "latest_version_path" (pakke lastet ned over) sørger for å identifisere siste versjon av datasettet.
statistikkaar = 2024
filsti_folkemengde_bydeler = latest_version_path(
    f"/buckets/delt-kostra-befolkning-delt/bydeler/2024/folkmengde_bydeler_p{statistikkaar}-12-31"
)
# Leser selve filen. Denne er lagret som en parquet-fil.
folketall_bydeler = pd.read_parquet(filsti_folkemengde_bydeler)
# Viser datasettet.
display(folketall_bydeler)

# %% [markdown]
# ### Utfører aggregering til KOSTRA-regionsgrupperinger (EAB) for bydelene
# #### folketall_bydeler_EAB = hierarki(folketall_bydeler) vil si at funksjonen "hierarki" utføres på datasettet "folketall_bydeler" og lagres i folketall_bydeler_EAB.
# #### Funksjonen trenger å vite om alle klassifikasjonsvariablene i datasettet. Den identifiserer alltid periode- og regionsvariabelen. De øvrige, i dette tilfellet kjonn og alder må du føre inn selv, skilt fra hverandre med komma.

# %%
folketall_bydeler_EAB = hierarki(folketall_bydeler)
display(folketall_bydeler_EAB)

# %%
