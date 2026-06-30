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
# ### I dette eksempelarket ser vi på hvordan vi aggregerer opp kommuner til KOSTRA-grupper.
# ### Funksjonen vi bruker heter "regionshierarki". Denne ligger på kostra-fellesfunksjoner/fellesfunksjoner/src/funksjoner.
# ### Vi laster den inn med "from functions.funksjoner import regionshierarki".

# %%
INPUT_PATCH_TARGET = "builtins.input"
from unittest.mock import patch

import duckdb
from fagfunksjoner import latest_version_path
from IPython.display import display  # for nice tables in notebooks

# from ssb_kostra_python import regionshierarki
from ssb_kostra_python.regionshierarki import hierarki

# %% [markdown]
# ### Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.
# ### Dataene kommer fra delt-bøtten til seksjon for befolkning.
# ### Vi må bearbeide dataene litt slik at de blir likere dataene slik vi kjenner dem i KOMPIS.
# ### Om du ønsker å hente data for et annet år, kan du bare sette statistikkaar til noe annet, slik: statistikkaar=20XX.

# %%
statistikkaar = 2015
df_kommuner_data_path = latest_version_path(
    f"/buckets/shared/bef-statistikk/folketall/bosatte/{statistikkaar}/bosatte_p{statistikkaar}-12-31.parquet"
)

df_folketall_kommuner = duckdb.query(f"""
    SELECT kjoenn AS kjonn,
           komm_nr AS kommuneregion,
           alder
    FROM '{df_kommuner_data_path}'
""").to_df()

df_folketall_kommuner["periode"] = statistikkaar
df_folketall_kommuner["personer"] = 1

df_folketall_kommuner = df_folketall_kommuner.groupby(
    ["periode", "kommuneregion", "kjonn", "alder"], as_index=False
)[["personer"]].sum()

display(df_folketall_kommuner)

# %% [markdown]
# ### Utfører aggregering til KOSTRA-regionsgrupperinger (EAK, EAKUO, EKA, EKG) for bydelene
# #### folketall_kommuner_KOSTRA = hierarki(df_folketall_kommuner) betyr at funksjonen “hierarki” utføres på datasettet “df_folketall_kommuner” og lagres i folketall_kommuner_KOSTRA.
# #### Funksjonen trenger å vite om alle klassifikasjonsvariablene i datasettet. Den identifiserer alltid periode- og regionsvariabelen. De øvrige, i dette tilfellet kjonn og alder må du føre inn selv, skilt fra hverandre med komma.

# %%
folketall_kommuner_KOSTRA = hierarki(df_folketall_kommuner)
display(folketall_kommuner_KOSTRA)

# %% [markdown]
# #### Det kan være slitsomt å måtte taste inn klassifikasjonsvariablene hver gang funksjonen kjøres.
# #### Når du setter opp et produksjonsløp og du vet hvilke klassifikasjonsvariable som inngår når funksjonen kjøres, kan du sette opp koden som vist under for å unngå dette.
# #### Du forhåndsdefinerer klassifikasjonsvariablene i "predefined_input". Deretter kopler du den opprinnelige funksjonen til de forhåndsdefinerte inputene som vist under. Funksjonen vil ta i bruk de forhåndsdefinerte inputene og kjøre uten å be deg taste dem inn.

# %%
predefined_input = "kjonn, alder"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    folketall_kommuner_KOSTRA = hierarki(df_folketall_kommuner)

display(folketall_kommuner_KOSTRA)

# %%
