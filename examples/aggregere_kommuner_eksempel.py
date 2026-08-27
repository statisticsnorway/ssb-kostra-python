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
# # I dette eksempelarket ser vi på hvordan vi aggregerer opp kommuner til KOSTRA-grupper.
# ## I dette eksemplet brukes befolkningsdata, men det du kan anvende funksjonen på dine egne kommunedata som du trenger å aggrere opp til KOSTRA-regionsgrupperinger.

# %% [markdown]
# Funksjonen vi bruker heter **regionshierarki**. Denne ligger på **kostra-fellesfunksjoner/fellesfunksjoner/src/funksjoner**.
# Vi laster den inn med `from ssb_kostra_python.regionshierarki import hierarki`.

# %%
INPUT_PATCH_TARGET = "builtins.input"
from unittest.mock import patch

import duckdb
from fagfunksjoner import latest_version_path
from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python.regionshierarki import hierarki
from ssb_kostra_python.titler_til_klasskoder import mapping_regionsnavn

# %% [markdown]
# ## Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.

# %% [markdown]
# Dataene kommer fra delt-bøtten til seksjon for befolkning.
# Vi må bearbeide dataene litt slik at de blir likere dataene slik vi kjenner dem i KOMPIS.
# Om du ønsker å hente data for et annet år, kan du bare sette statistikkaar til noe annet, slik: `statistikkaar=20XX`.

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
# ## Vi kan feste regionsnavn på regionskodene

# %%
df_folketall_kommuner_navn = mapping_regionsnavn(df_folketall_kommuner)
display(df_folketall_kommuner_navn)

# %% [markdown]
# ## Utfører aggregering til KOSTRA-regionsgrupperinger (EAK, EAKUO, EKA, EKG) for kommunene
# ### Du kan velge om datasettet skal inneholde regionsnavn etter aggregering.

# %% [markdown]
# `folketall_kommuner_KOSTRA = hierarki(df_folketall_kommuner_navn, add_region_names=True)` betyr at funksjonen **hierarki** utføres på datasettet **df_folketall_kommuner** og lagres i **folketall_kommuner_KOSTRA**. Om du presiserer at `add_region_names=True`, sørger du for at regionsnavnene inkluderes i datasettet etter hierarkioperasjonen. Om du ønsker regionsnavn i datasettet etter hierarki, må du presisere dette **også selv om datasettet inneholder regionsnavn før hierarkioperasjonen**. Dette er fordi hierarkioperasjonen ikke aggregerer regionsnavnene i tråd med en mapping, men fjerner regionsnavnene midlertidig før kodene aggregeres, og deretter legger dem på igjen. Om du setter `add_region_names=False`, vil det endelige datasettet ikke beholde regionsnavn, uansett om det inneholdt regionsnavn før hierarkioperasjonen eller ei. Du kan veksle mellom **True** og **False** i koden under for å se hvordan datasettet genereres på de to ulike måtene.
#
# Dersom funksjonen kjøres som vist under, vil den trenge å vite om alle klassifikasjonsvariablene i datasettet. Den identifiserer alltid **periode-** og **regionsvariabelen**. De øvrige, i dette tilfellet **kjonn** og **alder**, må du føre inn selv, skilt fra hverandre med komma.

# %%
folketall_kommuner_KOSTRA = hierarki(df_folketall_kommuner_navn, add_region_names=True)

print("\n")
print(f"ℹ️Det endelige datasettet heter her 'folketall_kommuner_KOSTRA'.")
display(folketall_kommuner_KOSTRA)

# %% [markdown]
# ## Det kan være slitsomt å måtte taste inn klassifikasjonsvariablene hver gang funksjonen kjøres.

# %% [markdown]
# Når du setter opp et fast produksjonsløp og du vet hvilke klassifikasjonsvariable som inngår når funksjonen kjøres, kan du sette opp koden som vist under for å unngå dette.
# Du forhåndsdefinerer klassifikasjonsvariablene i **predefined_input**. Deretter kopler du den opprinnelige funksjonen til de forhåndsdefinerte inputene som vist under. Funksjonen vil ta i bruk de forhåndsdefinerte inputene og kjøre uten å be deg taste dem inn.

# %%
predefined_input = "kjonn, alder"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    folketall_kommuner_KOSTRA = hierarki(df_folketall_kommuner, add_region_names=True)

print("\n")
print(f"ℹ️Det endelige datasettet heter her 'folketall_kommuner_KOSTRA'.")
display(folketall_kommuner_KOSTRA)

# %%
