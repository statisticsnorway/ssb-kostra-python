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
# ### I dette eksempelarket ser vi på hvordan vi aggregerer opp kommuner til fylkeskommunale KOSTRA-grupper.
# ### Funksjonen vi bruker heter "regionshierarki". Denne ligger på kostra-fellesfunksjoner/fellesfunksjoner/src/funksjoner.
# ### Vi laster den inn med "from functions.funksjoner import regionshierarki".

# %%


INPUT_PATCH_TARGET = "builtins.input"
import duckdb
from fagfunksjoner import latest_version_path
from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python.regionshierarki import hierarki

# %% [markdown]
# ### Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.
# ### Dataene kommer fra delt-bøtten til seksjon for befolkning.
# ### Vi må bearbeide dataene litt slik at de blir likere dataene slik vi kjenner dem i KOMPIS.
# ### Både de aggregerte kommunetabellene og fylkeskommunetabellene har sitt utspring i de delte kommunetabellene til seksjon for befolkning.
# ### Om du ønsker å hente data for et annet år, kan du bare sette statistikkaar til noe annet, slik: statistikkaar=20XX.

# %%
statistikkaar = 2017
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
# ### Utfører aggregering til fylkeskommuner.
# #### folketall_fylkeskommuner_KOSTRA = hierarki(df_folketall_kommuner, "kommune_til_fylkeskommune") betyr at funksjonen “hierarki” utføres på datasettet “df_folketall_kommuner” og lagres i folketall_fylkeskommuner_KOSTRA.
# #### Funksjonen trenger å vite om alle klassifikasjonsvariablene i datasettet. Den identifiserer alltid periode- og regionsvariabelen. De øvrige, i dette tilfellet kjonn og alder må du føre inn selv, skilt fra hverandre med komma.
# #### Når funksjonen aggregerer et kommunedatasett uten videre spesifikasjoner - folketall_kommuner_KOSTRA = hierarki(df_folketall_kommuner) - aggregeres kommunene opp til KOSTRA-kommunegrupperingene EAK, EAKUO, EKA og EKG. Om den spesifiseres slik - folketall_fylkeskommuner_KOSTRA = hierarki(df_folketall_kommuner, "kommune_til_fylkeskommune") - aggregeres de opp til fylkeskommuner, men kun til fylkeskommunene og ikke KOSTRA-grupperingene. Aggregering opp til KOSTRA-fylkeskommunegrupperinger må gjøres i et senere steg.

# %%
folketall_fylkeskommuner = hierarki(df_folketall_kommuner, "kommune_til_fylkeskommune")
display(folketall_fylkeskommuner)

# %% [markdown]
# ### Vi har nå et fylkekommunedatasett som vi kan aggregere opp til de fylkeskommunale KOSTRA-grupperingene.
# ### Vi utfører hierarkifunksjonen på nytt.

# %%
folketall_fylkeskommuner_KOSTRA = hierarki(folketall_fylkeskommuner)
display(folketall_fylkeskommuner_KOSTRA)

# %%
