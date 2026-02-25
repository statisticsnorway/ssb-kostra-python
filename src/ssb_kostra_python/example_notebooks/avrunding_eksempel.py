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

# %% [markdown]
# ### I dette eksempelarket ser vi på hvordan vi kan endre typen på variablene i et datasett.
# ### Funksjonen vi bruker heter "avrunding". Denne ligger på kostra-fellesfunksjoner/fellesfunksjoner/src/funksjoner
# ### Vi laster den inn med "from functions.funksjoner import avrunding"

# %%

import pandas as pd

INPUT_PATCH_TARGET = "builtins.input"
from ssb_kostra_python import avrunding
from IPython.display import display  # for nice tables in notebooks

# %% [markdown]
# ### Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.

# %%
folkemengde_kommune_2024 = pd.read_parquet(
    "gs://ssb-off-fin-data-delt-kostra-befolkning-delt-prod/kommune/folkemengde_kommune_2024.parquet"
)

# %% [markdown]
# ### Først en enkel visning av datasettet. Så en oppstilling av variabeltypene.
# #### Vi ser at alle variablene er formatert som heltall (int64). I dette tilfellet er "periode", "kommuneregion", "kjonn" og "alder" klassifikasjonsvariable, så det gir mening å omgjøre
# #### dem til tekstvariable (string[python])

# %%
# Skriver ut datasettet
display(folkemengde_kommune_2024)
# Skriver ut variabeltypene
display(folkemengde_kommune_2024.dtypes)

# %% [markdown]
# ### Nedenfor endrer vi variabeltypene. "personer" skal naturligvis være heltall, men klassifikasjonsvariablene kan gjøres om fra heltall til category (eller string).
# ### Det er en god idé å skrive ut instruksen for å se hvordan du skal lage mappingen. Dette gjør du med:
# #### instruks = avrunding.print_instruks_konverter_dtypes()
# ### Deretter utfører du selve avrundingen/konverteringen med:
# #### df_avrundet, dtypes = avrunding.konverter_dtypes(df_som_skal_behandles, dtype_mapping) der
# #### df_avrundet er det endelige datasettet, dtypes er de nye typene etter konvertering, df_som_skal_behandles er datasettet som skal behandles og dtype_mapping er mappingen du bestemmer.

# %%
# Skriver ut instruksen
instruks = avrunding.print_instruks_konverter_dtypes()

# Lager mappingen
dtype_mapping = {
    "klassifikasjonsvariabel": ["periode", "kommuneregion", "kjonn", "alder"],
    "heltall": ["personer"],
    "desimaltall_1_des": [],
    "desimaltall_2_des": [],
    "stringvar": [],
    "bool_var": [],
}

# Utfører avrundingen/konverteringen
folkemengde_kommune_2024, dtypes = avrunding.konverter_dtypes(
    folkemengde_kommune_2024, dtype_mapping
)


# %%
