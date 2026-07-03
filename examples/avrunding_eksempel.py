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
# ### I dette eksempelarket ser vi på hvordan vi kan endre typen på variablene i et datasett.
# ### Funksjonen vi bruker heter "avrunding". Denne ligger på kostra-fellesfunksjoner/fellesfunksjoner/src/funksjoner
# ### Vi laster den inn med "from ssb_kostra_python import avrunding"

# %%
import pandas as pd

INPUT_PATCH_TARGET = "builtins.input"
from fagfunksjoner import latest_version_path
from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python import avrunding

# %% [markdown]
# ### Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.

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
# ### Først en enkel visning av datasettet. Så en oppstilling av variabeltypene.
# #### Vi ser at alle variablene er formatert som heltall (int64). I dette tilfellet er "periode", "kommuneregion", "kjonn" og "alder" klassifikasjonsvariable, så det gir mening å omgjøre dem til tekstvariable (string[python])

# %%
# Skriver ut datasettet
display(folketall_bydeler)
# Skriver ut variabeltypene
display(folketall_bydeler.dtypes)

# %% [markdown]
# ### Manipulerer dataene litt

# %%
folketall_bydeler["personer"] = folketall_bydeler["personer"] / 2.32
display(folketall_bydeler)

# %% [markdown]
# ### Nedenfor endrer vi variabeltypene. "personer" skal naturligvis være heltall, men la oss si at vi ønsker å gjøre dem til desimaltall med to desimaler. Klassifikasjonsvariablene kan gjøres om fra heltall til category (eller string).
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
    "klassifikasjonsvariabel": ["periode", "bydelsregion", "kjonn", "alder"],
    "heltall": [],
    "desimaltall_1_des": [],
    "desimaltall_2_des": ["personer"],
    "stringvar": [],
    "bool_var": [],
}

# Utfører avrundingen/konverteringen
folkemengde_kommune_2024, dtypes = avrunding.konverter_dtypes(
    folketall_bydeler, dtype_mapping
)

# %% [markdown]
# ### Omformaterer variablene på en annen måte.

# %%
# Skriver ut instruksen
instruks = avrunding.print_instruks_konverter_dtypes()

# Lager mappingen
dtype_mapping = {
    "klassifikasjonsvariabel": [],
    "heltall": [],
    "desimaltall_1_des": ["personer"],
    "desimaltall_2_des": [],
    "stringvar": ["periode", "bydelsregion", "kjonn", "alder"],
    "bool_var": [],
}

# Utfører avrundingen/konverteringen
folkemengde_kommune_2024, dtypes = avrunding.konverter_dtypes(
    folkemengde_kommune_2024, dtype_mapping
)

# %%
df = pd.DataFrame(
    {
        "col1": ["1", "2", "3", "4", "5"],
        "var1": [1.5, 2.5, 3.5, -1.5, -2.5],
        "var2": [0.125, 0.575, 1.005, 1.275, 2.445],
    }
)
display(df)
# _round_half_up(df['var2'], 2)

# %%
# Lager mappingen
dtype_mapping = {
    "klassifikasjonsvariabel": [],
    "heltall": [],
    "desimaltall_1_des": [],
    "desimaltall_2_des": ["var2"],
    "stringvar": [],
    "bool_var": [],
}

# Utfører avrundingen/konverteringen
df, dtypes = avrunding.konverter_dtypes(df, dtype_mapping)

# %%
