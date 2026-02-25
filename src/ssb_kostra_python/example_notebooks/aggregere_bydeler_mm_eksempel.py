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
# ### I dette eksempelarket ser vi på hvordan vi:
# ##### aggregerer opp de ettårige aldersgrupperingene til sammenslåtte KOSTRA-aldersgrupperinger.
# ##### aggregerer opp Oslo-bydeler til samlegrupperingen EAB for alle bydelene.
# ### Funksjonen vi bruker heter “regionshierarki”. Denne ligger på kostra-fellesfunksjoner/fellesfunksjoner/src/funksjoner.
# ### Vi laster den inn med “from functions.funksjoner import regionshierarki”.

# %%
from unittest.mock import patch

import pandas as pd
from IPython.display import display  # for nice tables in notebooks

INPUT_PATCH_TARGET = "builtins.input"

from ssb_kostra_python.avrunding import konverter_dtypes
from ssb_kostra_python.avrunding import print_instruks_konverter_dtypes
from ssb_kostra_python.summere_til_aldersgrupperinger import summere_til_aldersgrupperinger
from ssb_kostra_python import enkel_editering
from ssb_kostra_python import hjelpefunksjoner

#from functions.funksjoner import avrunding
#from functions.funksjoner import enkel_editering
#from functions.funksjoner import hjelpefunksjoner
#from functions.funksjoner import summere_til_aldersgrupperinger

# %% [markdown]
# ### Først henter vi ned en folketallsfil som fordeler Oslo-befolkningen på kjønn, bydel og alder.

# %%
# Henter ned fil
folketall_bydeler = pd.read_csv(
    "gs://ssb-off-fin-data-tilsky-prod/befolkning/Folketall 3sifra 2024 Bydeler.csv",
    delimiter=";",
    encoding="latin1",
)
# Viser fil
display(folketall_bydeler)

# %% [markdown]
# ### Vi må også hente ned en manuelt laget mappingfil for aldersgrupperingene.
# #### Mappingen lagres som hierarki_path.

# %%
hierarki_path = "gs://ssb-off-fin-data-produkt-prod/befolkning/_config/mapping_aldershierarki.parquet"

# %% [markdown]
# ### Kjøre aldersaggregeringsfunksjon uten forhåndsdefinerte klassifikasjonsvariable.
# #### Når du kjører aldersaggregeringsfunksjonen må du alltid huske på at "to" også er en variabel du skal legge inn i tekstfeltet i tillegg til "kjonn" og "alder".
# #### df_sum_med_kjonn er datasettet som genereres etter at funksjonen er kjørt. Datasettet inneholder de aggregerte aldersgrupperingene.

# %%
rename_variabel, groupby_variable, df_sum_med_kjonn = (
    summere_til_aldersgrupperinger(
        folketall_bydeler, hierarki_path
    )
)

# %% [markdown]
# ### Kjøre aldersaggregeringsfunksjon med forhåndsdefinerte klassifikasjonsvariable
# #### Vi forhåndsdefinerer klassifikasjonsvariablene som ellers må føres inn i tekstfeltet som:
# #### predefined_input = "kjonn, alder, to"

# %%
predefined_input = "kjonn, alder, to"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    rename_variabel, groupby_variable, df_sum_med_kjonn = (
        summere_til_aldersgrupperinger(
            folketall_bydeler, hierarki_path
        )
    )

# %% [markdown]
# ### Vi ser at "personer" er ført som desimaltall. Vi kan konvertere denne variabelen til heltall.
# ### Vi kan skrive ut instruksene for å se hvordan vi skal utføre mappingen med:
# #### instruks = avrunding.print_instruks_konverter_dtypes()
# #### Mappingen som dukker opp kan først kopieres, og så kan du føre inn de variablene som skal konverteres.

# %%
# Skriver ut instruksen
instruks = print_instruks_konverter_dtypes()

# %%
dtype_mapping = {
    "klassifikasjonsvariabel": [
        "periode",
        "bydelsregion",
        "kjonn",
        "alder",
    ],
    "heltall": [
        "personer",
    ],
    "desimaltall_1_des": [],
    "desimaltall_2_des": [],
    "stringvar": [],
    "bool_var": [],
}

avrundet, avrundet_dtypes = konverter_dtypes(df_sum_med_kjonn, dtype_mapping)

# %%
display(avrundet)

# %%
predefined_input = "kjonn, alder"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    klassifikasjonsvariable, statistikkvariable = (
        hjelpefunksjoner.definere_klassifikasjonsvariable(avrundet)
    )


display(klassifikasjonsvariable)
display(statistikkvariable)

# %%
predefined_input = "kjonn, alder, __row_id__"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    get_results = enkel_editering.dataframe_cell_editor_mvp(avrundet)

# %%
df_edited, change_log_df = get_results()
display(df_edited)
display(change_log_df)

# %%
