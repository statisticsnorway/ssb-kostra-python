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
# ##### aggregerer opp de ettårige aldersgrupperingene til sammenslåtte KOSTRA-aldersgrupperinger.
# ##### aggregerer opp Oslo-bydeler til samlegrupperingen EAB for alle bydelene.
# ### Funksjonen vi bruker heter “regionshierarki”. Denne ligger på kostra-fellesfunksjoner/fellesfunksjoner/src/funksjoner.
# ### Vi laster den inn med “from functions.funksjoner import regionshierarki”.

# %%
from unittest.mock import patch

import pandas as pd
from IPython.display import display  # for nice tables in notebooks

INPUT_PATCH_TARGET = "builtins.input"

from ssb_kostra_python import enkel_editering
from ssb_kostra_python import hjelpefunksjoner
from ssb_kostra_python.avrunding import konverter_dtypes
from ssb_kostra_python.avrunding import print_instruks_konverter_dtypes
from ssb_kostra_python.summere_til_aldersgrupperinger import (
    summere_til_aldersgrupperinger,
)

# %% [markdown]
# ### Først henter vi ned en folketallsfil som fordeler Oslo-befolkningen på kjønn, bydel og alder.

# %%
folkemengde_bydeler_2024 = pd.read_parquet(
    "gs://ssb-dapla-felles-data-produkt-prod/kostra/eksempeldata/folketall_bydeler_2024.parquet"
)

display(folkemengde_bydeler_2024)

# %% [markdown]
# ### Vi må også hente ned en manuelt laget mappingfil for aldersgrupperingene.
# #### Mappingen lagres som hierarki_path.

# %%
hierarki_path = "gs://ssb-dapla-felles-data-produkt-prod/kostra/eksempeldata/mapping_aldershierarki.parquet"

# %% [markdown]
# ### Kjøre aldersaggregeringsfunksjon uten forhåndsdefinerte klassifikasjonsvariable.
# #### Når du kjører aldersaggregeringsfunksjonen må du alltid huske på at "to" også er en variabel du skal legge inn i tekstfeltet i tillegg til "kjonn" og "alder".
# #### df_sum_med_kjonn er datasettet som genereres etter at funksjonen er kjørt. Datasettet inneholder de aggregerte aldersgrupperingene.

# %%
rename_variabel, groupby_variable, df_sum_med_kjonn = summere_til_aldersgrupperinger(
    folkemengde_bydeler_2024, hierarki_path
)

# %% [markdown]
# ### Kjøre aldersaggregeringsfunksjon med forhåndsdefinerte klassifikasjonsvariable
# #### Vi forhåndsdefinerer klassifikasjonsvariablene som ellers må føres inn i tekstfeltet som:
# #### predefined_input = "kjonn, alder, to"

# %%
predefined_input = "kjonn, alder, to"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    rename_variabel, groupby_variable, df_sum_med_kjonn = (
        summere_til_aldersgrupperinger(folkemengde_bydeler_2024, hierarki_path)
    )

# %% [markdown]
# ### Vi ser at "personer" er ført som desimaltall. Vi kan konvertere denne variabelen til heltall.
# ### Vi kan skrive ut instruksene for å se hvordan vi skal utføre mappingen med:
# #### instruks = avrunding.print_instruks_konverter_dtypes()
# #### Mappingen som dukker opp kan først kopieres, og så kan du føre inn de variablene som skal konverteres.

# %%
# Skriver ut instruksen
instruks = print_instruks_konverter_dtypes()

# %% [markdown]
# #### Vi kan kopiere malen over og føre inn variablene som må endres.
# #### avrundet, avrundet_dtypes = konverter_dtypes(df_sum_med_kjonn, dtype_mapping) betyr:
# ##### avrundet er datasettet som spyttes ut etter konvertering.
# ##### avrundet_dtypes er oversikten over variabeltypene. Vi ser nederst at person har blitt omgjort til Int64.

# %%
dtype_mapping = {
    "klassifikasjonsvariabel": [
        "periode",
        "bydelsregion",
        "kjonn",
        "alder",
    ],
    "heltall": [
        "personer",  # <------- Vi gjør om "personer" til heltall (Int64)
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

# %% [markdown]
# ### Her kan vi editere et datasett.
# #### Vi må først angi klassifikasjonsvariablene utover periode og region. Gjør det med predefined_input = "kjonn, alder, __row_id__".
# #### __row_id__ legger seg alltid på datasettet for å kunne identifisere radnummer. Denne må også identifiseres som klassifikasjonsvariabel.
# #### Så kan du iverksette funksjonen på datasettet som skal editeres, i dette tilfelle "avrundet", som ble generert lenger oppe.
# #### Det åpner seg en meny der du fører inn endringene.
# #### Vi ønsker å endre rad 0 (__row_id__ = 0) og trykker "Apply filter".
# #### Under "preview & Edit" vises raden som er søkt opp.
# #### Vi ønsker å endre verdien på "personer" fra 480 til 100000. Vi fører 100000 i "New value".
# #### Vi må også føre en årsak til endring. Til slutt trykker vi  på "Commit Edit".
# #### Nytt datasett med endringer og logg.

# %%
predefined_input = "kjonn, alder, __row_id__"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    get_results = enkel_editering.dataframe_cell_editor_mvp(avrundet)

# %%
df_edited, change_log_df = get_results()
display(df_edited)
display(change_log_df)

# %%
