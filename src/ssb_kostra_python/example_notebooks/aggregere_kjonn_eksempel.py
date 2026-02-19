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
# ### I dette eksempelarket ser vi på hvordan vi aggregerer opp antallet mennesker på kjønn i et datasett som fordeler på mann og kvinne.
# ### Funksjonen vi bruker heter "summere_kjonn". Denne ligger på kostra-fellesfunksjoner/fellesfunksjoner/src/funksjoner.
# ### Vi laster den inn med "from functions.funksjoner import summere_kjonn".

# %%
import ast
import csv
import os
import re
from collections import Counter, defaultdict

import dapla as dp
import pandas as pd
import numpy as np
from dapla import FileClient
from klass import KlassClassification
from klass import KlassCorrespondence
from requests.exceptions import HTTPError
from unittest.mock import patch
INPUT_PATCH_TARGET = "builtins.input"
from fagfunksjoner import logger
from IPython.display import display  # for nice tables in notebooks
from functions.funksjoner import avrunding
from functions.funksjoner import summere_til_aldersgrupperinger
from functions.funksjoner import summere_kjonn
from functions.funksjoner import hjelpefunksjoner
from functions.funksjoner import regionshierarki
from functions.funksjoner import validering

# %% [markdown]
# ### Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.

# %%
folkemengde_kommune_2024 = pd.read_parquet("gs://ssb-off-fin-data-delt-kostra-befolkning-delt-prod/kommune/folkemengde_kommune_2024.parquet")

# %% [markdown]
# ### Videre utfører vi noen operasjoner med gjemt visning for å komme til funksjonen som skal gjennomgås.

# %% [markdown]
# <details>
# <summary><b>Dette er også beskrevet andre steder og er ikke viktig her. (Men klikk for å vise)</b></summary>
# ## Gjemt. Dobbeltklikk på den blå søylen til venstre for cellen for å åpne opp.
# ### Nedenfor endrer vi variabeltypene. "personer" skal naturligvis være heltall, men klassifikasjonsvariablene må gjøres om fra heltall til string. 
# ### Det er en god idé å skrive ut instruksen for å se hvordan du skal lage mappingen. Dette gjør du med: 
# #### instruks = avrunding.print_instruks_konverter_dtypes()
# ### Deretter utfører du selve avrundingen/konverteringen med: 
# #### df_avrundet, dtypes = avrunding.konverter_dtypes(df_som_skal_behandles, dtype_mapping) der
# #### df_avrundet er det endelige datasettet, dtypes er de nye typene etter konvertering, df_som_skal_behandles er datasettet som skal behandles og dtype_mapping er mappingen du bestemmer.
# </details>

# %% [markdown]
# <details>
# <summary><b>Dette er også beskrevet andre steder og er ikke viktig her. (Men klikk for å vise)</b></summary>
# ### Gjemt. Dobbeltklikk på den blå søylen til venstre for å åpne opp.
# #### Her konverterer vi datatypene. Dette er tidligere vist i eksempelarket "avrunding_eksempel", og er ikke poenget med dette eksempelarket.
# <details>

# %% [markdown]
# #### Kjør koden under. Output genereres, men blir ikke vist.

# %%
# %%capture
### Gjemt. Dobbeltklikk på den blå søylen til ventre for cellen for å åpne opp.
# Skriver ut instruksen
instruks = avrunding.print_instruks_konverter_dtypes()

# Lager mappingen
dtype_mapping = {
        "heltall":              ["personer"],
        "desimaltall_1_des":    [],
        "desimaltall_2_des":    [],
        "stringvar":            ["periode", "kommuneregion", "alder", "kjonn"],
        "bool_var":             [],
    }

# %%capture
# Utfører avrundingen/konverteringen
folkemengde_kommune_2024, dtypes = avrunding.konverter_dtypes(folkemengde_kommune_2024, dtype_mapping)


# %% [markdown]
# <details>
# <summary><b>Dette er også beskrevet andre steder og er ikke viktig her. (Men klikk for å vise)</b></summary>
# ## Gjemt. Dobbeltklikk på den blå søylen til venstre for cellen for å åpne opp.
# ## Her summerer vi opp de ettårige aldersgruppene til aggregerte KLASS-aldersgrupperinger.
# #### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# #### Vi trenger å hente ned en manuelt laget mappingfil som viser hvordan undergruppene settes sammen. Denne lagres som "hierarki_path".
# #### Deretter bruker vi funksjonen "summere_til_aldersgrupperinger" til å utføre operasjonen.
# #### Funksjonen ligger i en mappe som også heter "summere_til_aldersgrupperinger".
# #### Argumentene i parentesen er først det opprinnelige datasettet, og deretter mappingfilen, så funksjonen blir seende slik ut: 
# ##### summere_til_aldersgrupperinger.summere_til_aldersgrupperinger(folkemengde_kommune_2024, hierarki_path)
# #### De tre objektene til venstre for likhetstegnet, her "rename_variabel", "groupby_variable" og "df_sum_med_kjonn" er outputen funksjonen genererer. Hva de kalles er uviktig, men rekkefølgen betyr noe.
# #### Det første objektet viser til klassifikasjonsvariabelen "alder" (aldersvariabelen må hete "alder") som det SKAL summeres over, og det andre viser til de øvrige klassifikasjonsvariablene det IKKE skal summeres over.
# #### Det tredje objektet er det endelige datasettet, som er det opprinnelige datasettet pluss KOSTRA-aldersgrupperingene. Funksjonen sørger for å spytte ut det endelige datasettet.
# <details>

# %% [markdown]
# #### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# %% [markdown]
# <details>
# <summary><b>Dette er også beskrevet andre steder og er ikke viktig her. (Men klikk for å vise)</b></summary>
# ## Gjemt. Dobbeltklikk på den blå søylen til venstre for cellen for å åpne opp.
# #### 1) I den første måten å gjøre det på må du angi manuelt klassifikasjonsvariablene i datasettet som behandles. 
# #### Vi må hente ned en fil som mapper de ettårige aldersgruppene i KOSTRA-aldersgrupperingene. Vi kaller denne "hierarki_path" her.
# #### Datasettet som behandles er IKKE det opprinnelige datasettet alene, men et datasett slått sammen av det opprinnelige og mappingdatasettet "hierarki_path". 
# #### Du vil se når funksjonen kjøres at "to" også er identifisert som en variabel. Den må også angis som klassifikasjonsvariabel i tekstfeltet. I dette tilfellet blir klassifikasjonsvariablene du skal angi i tekstfeltet (uten anførselstegn og adskilt med komma):  kjonn, alder, to.
# <details>

# %% [markdown]
# <details>
# <summary><b>Dette er også beskrevet andre steder og er ikke viktig her. (Men klikk for å vise)</b></summary>
# ## Gjemt. Dobbeltklikk på den blå søylen til venstre for cellen for å åpne opp.
# #### 2) Du kan alternativt forhåndsdefinere klassifikasjonsvariablene slik at slipper å skrive inn i tekstfeltet.
# #### Vi må hente ned en fil som mapper de ettårige aldersgruppene i KOSTRA-aldersgrupperingene. Vi kaller denne “hierarki_path” her.
# #### Siden tekstfeltet etterspør klassifikasjonsvariablene utover periode og region, legger vi inn "kjonn", "alder" og "to" (ikke glem "to") inn i det forhåndsdefinerte objektet:
# ##### predefined_input = "kjonn, alder, to"
# <details>

# %% [markdown]
# #### Kjør koden under. Output genereres, men blir ikke vist.

# %%
# %%capture
hierarki_path = "gs://ssb-off-fin-data-produkt-prod/befolkning/_config/mapping_aldershierarki.parquet"

predefined_input = "kjonn, alder, to"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    rename_variabel, groupby_variable, df_sum_med_kjonn = summere_til_aldersgrupperinger.summere_til_aldersgrupperinger(folkemengde_kommune_2024, hierarki_path)

# %% [markdown]
# ### Her er funksjonen som skal gjennomgås.
# ### Summere over kjønnene

# %% [markdown]
# ### Som vist andre steder, kan dette gjøres på to måter, enten ved å føre klassifikasjonsvariablene manuelt inn i tekstfeltet, eller ved å forhåndsdefinere dem.
# ### I det første eksemplet fører vi dem inn manuelt. Funksjonen identifiserer ['periode', 'kommuneregion', 'kjonn', 'alder', 'personer']. "kjonn" og "alder" de klassifikasjonsvasiablene som kommer i tillegg til de faste variablene for periode og region. Så da fører vi det inn i tekstfeltet, uten anførselstegn og med komma mellom.
# ### Du vil se i det genererte datasettet at alderskolonnen har forsvunnet, for nå er kjønnene summert opp.

# %%
# Kjører funksjonen. df_sum_kjonn er det endelige datasettet som genereres.
df_sum_kjonn = summere_kjonn.summere_over_kjonn(df_sum_med_kjonn)
# Viser det genererte datasettet.
display(df_sum_kjonn)

# %% [markdown]
# #### I dette eksemplet forhåndsdefinerer vi klassifikasjonsvariablene vi ellers hadde måttet føre inn i tekstfeltet. 
# #### Vi får det samme resultatet som i metoden over.

# %%
predefined_input = "kjonn, alder"

with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
    df_sum_kjonn = summere_kjonn.summere_over_kjonn(df_sum_med_kjonn)

display(df_sum_kjonn)
