# # Vi henter folkemengdedata for kommunene for 20XX.

from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python.hente_data_folkemengde import hente_data_folkemengde, hente_data_folkemengde_v2

# Til v2
import pandas as pd
from fagfunksjoner.fagfunksjoner_logger import logger
from klass import KlassClassification
from ssb_kostra_python import hjelpefunksjoner
from ssb_kostra_python import regionshierarki
INPUT_PATCH_TARGET = "builtins.input"
from unittest.mock import patch

# ## Vi henter kommunedata for 20XX.
# `datasett_reelle_data = hente_data_folkemengde(2024, "kommune")` og `datasett_reelle_data = hente_data_folkemengde(2024, "kommune", False)` er likestilt. Om du ikke angir at du skal lage testdata **(testdata = True)**, genererer funksjonen automatisk et datasett for år **20XX** som er laget av befolkningsdata for det samme året. **False** betyr altså det motsatte, du ønsker ikke å lage testdata, men heller reelle data.

datasett_reelle_data = hente_data_folkemengde(2024, "kommune")
display(datasett_reelle_data)

# #### Vi lager et testdatasett for 20XX ved bruk av en inputfil fra 20XX-1.

# `testdatasett = hente_data_folkemengde(2024, "kommune", testdata = True)` vil for eksempel generere et datasett for **2024**, men inputdataene vil komme fra en **2023**-fil. Når du setter **testdata = True**, bestemmer du at du ønsker å generere data for **t** med data fra **t-1**.

statistikkaar = 2026

testdatasett = hente_data_folkemengde(statistikkaar, "kommune", testdata=True)
display(testdatasett)

testdatasett_v2 = hente_data_folkemengde_v2(statistikkaar, "kommune", testdata=True)
display(testdatasett_v2)

# +
nokler = [
    "periode",
    "bydelsregion",
    "alder",
    "personer",
]

only_in_testdatasett = (
    testdatasett.merge(testdatasett_v2, how="left", indicator=True)
       .query("_merge == 'left_only'")
       .drop(columns="_merge")
)

only_in_testdatasett_v2 = (
    testdatasett_v2.merge(testdatasett, how="left", indicator=True)
       .query("_merge == 'left_only'")
       .drop(columns="_merge")
)

display(only_in_testdatasett)

display(only_in_testdatasett_v2)
