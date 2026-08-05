# # Vi henter folkemengdedata for kommunene for 20XX.

# Til v2
from IPython.display import display

from ssb_kostra_python.hente_data_folkemengde import hente_data_folkemengde

INPUT_PATCH_TARGET = "builtins.input"

# ## Vi henter kommunedata for 20XX.
# `datasett_reelle_data = hente_data_folkemengde(2024, "kommune")`, `datasett_reelle_data = hente_data_folkemengde(2024, "kommune", False)` og `datasett_reelle_data = hente_data_folkemengde(2024, "kommune", testdata = False)` er likestilt. Om du ikke angir at du skal lage testdata **(testdata = True)**, genererer funksjonen automatisk et datasett for år **20XX** som er laget av befolkningsdata for det samme året. **False** betyr altså det motsatte, du ønsker ikke å lage testdata, men heller reelle data.

datasett_reelle_data = hente_data_folkemengde(2024, "kommune", False)
display(datasett_reelle_data)

# #### Vi lager et testdatasett for 20XX ved bruk av en inputfil fra 20XX-1.

# `testdatasett = hente_data_folkemengde(2024, "kommune", True)`, alternativt `testdatasett = hente_data_folkemengde(2024, "kommune", testdata = True)`, vil for eksempel generere et datasett for **2024**, men inputdataene vil komme fra en **2023**-fil. Når du setter **testdata = True** eller gannske enkelt **True**, bestemmer du at du ønsker å generere data for **t** med data fra **t-1**.

statistikkaar = 2024

testdatasett = hente_data_folkemengde(statistikkaar, "kommune", testdata=True)
display(testdatasett)
