# # Vi henter folkemengdedata for bydelene for 20XX.

from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python.hente_data_folkemengde import hente_data_folkemengde
from ssb_kostra_python.hente_data_folkemengde import hente_data_folkemengde_v2

# `datasett_reelle_data = hente_data_folkemengde(2024, "bydel")` og `datasett_reelle_data = hente_data_folkemengde(2024, "bydel", False)` er likestilt. Om du ikke angir at du skal lage testdata **(testdata = True)** eller ganske enkelt **(True)**, genererer funksjonen automatisk et datasett for år **20XX** som er laget av befolkningsdata for det samme året. False betyr altså det motsatte, du ønsker ikke å lage testdata, men heller reelle data.

datasett_reelle_data = hente_data_folkemengde(2024, "bydel", False)
display(datasett_reelle_data)

# ## Vi lager et testdatasett for 20XX ved bruk av en inputfil fra 20XX-1.

# `testdatasett = hente_data_folkemengde(2024, "bydel", True)` vil generere et datasett for **2024**, men inputdataene vil komme fra en **2023**-fil. Når du setter **testdata = True**, bestemmer du at du ønsker å generere data for **t** med data fra **t-1**.

statistikkaar = 2024

testdatasett = hente_data_folkemengde(statistikkaar, "bydel", testdata=True)
display(testdatasett)

testdatasett_v2 = hente_data_folkemengde_v2(statistikkaar, "bydel", testdata=True)
display(testdatasett_v2)

# +
# forskjeller = testdatasett.compare(testdatasett_v2)

# if forskjeller.empty:
#     print("✅ Datasettene er identiske.")
# else:
#     print("❌ Datasettene er forskjellige.")
#     display(forskjeller)

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
# -
