from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python.hente_data_folkemengde import hente_data_folkemengde

# #### Vi henter bydelsdata for 20XX.
# #### datasett_reelle_data = hente_data_folkemengde(2024, "kommune") og datasett_reelle_data = hente_data_folkemengde(2024, "kommune", False) er likestilt. Om du ikke angir at du skal lage testdata (True), genererer funksjonen automatisk et datasett for år 20XX som er laget av befolkningsdata for det samme året. False betyr altså det motsatte, du ønsker ikke å lage testdata, men heller reelle data.

datasett_reelle_data = hente_data_folkemengde(2024, "kommune")
display(datasett_reelle_data)

# #### Vi lager et testdatasett for 20XX ved bruk av en inputfil fra 20XX-1.
# #### testdatasett = hente_data_folkemengde(2024, "kommune", True) vil generere et datasett for 2024, men inputdataene vil komme fra en 2023-fil.

testdatasett = hente_data_folkemengde(2024, "kommune", True)
display(testdatasett)
