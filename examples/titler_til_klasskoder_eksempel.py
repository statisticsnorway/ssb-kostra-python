# # I dette eksempelarket ser vi på hvordan vi fester KLASS-kodenavn på KLASS-koder.

# I KOMPIS skjer dette automatisk så lenge KLASS-koden er gyldig. Her må vi gjøre det selv.
#
# Vi har to funksjoner som gjør dette for deg, én enkel en for regionene, og én litt mer omfattende en for alle klassifikasjonsvariable i KLASS.
#
# ssb-kostra-python/src/ssb_kostra_python/**mapping_regionsnavn.py** fester regionsnavn på klasskodene dine automatisk så lenge regionskolonnen heter **bydelsregion**, **kommuneregion** eller **fylkesregion**.
#
# ssb-kostra-python/src/ssb_kostra_python/**titler_til_klasskoder.py** fester KLASS-kodenavn på en hvilken som helst klassifikasjonsvariabel. Du kan gjøre dette for flere klassifikasjonsvariable samtidig. Men du må selv sørge for å **angi klass-id** for variabelen/variablene.
#
# Vi laster dem inn med `from ssb_kostra_python.titler_til_klasskoder import (kodelister_navn, mapping_regionsnavn)`.

# Laster ned nødvendige pakker
INPUT_PATCH_TARGET = "builtins.input"
import duckdb
import pandas as pd
from fagfunksjoner import latest_version_path
from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python.hjelpefunksjoner import format_fil
from ssb_kostra_python.titler_til_klasskoder import kodelister_navn
from ssb_kostra_python.titler_til_klasskoder import mapping_regionsnavn

# ## Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.

# Dataene kommer fra delt-bøtten til seksjon for befolkning.
# Vi må bearbeide dataene litt slik at de blir likere dataene slik vi kjenner dem i KOMPIS.
# Om du ønsker å hente data for et annet år, kan du bare sette statistikkaar til noe annet, slik: `statistikkaar=20XX`.

# +
statistikkaar = 2024
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
# -

# ## I tabellen over ser vi at alder ikke er på tresifret format. Derfor er ingen av kodene i denne kolonnen gyldige.

# At kodene i alderskolonnen ikke er gyldige, betyr ikke at datasettet ikke kan bearbeides. Men vi formaterer filen først.
#
# Det finnes en funksjon `format_fil(uformatert_fil)` som formaterer variablene **periode**, **bydelsregion**, **kommuneregion**, **fylkesregion** og **alder**.
# Om variabelen du trenger å formatere ikke dekkes av denne funksjonen, kan du for eksempel gjøre det selv med:
# `df_formatert = df_uformatert.copy()`,
# `df_formatert[“variabel_som_formateres”] = df_formatert[“variabel_som_formateres”].astype(“string”).str.zfill(6)` <— i dette eksemplet er antall sifre satt til 6.

# Formaterer fil
df_folketall_kommuner_formatert = format_fil(df_folketall_kommuner)
display(df_folketall_kommuner_formatert)

# ## Vi ser at "alder" har fått tresifrede koder. Vi fester regionsnavn på kodene.

# Funksjonen ser etter **bydelsregion**, **kommuneregion** og **fylkesregion**.

df_kommuner_regionsnavn = mapping_regionsnavn(df_folketall_kommuner_formatert)
display(df_kommuner_regionsnavn)

# Om vi ønsker å feste kodenavn på andre klassifikasjonsvariable enn regionene, kan vi bruke funksjonen under.

# Her må vi selv angi **hvilken variabel det gjelder, klass-id som hører til og hva den nye kolonnen med kodenavnene skal hete**. **select_level** settes til **1**. Det hele settes sammen til en såkalt mapping.
#
# Deretter kjøres funksjonen. Funksjonen genererer to resultater. Det første er det nye datasettet. Det andre er et sammendrag over variablene som er behandlet.
#
# Først lager du mappingen **`mapping_klassifikasjonsvariable =`** .
#
# ```mapping_klassifikasjonsvariable =
# [{“code_col”: “kommuneregion”,
# “klass_id”: 231,
# “name_col_out”: “kommuneregion_navn”,
# “select_level”: 1},
# {“code_col”: “alder”,
# “klass_id”: 248,
# “name_col_out”: “alder_navn”,
# “select_level”: 1},]
# ```
#
# I mappingen over er variabelen **kommuneregion** koplet til kodeliste **231** i KLASS. Variabelen **alder** er koplet kodeliste **248**. Kodeliste **2** hører til **kjonn**.
#
# Så kommer selve funksjonen (to resultater) **`df_med_kodenavn, sammendrag =`** .
#
# ```df_med_kodenavn, sammendrag =
# titler_til_klasskoder.kodelister_navn(df_folketall_kommuner_formatert,
# mappings=mapping_klassifikasjonsvariable,
# language=“nb”,
# include_future=True,
# verbose=True,)```

# +
mapping_klassifikasjonsvariable = [
    {
        "code_col": "kommuneregion",
        "klass_id": 231,
        "name_col_out": "kommuneregion_navn",
        "select_level": 1,
    },
    {
        "code_col": "alder",
        "klass_id": 248,
        "name_col_out": "alder_navn",
        "select_level": 1,
    },
    {
        "code_col": "kjonn",
        "klass_id": 2,
        "name_col_out": "kjonn_navn",
        "select_level": 1,
    },
]

# df_kommuner_med_kodenavn, sammendrag = titler_til_klasskoder.kodelister_navn(
df_kommuner_med_kodenavn, sammendrag = kodelister_navn(
    df_folketall_kommuner_formatert,
    mappings=mapping_klassifikasjonsvariable,
    language="nb",
    include_future=True,
    verbose=True,
)

display(df_kommuner_med_kodenavn)
# -

# # Test-eksempel med et lite og fiktivt testdatasett

# Om du ikke har tilgang til delt-bøtten til seksjon for befolkning, kan du kjøre dette. Vi legger med vilje til noen ugyldige koder for klassifikasjonsvariablene **kommuneregion**, **kjonn** og **alder** for å se hva som skjer med mappingen.

# +
statistikkaar = 2024
test_df = pd.DataFrame(
    {
        "kommuneregion": [
            "0301",
            "1103",
            "0301",
            "1103",
            "0301",
            "1103",
            "0301",
            "1103",
            "0301",
            "1103",
            "1299",
            "4601",
            "4699",
        ],
        "kjonn": ["1", "1", "1", "1", "1", "2", "2", "2", "2", "2", "1", "3", "4"],
        "alder": [13, 13, 13, 13, 45, 45, 13, 13, 13, 45, 36, 128, 199],
    }
)

test_df["periode"] = statistikkaar
test_df["personer"] = 1

test_df = test_df.groupby(
    ["periode", "kommuneregion", "kjonn", "alder"], as_index=False
)[["personer"]].sum()

display(test_df)
# -

# Vi må formatere dette datasettet, siden **alder** ikke er formatert til å være tresifrede koder.

# Formaterer fil
df_test_formatert = format_fil(test_df)
display(df_test_formatert)

# Kjører den enkle funksjonen som fester regionsnavn på kommuneregionskodene.

df_test_regionsnavn = mapping_regionsnavn(df_test_formatert)
display(df_test_regionsnavn)

# Vi ser at kommuneregion_navn returnerer **< NA >** for regionskodene **1299** og **4699**. Det er fordi disse kodene ikke er gyldige regionskoder for dette året.

# Vi kan også kjøre den andre funksjonen med manuell mapping.

# +
mapping_klassifikasjonsvariable = [
    {
        "code_col": "kommuneregion",
        "klass_id": 231,
        "name_col_out": "kommuneregion_navn",
        "select_level": 1,
    },
    {
        "code_col": "alder",
        "klass_id": 248,
        "name_col_out": "alder_navn",
        "select_level": 1,
    },
    {
        "code_col": "kjonn",
        "klass_id": 2,
        "name_col_out": "kjonn_navn",
        "select_level": 1,
    },
]

# df_kommuner_med_kodenavn, sammendrag = titler_til_klasskoder.kodelister_navn(
df_kommuner_med_kodenavn, sammendrag = kodelister_navn(
    df_test_formatert,
    mappings=mapping_klassifikasjonsvariable,
    language="nb",
    include_future=True,
    verbose=True,
)

display(df_kommuner_med_kodenavn)
# -

# Det endelige datasettet viser **< NA >** for kodenavnene med ugyldige koder. Sammendraget, som du også genererer til venstre for likhetstegnet (`df_kommuner_med_kodenavn, sammendrag = kodelister_navn(...)`, forteller deg hvilke det gjelder.
