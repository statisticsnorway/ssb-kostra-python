# ### I dette eksempelarket ser vi på hvordan vi fester KLASS-kodenavn på KLASS-koder.
# ### I KOMPIS skjer dette automatisk så lenge KLASS-koden er gyldig. Her må vi gjøre det selv.
# ### Vi har to funksjoner som gjør dette for deg.
# #### ssb-kostra-python/src/ssb_kostra_python/mapping_regionsnavn.py fester regionsnavn på klasskodene dine automatisk så lenge kolonnen heter "bydelsregion", "kommuneregion" eller "fylkesregion".
# #### ssb-kostra-python/src/ssb_kostra_python/titler_til_klasskoder.py fester KLASS-kodenavn på en hvilken som helst klassifikasjonsvariabel. Du kan gjøre dette for flere klassifikasjonsvariable samtidig. Men du må selv sørge for å angi klass-id for variabelen/variablene.
# ### Vi laster den inn med “from ssb_kostra_python.titler_til_klasskoder import (kodelister_navn, mapping_regionsnavn)”.

# Laster ned nødvendige pakker
INPUT_PATCH_TARGET = "builtins.input"
import duckdb
from fagfunksjoner import latest_version_path
from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python.hjelpefunksjoner import format_fil
from ssb_kostra_python.regionshierarki import hierarki
from ssb_kostra_python.titler_til_klasskoder import kodelister_navn
from ssb_kostra_python.titler_til_klasskoder import mapping_regionsnavn

# ### Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.
# ### Dataene kommer fra delt-bøtten til seksjon for befolkning.
# ### Vi må bearbeide dataene litt slik at de blir likere dataene slik vi kjenner dem i KOMPIS.
# ### Om du ønsker å hente data for et annet år, kan du bare sette statistikkaar til noe annet, slik: statistikkaar=20XX.

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

# ### I tabellen over ser vi at alder ikke er på tresifret format. Derfor er ingen av kodene i denne kolonnen gyldige, og vi må formatere filen først.
# ### Det finnes en funksjon format_fil(uformatert_fil) som formaterer variablene "periode", "bydelsregion", "kommuneregion", "fylkesregion" og "alder".
# ### Om variabelen du trenger å formatere ikke dekkes av funksjonen, kan du for eksempel gjøre det selv med:
# #### df_formatert = df_uformatert.copy()
# #### df_formatert["variabel_som_formateres"] = df_formatert["variabel_som_formateres"].astype("string").str.zfill(6) <--- i dette eksemplet er antall sifre satt til 6.

# Formaterer fil
df_folketall_kommuner_formatert = format_fil(df_folketall_kommuner)
display(df_folketall_kommuner_formatert)

# ### Vi ser at "alder" har fått tresifrede koder.

# ### Vi fester regionsnavn på kodene.
# ### Funksjonen ser etter "bydelsregion", "kommuneregion" og "fylkesregion".

df_kommuner_regionsnavn = mapping_regionsnavn(df_folketall_kommuner_formatert)
display(df_kommuner_regionsnavn)

# ### Om vi ønsker å feste kodenavn på andre klassifikasjonsvariable enn regionene, kan vi bruke funksjonen under.
# ### Her må vi selv angi hvilken variabel det gjelder, klass-id som hører til og hva den nye kolonnen med kodenavnene skal hete. select_level settes til 1. Det hele settes sammen til en såkalt mapping.
# ### Deretter kjøres funksjonen. Funksjonen genererer to resultater. Det første er det nye datasettet. Det andre er et sammendrag over variablene som er behandlet.
#
# #### mapping_klassifikasjonsvariable =
# #### [{"code_col": "kommuneregion", "klass_id": 231, "name_col_out": "kommuneregion_navn", "select_level": 1}, <--- mapping for "kommuneregion"
# #### {"code_col": "alder",       "klass_id": 248, "name_col_out": "alder_navn", "select_level": 1},] <--- mapping for "alder"
#
# #### df_med_kodenavn, sammendrag = titler_til_klasskoder.kodelister_navn(
# #### df_folketall_kommuner_formatert, <--- datasettet som skal behandles
# #### mappings=mapping_klassifikasjonsvariable, <--- mappingen du definerte i forkant
# #### language="nb", <--- språk, "nb" for bokmål
# #### include_future=True,
# #### verbose=True,)

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

# ### Om du ønsker å utføre en regionshierarki-operasjon på datasettet ovenfor, vil ikke hierarki-funksjonen henføre/aggregere kodene riktig på det aggregerte datasettet.
# ### Dette er fordi hierarki-funksjonen aggregerer kodene, men ikke navnene.
# ### Da er det best å fjerne kolonnene som inneholder kodenavn først, deretter utføre hierarki-operasjonen og etter det feste kodenavnene på nytt.

# Fjerner overflødige kolonner
df_kommuner_uten_kodenavn = df_kommuner_med_kodenavn.drop(
    columns=["kommuneregion_navn", "alder_navn"]
)
display(df_kommuner_uten_kodenavn)
# Utfører hierarkioperasjon på "kommuneregion"
folketall_kommuner_KOSTRA = hierarki(df_kommuner_uten_kodenavn)
display(folketall_kommuner_KOSTRA)

# ### Når de overflødige kolonnene er fjernet, og hierarki-funksjonen er utført, kan vi utføre mappingen på nytt.
# ### Denne gangen mapper vi også "kjonn", som har klass-id: 2.

# +
# Utfører mapping på nytt
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
df_kommuner_og_KOSTRA_med_kodenavn, sammendrag = kodelister_navn(
    folketall_kommuner_KOSTRA,
    mappings=mapping_klassifikasjonsvariable,
    language="nb",
    include_future=True,
    verbose=True,
)

display(df_kommuner_og_KOSTRA_med_kodenavn)
