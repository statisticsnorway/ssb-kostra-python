INPUT_PATCH_TARGET = "builtins.input"

import duckdb
import numpy as np
from fagfunksjoner import latest_version_path
from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python import summere_kjonn
from ssb_kostra_python.regionshierarki import vektet_gjennomsnitt_aggregerte_regioner

# ## Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.

# Dataene kommer fra delt-bøtten til seksjon for befolkning. Vi må bearbeide dataene litt slik at de blir likere dataene slik vi kjenner dem i KOMPIS. Vi bruker **duckdb** til dette. Husk å importere pakken.
# Om du ønsker å hente data for et annet år, kan du bare sette statistikkaar til noe annet, slik: statistikkaar=20XX.

# region
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
# endregion

# Når du kjører den nedenstående funksjonen, blir du bedt om å legge inn **øvrige klassifikasjonsvariable** utover **periode** og **region**. I dette datasettet har vi **kjonn** og **alder** i tillegg. Før dem inn i tekstfeltet, adskilt med komma.

# Kjører funksjonen. folketall_bydeler_sum_kjonn er det endelige datasettet som genereres.
df_folketall_kommuner_sum_kjonn = summere_kjonn.summere_over_kjonn(
    df_folketall_kommuner
)
# Viser det genererte datasettet. Du vil se at kolonnen for kjønn er borte, for nå er kjønnene summert opp.
print("\n")
print("ℹ️Det endelige datasettet heter her 'folketall_bydeler_sum_kjonn'.")
display(df_folketall_kommuner_sum_kjonn)

# I den nedenstående funksjonen summerer vi "personer" over klassifikasjonsvariabelen "alder". Dette tilsvarer **sum(df_folketall_kommuner_sum_kjonn) along alder** i KOMPIS.

# region
df_folketall_kommuner_summert_over_kjonn = df_folketall_kommuner_sum_kjonn.groupby(
    ["periode", "kommuneregion"], as_index=False, observed=True
)["personer"].sum()

display(df_folketall_kommuner_summert_over_kjonn)
# endregion

# Siden poenget med dette eksempelarket er å aggregere på forskjellige måter, må vi generere noen fiktive data som det er naturlig å beregne uvektet og vektet gjennomsnitt av.
# Vi lager oss tre variable med tilfeldige verdier - **formuesskatt_prosent**, **voldsdom_prosent** og **antall_avisbud**.

# region
rng = np.random.default_rng(seed=42)

df_test = df_folketall_kommuner_summert_over_kjonn.copy()

df_test["formuesskatt_prosent"] = rng.integers(0, 26, size=len(df_test)) / 10

df_test["voldsdom_prosent"] = rng.integers(
    0,
    11,
    size=len(df_test),
)

df_test["antall_avisbud"] = rng.integers(
    0,
    100,
    size=len(df_test),
)

display(df_test)
print("Før:")
display(df_test.dtypes)
# endregion

# ### Aggregering når datasettet ikke inneholder manglende verdier
#
# Funksjonen kan behandle statistikkvariablene på tre forskjellige måter når det dannes aggregerte regioner:
#
# 1. **Summering:** Statistikkvariable som ikke er oppgitt som gjennomsnittsvariable eller vektede variable, summeres.
# 2. **Uvektet gjennomsnitt:** Variable som oppgis i `gjennomsnittsvariable`, beregnes som vanlig aritmetisk gjennomsnitt.
# 3. **Vektet gjennomsnitt:** Variable som oppgis i `vektede_variable`, beregnes som vektet gjennomsnitt ved hjelp av den tilhørende vektvariabelen.
#
# I eksemplet nedenfor betyr dette at:
#
# - **personer** summeres.
# - **antall_avisbud** beregnes som et vanlig, uvektet gjennomsnitt.
# - **formuesskatt_prosent** beregnes som et vektet gjennomsnitt med **personer** som vekt.
# - **voldsdom_prosent** beregnes som et vektet gjennomsnitt med **personer** som vekt.
#
# For et vektet gjennomsnitt vil en kommune med høy verdi i **personer** få større betydning for resultatet enn en kommune med lav verdi i **personer**.
#
# Når alle observasjonene har gyldige verdier og vekter, inngår alle observasjonene i beregningen. Funksjonen skriver da ut:
#
# **✅ Ingen utelatte observasjoner eller aggregerte problemer å rapportere.**

# Her kjører vi funksjonen.
#
# **`resultat, rapport =`**
#
# ```    inputfil=df_test,
#     klassifikasjonsvariable=[],
#     statistikkvariable=[
#         "personer",
#         "formuesskatt_prosent",
#         "voldsdom_prosent",
#         "antall_avisbud",
#     ],
#     vektede_variable={
#         "formuesskatt_prosent": "personer",
#         "voldsdom_prosent": "personer",
#     },
#     gjennomsnittsvariable=["antall_avisbud"],
#     aggregeringstype=None,
#     add_region_names=True,
#     return_report=True,
# )
# ```
# - **inputfil** er datasettet som skal behandles.
# - **klassifikasjonsvariable** inneholder eventuelle klassifikasjonsvariable utover **periode** og regionsvariabelen. Disse to identifiseres automatisk. I dette datasettet har vi ingen ytterligere klassifikasjonsvariable, og listen kan derfor stå tom. Du **kan** også oppgi **periode-** og **regionsvariabelen** eksplisitt, slik vi gjør i eksemplet nedenfor. Dette er ikke nødvendig, men det er heller ikke noe problem.
# - **statistikkvariable** angir statistikkvariablene som skal behandles.
# - **gjennomsnittsvariable** angir variablene som skal beregnes som vanlig, uvektet gjennomsnitt. Her gjelder dette **antall_avisbud**.
# - **vektede_variable** angir både hvilke variable som skal beregnes som vektet gjennomsnitt, og hvilken variabel som skal brukes som vekt. Her beregnes **formuesskatt_prosent** og **voldsdom_prosent** med **personer** som vekt.
# - Statistikkvariable som ikke er oppgitt i **gjennomsnittsvariable** eller **vektede_variable**, summeres. Her gjelder dette **personer**.
# - **aggregeringstype** kan vanligvis stå som `None`. Funksjonen identifiserer da regionsnivået fra regionsvariabelen. Dersom kommuner i stedet skal aggregeres til fylkeskommuneregioner, brukes `aggregeringstype="kommune_til_fylkeskommune"`.
# - **add_region_names=True** legger regionsnavn til resultatet.
# - **return_report=False** betyr at bare det ferdige datasettet returneres. Dersom denne settes til `True`, returneres også en rapport med informasjon om blant annet utelatte observasjoner og problemer på aggregert nivå.

resultat = vektet_gjennomsnitt_aggregerte_regioner(
    inputfil=df_test,
    klassifikasjonsvariable=["periode", "kommuneregion"],
    statistikkvariable=[
        "personer",
        "formuesskatt_prosent",
        "voldsdom_prosent",
        "antall_avisbud",
    ],
    vektede_variable={
        "formuesskatt_prosent": "personer",
        "voldsdom_prosent": "personer",
    },
    gjennomsnittsvariable=["antall_avisbud"],
    aggregeringstype=None,
    add_region_names=True,
    return_report=False,
)

display(resultat)


# ### Aggregering når datasettet inneholder manglende verdier
#
# Manglende verdier (`NaN`) krever litt ekstra oppmerksomhet, særlig ved beregning av vektede gjennomsnitt.
#
# For et **vektet gjennomsnitt** må både verdien som skal gjennomsnittsberegnes og den tilhørende vekten finnes. Dersom én av dem mangler, utelates den aktuelle observasjonen fra beregningen av dette vektede gjennomsnittet.
#
# Det betyr:
#
# - **Verdi mangler, vekt finnes:** Observasjonen utelates. Vekten tas heller ikke med i nevneren.
# - **Verdi finnes, vekt mangler:** Observasjonen utelates.
# - **Både verdi og vekt mangler:** Observasjonen utelates.
# - **Både verdi og vekt finnes:** Observasjonen inngår som normalt.
#
# En manglende verdi (`NaN`) tolkes **ikke** som 0. Dersom en tom celle i kildedataene egentlig betyr 0, må denne derfor erstattes med 0 før funksjonen kjøres.
#
# Funksjonen gir advarsler når observasjoner utelates og viser hvilke observasjoner dette gjelder. Det vektede gjennomsnittet beregnes likevel på grunnlag av de gjenværende observasjonene som har både gyldig verdi og gyldig vekt.

# region
df_test_flere_nan = df_test.copy()

# 5026: target mangler, men vekten finnes
df_test_flere_nan.loc[
    df_test_flere_nan["kommuneregion"] == "5026",
    "formuesskatt_prosent",
] = np.nan

# 4634: target finnes, men vekten mangler
df_test_flere_nan.loc[
    df_test_flere_nan["kommuneregion"] == "4634",
    "personer",
] = np.nan

# 4218: både target og vekt mangler
df_test_flere_nan.loc[
    df_test_flere_nan["kommuneregion"] == "4218",
    ["formuesskatt_prosent", "personer"],
] = np.nan

# 0301: både target og vekt mangler.
# Oslo utgjør alene EKA03 og EKG13, slik at disse ikke kan beregnes.
df_test_flere_nan.loc[
    df_test_flere_nan["kommuneregion"] == "0301",
    ["formuesskatt_prosent", "personer"],
] = np.nan

display(
    df_test_flere_nan.loc[
        df_test_flere_nan["kommuneregion"].isin(["5026", "4634", "4218", "0301"])
    ]
)
# endregion

display(df_test_flere_nan)

# Vi har nå laget tre forskjellige typer mangler som påvirker beregningen av vektede gjennomsnitt, fordelt på fire kommuner:
#
# - Kommune **5026** mangler `formuesskatt_prosent`, men har `personer`. Kommunen utelates derfor fra det vektede gjennomsnittet av `formuesskatt_prosent`, og kommunens `personer` brukes heller ikke som vekt i denne beregningen.
# - Kommune **4634** har `formuesskatt_prosent`, men mangler `personer`. Kommunen kan derfor ikke inngå i det vektede gjennomsnittet.
# - Kommune **4218** mangler både `formuesskatt_prosent` og `personer` og utelates derfor også.
# - Kommune **0301** mangler både `formuesskatt_prosent` og `personer` og utelates av samme grunn.
#
# De øvrige gyldige observasjonene brukes fortsatt. En enkelt manglende observasjon gjør altså ikke automatisk hele det aggregerte gjennomsnittet til `NaN`.

# **Merk:** `personer` brukes både som en statistikkvariabel som summeres og som vekt for de vektede gjennomsnittene. Dersom `personer` mangler for én kommune, utelates kommunen fra det aktuelle vektede gjennomsnittet. Ved summeringen av selve variabelen `personer` summeres derimot de øvrige gyldige verdiene. Resultatet blir dermed en ufullstendig sum dersom enkelte verdier mangler, ikke automatisk `NaN`.
#
# Funksjonen varsler om manglende verdier slik at slike resultater kan vurderes før de brukes videre.

# #### Når manglende observasjoner gjør at et aggregert resultat ikke kan beregnes
#
# Som regel kan et vektet gjennomsnitt fortsatt beregnes selv om enkelte observasjoner
# utelates. Funksjonen bruker da de gjenværende observasjonene som har både gyldig verdi
# og gyldig vekt.
#
# Det finnes imidlertid tilfeller der det ikke er noen gyldige vekter igjen for en
# aggregert region. Da kan det vektede gjennomsnittet ikke beregnes.
#
# Dette demonstreres her med kommune **0301**. Oslo utgjør alene enkelte aggregerte
# regioner på aggregert nivå, **EKA03** og **EKG13**. Når `personer` mangler for "0301", finnes det derfor ingen gyldig vekt for disse
# regionene. Det vektede gjennomsnittet kan da ikke beregnes, og resultatet blir `NaN`.
#
# Funksjonen rapporterer slike tilfeller som **aggregerte problemer**. Disse må skilles fra
# **utelatte observasjoner**:
#
# - **Utelatte observasjoner** betyr at én eller flere observasjoner ikke kunne brukes i
#   beregningen. Det aggregerte resultatet kan likevel ofte beregnes fra de gjenværende
#   observasjonene.
# - **Aggregerte problemer** betyr at det ikke finnes tilstrekkelig grunnlag for å beregne
#   resultatet for den aktuelle aggregerte regionen, for eksempel fordi summen av gyldige
#   vekter er 0 eller mangler.

# # +
resultat, rapport = vektet_gjennomsnitt_aggregerte_regioner(
    inputfil=df_test_flere_nan,
    klassifikasjonsvariable=[],
    statistikkvariable=[
        "personer",
        "formuesskatt_prosent",
        "voldsdom_prosent",
        "antall_avisbud",
    ],
    vektede_variable={
        "formuesskatt_prosent": "personer",
        "voldsdom_prosent": "personer",
    },
    gjennomsnittsvariable=["antall_avisbud"],
    aggregeringstype=None,
    add_region_names=True,
    return_report=True,
)

print("ℹ️Endelig datasett:")
display(resultat)


# ### Hente ut rapporten som egne tabeller
#
# Når `return_report=True`, returnerer funksjonen både det ferdige datasettet og et
# rapportobjekt:
#
# `resultat, rapport = ...`
#
# Rapporten inneholder blant annet to tabeller som kan være nyttige ved kontroll av
# resultatet:
#
# - `rapport["utelatte_observasjoner"]` viser enkeltobservasjoner som ikke kunne inngå i
#   beregningen av et vektet gjennomsnitt fordi verdi, vekt eller begge manglet.
# - `rapport["aggregerte_problemer"]` viser aggregerte regioner der et gjennomsnitt ikke
#   kunne beregnes. For vektede gjennomsnitt skjer dette når summen av gyldige vekter er
#   0 eller mangler.
#
# Tabellene kan hentes ut og behandles som vanlige DataFrames:

utelatte_observasjoner = rapport["utelatte_observasjoner"]
aggregerte_problemer = rapport["aggregerte_problemer"]
display(utelatte_observasjoner)
display(aggregerte_problemer)

print("Før:")
display(df_test.dtypes)

print("Etter at NaN er lagt inn:")
display(df_test_flere_nan.dtypes)
