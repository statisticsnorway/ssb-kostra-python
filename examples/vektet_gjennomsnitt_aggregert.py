# +
INPUT_PATCH_TARGET = "builtins.input"
from unittest.mock import patch

import duckdb
from fagfunksjoner import latest_version_path
from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python.regionshierarki import hierarki
from ssb_kostra_python.titler_til_klasskoder import mapping_regionsnavn

from ssb_kostra_python.regionshierarki import _validate_and_normalize_region_col, _select_mapping, _postprocess_combined, vektet_gjennomsnitt_aggregerte_regioner
from fagfunksjoner.fagfunksjoner_logger import logger
from ssb_kostra_python import summere_kjonn
from typing import Any, cast
import pandas as pd
import numpy as np
# -

# ## Henter først inn et datasett vi kan jobbe med, som inneholder befolkning fordelt på region, kjønn og alder.

# Dataene kommer fra delt-bøtten til seksjon for befolkning. Vi må bearbeide dataene litt slik at de blir likere dataene slik vi kjenner dem i KOMPIS. Til bruker **duckdb** til dette. Husk å importere pakken.
# Om du ønsker å hente data for et annet år, kan du bare sette statistikkaar til noe annet, slik: statistikkaar=20XX.

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

# Når du kjører den nedenstående funksjonen, blir du bedt om å legge inn **øvrige klassifikasjonsvariable** utover **periode** og **region**. I dette datasettet har vi **kjonn** og **alder** i tillegg. Før dem inn i tekstfeltet, adskilt med komma.

# +
# Kjører funksjonen. folketall_bydeler_sum_kjonn er det endelige datasettet som genereres.
df_folketall_kommuner_sum_kjonn = summere_kjonn.summere_over_kjonn(df_folketall_kommuner)
# Viser det genererte datasettet. Du vil se at kolonnen for kjønn er borte, for nå er kjønnene summert opp.

print("\n")
print("ℹ️Det endelige datasettet heter her 'folketall_bydeler_sum_kjonn'.")
display(df_folketall_kommuner_sum_kjonn)
# -

# I den nedenstående funksjonen summerer vi "personer" over klassifikasjonsvariabelen "alder". Dette tilsvarer **sum(df_folketall_kommuner_sum_kjonn) along alder** i KOMPIS.

# +
df_folketall_kommuner_summert_over_kjonn = df_folketall_kommuner_sum_kjonn.groupby(
        ['periode', 'kommuneregion'], as_index=False, observed=True
    )['personer'].sum()

display(df_folketall_kommuner_summert_over_kjonn)
# -

# Siden poenget med dette eksempelarket er å aggregere på forskjellige måter, må vi generere noen fiktive data som det er naturlig å beregne uvektet og vektet gjennomsnitt av.
# Vi lager oss tre variable med tilfeldige verdier - **formuesskatt_prosent**, **voldsdom_prosent** og **antall_avisbud**.

# +
rng = np.random.default_rng(seed=42)

df_test = df_folketall_kommuner_summert_over_kjonn.copy()

df_test["formuesskatt_prosent"] = (
    rng.integers(0, 26, size=len(df_test)) / 10
)

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
# -

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
# Funksjonen returnerer to objekter. **resultat_visning** er datasettet etter behandling. **return_report** er en oppsummering av prosedyren. Oppsummeringen kan påpeke lite eller mye, avhengig av datasettet som ble brukt i funksjonen. Det ser vi på i neste eksempel. **inputfil** er naturligvis datasettet som skal behandles. **klassifikasjonsvariable** er klassifikasjonsvariablene utenom **periode** og **kommuneregion**. I dette datasettet finnes det ingen slike. Det spiller ingen rolle om du lar feltet stå tomt eller om du legger det inn. I **statistikkvariable** legger du inn statistikkvariablene. I **gjennomsnittsvariable** legger du inn variablene du ønsker å beregne **uvektet gjennomsnitt** av. I dette datasettet er det satt til **antall_avisbud**. I **vektede_variable** må du angi variablene som skal få **vektet gjennomsnitt** på aggregerte regioner, og du må også angi vektvariabelen. For eksempel beregnes det vektede gjennomsnittet av **formuesskatt_prosent** og **voldsdom_prosent** og vekten som brukes er **personer**. **aggregeringstype** kan stort sett stå tom, som ```aggregeringstype = None```, fordi funksjonen automatisk identifiserer regionsnivået på navnet **kommuneregion**, og aggregerer opp til EKA, EKG og EAK(UO). I visse tilfeller er det ønskelig å aggregere videre opp til fylkeskommuneregionsnivå, med EAFK(UO), og denne valgmuligheten er reservert disse tilfellene. 

# +
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


# + endofcell="--"
df_test_flere_nan = df_test.copy()

# # +
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
# -

display(
    df_test_flere_nan.loc[
        df_test_flere_nan["kommuneregion"].isin(
            ["5026", "4634", "4218"]
        )
    ]
)

display(df_test_flere_nan)
# --

# +
#resultat, rapport = vektet_gjennomsnitt_aggregerte_regioner(
# resultat, rapport= vektet_gjennomsnitt_aggregerte_regioner(
resultat= vektet_gjennomsnitt_aggregerte_regioner(
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
    return_report=False,
)


# display(return_report)
print("ℹ️Endelig datasett:")
display(resultat) 
# display(rapport)


# +
print("Før:")
display(df_test.dtypes)

print("Etter at NaN er lagt inn:")
display(df_test_flere_nan.dtypes)
# -


