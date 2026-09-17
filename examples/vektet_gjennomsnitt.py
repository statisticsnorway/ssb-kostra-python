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

# +
# Kjører funksjonen. folketall_bydeler_sum_kjonn er det endelige datasettet som genereres.
df_folketall_kommuner_sum_kjonn = summere_kjonn.summere_over_kjonn(df_folketall_kommuner)
# Viser det genererte datasettet. Du vil se at kolonnen for kjønn er borte, for nå er kjønnene summert opp.

print("\n")
print("ℹ️Det endelige datasettet heter her 'folketall_bydeler_sum_kjonn'.")
display(df_folketall_kommuner_sum_kjonn)

# +
df_folketall_kommuner_summert_over_kjonn = df_folketall_kommuner_sum_kjonn.groupby(
        ['periode', 'kommuneregion'], as_index=False, observed=True
    )['personer'].sum()

display(df_folketall_kommuner_summert_over_kjonn)

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

# +
resultat_visning, return_report = vektet_gjennomsnitt_aggregerte_regioner(
    inputfil=df_test,
    klassifikasjonsvariable =[],
    statistikkvariable=[
        "personer",
        "formuesskatt_prosent",
        "voldsdom_prosent",
        'antall_avisbud'
    ],
    gjennomsnittsvariable = ['antall_avisbud'],
    vektede_variable={
        "formuesskatt_prosent": "personer",
        "voldsdom_prosent": "personer",
    },
    restore_original_dtype = True,
    decimals=1,
    return_report= True,
    vis_rapport=True,
)

display(resultat_visning) 
display(return_report)

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
resultat_visning, return_report = vektet_gjennomsnitt_aggregerte_regioner(
    inputfil=df_test_flere_nan,
    klassifikasjonsvariable= [],
    statistikkvariable=[
        "personer",
        "formuesskatt_prosent",
        "voldsdom_prosent",
        "antall_avisbud",
    ],
    gjennomsnittsvariable = ["antall_avisbud"],
    vektede_variable={
        "formuesskatt_prosent": "personer",
        "voldsdom_prosent": "personer",
    },
    decimals=2,
    return_report= True,
    vis_rapport=True,
)

# display(return_report)
print("ℹ️Endelig datasett:")
display(resultat_visning) 


# +
print("Før:")
display(df_test.dtypes)

print("Etter at NaN er lagt inn:")
display(df_test_flere_nan.dtypes)
# -


