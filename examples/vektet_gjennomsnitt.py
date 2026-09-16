# +
INPUT_PATCH_TARGET = "builtins.input"
from unittest.mock import patch

import duckdb
from fagfunksjoner import latest_version_path
from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python.regionshierarki import hierarki
from ssb_kostra_python.titler_til_klasskoder import mapping_regionsnavn



from ssb_kostra_python.regionshierarki import _validate_and_normalize_region_col, _select_mapping, _postprocess_combined
from fagfunksjoner.fagfunksjoner_logger import logger
from ssb_kostra_python import summere_kjonn
from typing import Any, cast
import pandas as pd

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
import numpy as np

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

display(df_test)

# +
resultat, rapport = (
    vektet_gjennomsnitt_aggregerte_regioner(
        inputfil=df_test,
        statistikkvariable=[
            "personer",
            "formuesskatt_prosent",
            "voldsdom_prosent",
        ],
        vektede_variable={
            "formuesskatt_prosent": "personer",
            "voldsdom_prosent": "personer",
        },
        decimals=2,
        return_report=True,
    )
)

display(rapport)
display(resultat)

# +
mappingfil, join_col, replace_col, post_filter, rename_cols = (
    _select_mapping(
        aggregeringstype="kommune_til_landet",
        region_col="kommuneregion",
        periode="2024",
    )
)

display(mappingfil)

# +
kommuner_EKG14 = (
    mappingfil.loc[
        mappingfil["to"] == "EKG14",
        "from",
    ]
    .astype(str)
    .str.zfill(4)
    .tolist()
)

print(kommuner_EKG14)

# +
df_EKG14 = df_test.loc[
    df_test["kommuneregion"].isin(kommuner_EKG14)
].copy()

display(df_EKG14)
# -

print(df_EKG14["personer"].sum())

# +
teller_formuesskatt = (
    df_EKG14["formuesskatt_prosent"]
    * df_EKG14["personer"]
).sum()

print(teller_formuesskatt)

# +
nevner_formuesskatt = df_EKG14["personer"].sum()

print(nevner_formuesskatt)

# +
vektet_formuesskatt = (
    teller_formuesskatt / nevner_formuesskatt
)

print(vektet_formuesskatt)
print(round(vektet_formuesskatt, 2))

# +
teller_voldsdom = (
    df_EKG14["voldsdom_prosent"]
    * df_EKG14["personer"]
).sum()

nevner_voldsdom = df_EKG14["personer"].sum()

vektet_voldsdom = teller_voldsdom / nevner_voldsdom

print(vektet_voldsdom)
print(round(vektet_voldsdom, 2))

# +
df_test_nan = df_test.copy()

df_test_nan.loc[
    df_test_nan["kommuneregion"] == "5026",
    "formuesskatt_prosent",
] = np.nan
# -

display(
    df_test_nan.loc[
        df_test_nan["kommuneregion"] == "5026"
    ]
)

resultat_nan, rapport_nan = (
    vektet_gjennomsnitt_aggregerte_regioner(
        inputfil=df_test_nan,
        statistikkvariable=[
            "personer",
            "formuesskatt_prosent",
            "voldsdom_prosent",
        ],
        vektede_variable={
            "formuesskatt_prosent": "personer",
            "voldsdom_prosent": "personer",
        },
        decimals=2,
        return_report=True,
    )
)

# +
display(
    resultat_nan.loc[
        resultat_nan["kommuneregion"] == "EKG14"
    ]
)

display(rapport_nan["utelatte_observasjoner"])

# +
df_test_missing_weight = df_test.copy()

df_test_missing_weight.loc[
    df_test_missing_weight["kommuneregion"] == "4634",
    "personer",
] = np.nan

display(
    df_test_missing_weight.loc[
        df_test_missing_weight["kommuneregion"] == "4634"
    ]
)
# -

resultat_missing_weight, rapport_missing_weight = (
    vektet_gjennomsnitt_aggregerte_regioner(
        inputfil=df_test_missing_weight,
        statistikkvariable=[
            "personer",
            "formuesskatt_prosent",
            "voldsdom_prosent",
        ],
        vektede_variable={
            "formuesskatt_prosent": "personer",
            "voldsdom_prosent": "personer",
        },
        decimals=2,
        return_report=True,
    )
)

# +
display(
    resultat_missing_weight.loc[
        resultat_missing_weight["kommuneregion"] == "EKG14"
    ]
)

display(
    rapport_missing_weight["utelatte_observasjoner"]
)
# -

df_test_flere_nan = df_test.copy()

# +
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

# +
resultat, rapport_flere_nan = (
    vektet_gjennomsnitt_aggregerte_regioner(
        inputfil=df_test_flere_nan,
        statistikkvariable=[
            "personer",
            "formuesskatt_prosent",
            "voldsdom_prosent",
        ],
        vektede_variable={
            "formuesskatt_prosent": "personer",
            "voldsdom_prosent": "personer",
        },
        decimals=2,
        return_report=True,
    )
)

display(rapport_flere_nan)
display(resultat)
# -

display(rapport_flere_nan["utelatte_observasjoner"])

display(rapport_flere_nan["utelatte_observasjoner"])


def vektet_gjennomsnitt_aggregerte_regioner(
    inputfil: pd.DataFrame,
    klassifikasjonsvariable: list[str] | None = None,
    statistikkvariable: list[str] | None = None,
    vektede_variable: dict[str, str] | None = None,
    gjennomsnittsvariable: list[str] | None = None,
    aggregeringstype: str | None = None,
    decimals: int | None = None,
    restore_original_dtype: bool = False,
    add_region_names: bool = False,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Aggreger regioner med sum, gjennomsnitt og vektet gjennomsnitt.

    Funksjonen aggregerer et datasett på kommune-, fylkeskommune- eller
    bydelsnivå til tilhørende aggregerte regioner.

    Statistikkvariablene kan behandles på tre måter:

    1. Variable oppgitt i ``vektede_variable`` beregnes som vektet
       gjennomsnitt.
    2. Variable oppgitt i ``gjennomsnittsvariable`` beregnes som vanlig
       aritmetisk gjennomsnitt.
    3. Øvrige statistikkvariable summeres.

    ``vektede_variable`` er en mapping mellom variabelen som skal
    gjennomsnittsberegnes og variabelen den skal vektes med, for eksempel::

        {
            "levekarsindeks": "personer",
            "saksbehandlingstid": "antall_saker",
        }

    Manglende verdier
    ------------------
    Dersom verdien som skal gjennomsnittsberegnes mangler, utelates både
    verdien og tilhørende vekt fra beregningen av det aktuelle gjennomsnittet.

    Dersom vekten mangler, utelates observasjonen tilsvarende.

    Manglende verdi tolkes aldri automatisk som 0. Dersom en manglende verdi
    i realiteten betyr 0, må inputdatasettet korrigeres før funksjonen kjøres.

    Null er en gyldig verdi både for statistikkvariabelen og vekten.
    Negative vekter er ikke tillatt.

    Klassifikasjons- og statistikkvariable
    ---------------------------------------
    ``periode`` og regionsvariabelen registreres automatisk som
    klassifikasjonsvariable.

    ``klassifikasjonsvariable`` brukes derfor bare til eventuelle ytterligere
    klassifikasjonsvariable, for eksempel ``["kjonn", "alder"]``.

    ``statistikkvariable`` kan oppgis eksplisitt. Dersom parameteren er
    ``None``, defineres statistikkvariablene som alle kolonner som ikke er
    klassifikasjonsvariable.

    Variable i ``vektede_variable`` og ``gjennomsnittsvariable`` er
    statistikkvariable med en særskilt aggregeringsmetode. Alle øvrige
    statistikkvariable summeres.

    Parametre
    ---------
    inputfil:
        DataFrame som skal aggregeres.

    klassifikasjonsvariable:
        Klassifikasjonsvariable utover periode- og regionsvariabelen.

    statistikkvariable:
        Statistikkvariable i datasettet. Dersom ``None``, utledes disse
        automatisk.

    vektede_variable:
        Mapping ``{variabel: vektvariabel}``.

    gjennomsnittsvariable:
        Variable som skal beregnes som vanlig, uvektet gjennomsnitt.

    aggregeringstype:
        Samme regionale aggregeringstyper som brukes av ``hierarki``.

    decimals:
        Antall desimaler for beregnede gjennomsnitt. Dersom ``None``,
        beholdes full beregnet presisjon.

    restore_original_dtype:
        Dersom ``True``, forsøkes opprinnelig dtype gjenopprettet for
        gjennomsnittsvariablene. Standard er ``False`` fordi et gjennomsnitt
        av heltall kan være et desimaltall.

    add_region_names:
        Dersom ``True``, legges regionsnavn til gjennom eksisterende
        etterbehandling.

    return_report:
        Dersom ``True``, returneres også en rapport med variabelbehandling,
        utelatte observasjoner og dtype-informasjon.

    Returnerer
    ----------
    pandas.DataFrame
        Opprinnelige rader og aggregerte rader.

    eller

    tuple[pandas.DataFrame, dict[str, Any]]
        DataFrame og rapport dersom ``return_report=True``.
    """
    df = inputfil.copy()

    klassifikasjonsvariable = klassifikasjonsvariable or []
    vektede_variable = vektede_variable or {}
    gjennomsnittsvariable = gjennomsnittsvariable or []

    region_names = [
        "kommuneregion_navn",
        "fylkesregion_navn",
        "bydelsregion_navn",
    ]

    if any(col in df.columns for col in region_names):
        logger.info(
            "Datasettet inneholder regionsnavn. "
            "Regionsnavnkolonnene fjernes før aggregering."
        )
        df = df.drop(columns=region_names, errors="ignore")

    if "periode" not in df.columns:
        raise KeyError("Datasettet mangler obligatorisk kolonne 'periode'.")

    if df["periode"].nunique() > 1:
        raise KeyError("Mer enn 1 periode i datasettet")

    df["periode"] = df["periode"].astype(str)
    periode = df["periode"].unique()[0]

    region_col, df = _validate_and_normalize_region_col(df)

    # ---------------------------------------------------------
    # Klassifikasjonsvariable
    # ---------------------------------------------------------
    manglende_klassifikasjonsvariable = [
        col for col in klassifikasjonsvariable if col not in df.columns
    ]
    if manglende_klassifikasjonsvariable:
        raise KeyError(
            "Følgende klassifikasjonsvariable finnes ikke i datasettet: "
            f"{manglende_klassifikasjonsvariable}"
        )

    faste_klassifikasjonsvariable = ["periode", region_col]

    alle_klassifikasjonsvariable = list(
        dict.fromkeys(
            faste_klassifikasjonsvariable + klassifikasjonsvariable
        )
    )

    # ---------------------------------------------------------
    # Statistikkvariable
    # ---------------------------------------------------------
    if statistikkvariable is None:
        statistikkvariable = [
            col
            for col in df.columns
            if col not in alle_klassifikasjonsvariable
        ]
        logger.info(
            "ℹ️ Statistikkvariable er ikke oppgitt eksplisitt. "
            "De utledes derfor som alle variable som ikke er "
            "klassifikasjonsvariable."
        )
    else:
        statistikkvariable = list(dict.fromkeys(statistikkvariable))

        manglende_statistikkvariable = [
            col for col in statistikkvariable if col not in df.columns
        ]
        if manglende_statistikkvariable:
            raise KeyError(
                "Følgende statistikkvariable finnes ikke i datasettet: "
                f"{manglende_statistikkvariable}"
            )

        overlapp = sorted(
            set(statistikkvariable) & set(alle_klassifikasjonsvariable)
        )
        if overlapp:
            raise ValueError(
                "Følgende variable er oppgitt både som klassifikasjonsvariable "
                f"og statistikkvariable: {overlapp}"
            )

        klassifiserte = set(alle_klassifikasjonsvariable) | set(
            statistikkvariable
        )
        uklassifiserte = [
            col for col in df.columns if col not in klassifiserte
        ]

        if uklassifiserte:
            raise ValueError(
                "Følgende variable er verken definert som "
                "klassifikasjonsvariable eller statistikkvariable: "
                f"{uklassifiserte}"
            )

    # ---------------------------------------------------------
    # Valider valgte gjennomsnittsvariable
    # ---------------------------------------------------------
    vektede_kolonner = list(vektede_variable.keys())

    manglende_vektede = [
        col for col in vektede_kolonner if col not in df.columns
    ]
    if manglende_vektede:
        raise KeyError(
            "Følgende variable som skal vektes finnes ikke i datasettet: "
            f"{manglende_vektede}"
        )

    manglende_vekter = sorted(
        {
            weight
            for weight in vektede_variable.values()
            if weight not in df.columns
        }
    )
    if manglende_vekter:
        raise KeyError(
            "Følgende vektvariable finnes ikke i datasettet: "
            f"{manglende_vekter}"
        )

    manglende_gjennomsnittsvariable = [
        col for col in gjennomsnittsvariable if col not in df.columns
    ]
    if manglende_gjennomsnittsvariable:
        raise KeyError(
            "Følgende gjennomsnittsvariable finnes ikke i datasettet: "
            f"{manglende_gjennomsnittsvariable}"
        )

    overlapp_gjennomsnitt = sorted(
        set(vektede_kolonner) & set(gjennomsnittsvariable)
    )
    if overlapp_gjennomsnitt:
        raise ValueError(
            "Samme variabel kan ikke beregnes både som vektet og vanlig "
            f"gjennomsnitt: {overlapp_gjennomsnitt}"
        )

    spesialvariable = set(vektede_kolonner) | set(
        gjennomsnittsvariable
    )

    ikke_statistikkvariable = sorted(
        spesialvariable - set(statistikkvariable)
    )
    if ikke_statistikkvariable:
        raise ValueError(
            "Variable som skal gjennomsnittsberegnes må også være definert "
            "som statistikkvariable. Følgende mangler: "
            f"{ikke_statistikkvariable}"
        )

    vektvariable = sorted(set(vektede_variable.values()))

    vekt_ikke_statistikkvariable = sorted(
        set(vektvariable) - set(statistikkvariable)
    )
    if vekt_ikke_statistikkvariable:
        raise ValueError(
            "Vektvariable må være definert som statistikkvariable. "
            f"Følgende mangler: {vekt_ikke_statistikkvariable}"
        )

    # ---------------------------------------------------------
    # Numerisk validering
    # ---------------------------------------------------------
    for col in statistikkvariable:
        if (
            not pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        ):
            raise TypeError(
                f"Statistikkvariabelen '{col}' må være numerisk. "
                f"Fant dtype {df[col].dtype}."
            )

    for weight_col in vektvariable:
        if (df[weight_col].dropna() < 0).any():
            negative_regioner = (
                df.loc[df[weight_col] < 0, region_col]
                .astype(str)
                .tolist()
            )
            raise ValueError(
                f"Vektvariabelen '{weight_col}' inneholder negative verdier "
                f"for regionene {negative_regioner}. Negative vekter er "
                "ikke tillatt."
            )

    # ---------------------------------------------------------
    # Vis hvordan brukerens valg tolkes
    # ---------------------------------------------------------
    summeres = [
        col
        for col in statistikkvariable
        if col not in spesialvariable
    ]

    logger.info(
        "ℹ️ Variablene behandles slik:\n"
        f"Klassifikasjonsvariable: {alle_klassifikasjonsvariable}\n"
        f"Statistikkvariable: {statistikkvariable}\n"
        f"Vektede gjennomsnitt: {vektede_variable}\n"
        f"Vanlige gjennomsnitt: {gjennomsnittsvariable}\n"
        f"Summeres: {summeres}"
    )

    # Dtypes før beregning
    gjennomsnittskolonner = list(
        dict.fromkeys(vektede_kolonner + gjennomsnittsvariable)
    )

    original_dtypes: dict[str, Any] = cast(
        dict[str, Any],
        df[gjennomsnittskolonner].dtypes.to_dict(),
    )

    # ---------------------------------------------------------
    # Midlertidige kolonner og rapport om utelatte observasjoner
    # ---------------------------------------------------------
    temp_cols: list[str] = []
    weighted_temp: dict[str, tuple[str, str]] = {}
    mean_temp: dict[str, str] = {}
    excluded_records: list[dict[str, Any]] = []

    rapport_klassifikasjonsvariable = [
        col
        for col in alle_klassifikasjonsvariable
        if col != region_col
    ]

    for i, (target_col, weight_col) in enumerate(
        vektede_variable.items()
    ):
        numerator_col = f"__vektet_{i}_teller"
        denominator_col = f"__vektet_{i}_nevner"

        if numerator_col in df.columns or denominator_col in df.columns:
            raise ValueError(
                "Datasettet inneholder et reservert midlertidig "
                f"kolonnenavn: '{numerator_col}' eller "
                f"'{denominator_col}'."
            )

        valid = df[target_col].notna() & df[weight_col].notna()

        df[numerator_col] = (
            df[target_col] * df[weight_col]
        ).where(valid)

        df[denominator_col] = df[weight_col].where(valid)

        temp_cols.extend([numerator_col, denominator_col])
        weighted_temp[target_col] = (
            numerator_col,
            denominator_col,
        )
        ###
        manglende_verdi = df[target_col].isna()
        manglende_vekt = df[weight_col].isna()
        excluded = manglende_verdi | manglende_vekt
        
        if excluded.any():
            kun_manglende_verdi = manglende_verdi & ~manglende_vekt
            kun_manglende_vekt = ~manglende_verdi & manglende_vekt
            mangler_begge = manglende_verdi & manglende_vekt
        
            antall_manglende_verdi = int(kun_manglende_verdi.sum())
            antall_manglende_vekt = int(kun_manglende_vekt.sum())
            antall_mangler_begge = int(mangler_begge.sum())
            antall_utelatte = int(excluded.sum())
        
            warning_lines = [
                f"⚠️ '{target_col}':",
                (
                    f"{antall_utelatte} observasjon(er) utelates fra "
                    "beregningen av vektet gjennomsnitt:"
                ),
            ]
        
            if antall_manglende_verdi:
                warning_lines.append(
                    f"- {antall_manglende_verdi} med manglende verdi"
                )
        
            if antall_manglende_vekt:
                warning_lines.append(
                    f"- {antall_manglende_vekt} med manglende vekt"
                )
        
            if antall_mangler_begge:
                warning_lines.append(
                    f"- {antall_mangler_begge} med både manglende verdi og vekt"
                )
        
            if antall_manglende_verdi or antall_mangler_begge:
                warning_lines.extend(
                    [
                        "",
                        (
                            "Dersom en manglende verdi egentlig betyr 0, må "
                            "inputdatasettet korrigeres før funksjonen kjøres."
                        ),
                    ]
                )
        
            logger.warning("\n".join(warning_lines))
        
            for idx in df.index[excluded]:
                if manglende_verdi.loc[idx] and manglende_vekt.loc[idx]:
                    reason = "manglende verdi og vekt"
                elif manglende_verdi.loc[idx]:
                    reason = "manglende verdi"
                else:
                    reason = "manglende vekt"
        
                record: dict[str, Any] = {
                    "variabel": target_col,
                    "vektvariabel": weight_col,
                    "region": df.at[idx, region_col],
                    "grunn": reason,
                    "verdi": df.at[idx, target_col],
                    "vekt": df.at[idx, weight_col],
                }
        
                for class_col in rapport_klassifikasjonsvariable:
                    record[class_col] = df.at[idx, class_col]
        
                excluded_records.append(record)
            ###

    for i, target_col in enumerate(gjennomsnittsvariable):
        count_col = f"__gjennomsnitt_{i}_antall"

        if count_col in df.columns:
            raise ValueError(
                "Datasettet inneholder et reservert midlertidig "
                f"kolonnenavn: '{count_col}'."
            )

        valid = df[target_col].notna()

        df[count_col] = (
            valid.astype("float64")
            .replace(0.0, float("nan"))
        )

        temp_cols.append(count_col)
        mean_temp[target_col] = count_col

        if (~valid).any():
            manglende_regioner = (
                df.loc[~valid, region_col]
                .astype(str)
                .tolist()
            )
            logger.warning(
                f"⚠️ '{target_col}' har manglende verdier for "
                f"{int((~valid).sum())} observasjoner. "
                "Disse observasjonene inngår ikke i beregningen av "
                f"gjennomsnitt. Regioner: {manglende_regioner}"
            )

    # ---------------------------------------------------------
    # Bruk samme regionale mapping som hierarki()
    # ---------------------------------------------------------
    mappingfil, join_col, replace_col, post_filter, rename_cols = (
        _select_mapping(
            aggregeringstype,
            region_col,
            periode,
        )
    )

    df_merged = df.merge(
        mappingfil,
        left_on=join_col,
        right_on="from",
        how="inner",
    )

    df_merged[replace_col] = df_merged["to"]

    aggregeringskolonner = list(
        dict.fromkeys(statistikkvariable + temp_cols)
    )

    df_agg = df_merged.groupby(
        alle_klassifikasjonsvariable,
        as_index=False,
        observed=True,
    )[aggregeringskolonner].sum(min_count=1)

    # ---------------------------------------------------------
    # Beregn vektede gjennomsnitt
    # ---------------------------------------------------------
    aggregerte_problemer: list[dict[str, Any]] = []

    for target_col, (
        numerator_col,
        denominator_col,
    ) in weighted_temp.items():
        result = df_agg[numerator_col] / df_agg[denominator_col]

        if decimals is not None:
            result = result.round(decimals)

        df_agg[target_col] = result

        ugyldig_nevner = (
            df_agg[denominator_col].isna()
            | (df_agg[denominator_col] == 0)
        )

        if ugyldig_nevner.any():
            problem_regioner = (
                df_agg.loc[ugyldig_nevner, region_col]
                .astype(str)
                .tolist()
            )

            logger.warning(
                f"⚠️ '{target_col}' kan ikke beregnes for følgende "
                "aggregerte regioner fordi ingen gyldig positiv "
                f"vekt inngår i nevneren: {problem_regioner}"
            )

            for idx in df_agg.index[ugyldig_nevner]:
                aggregerte_problemer.append(
                    {
                        "variabel": target_col,
                        "region": df_agg.at[idx, region_col],
                        "grunn": "ingen gyldig vekt i nevner",
                    }
                )

    # ---------------------------------------------------------
    # Beregn vanlige gjennomsnitt
    # ---------------------------------------------------------
    for target_col, count_col in mean_temp.items():
        result = df_agg[target_col] / df_agg[count_col]

        if decimals is not None:
            result = result.round(decimals)

        df_agg[target_col] = result

        ingen_observasjoner = df_agg[count_col].isna()

        if ingen_observasjoner.any():
            problem_regioner = (
                df_agg.loc[ingen_observasjoner, region_col]
                .astype(str)
                .tolist()
            )

            logger.warning(
                f"⚠️ '{target_col}' kan ikke beregnes for følgende "
                "aggregerte regioner fordi alle observasjoner mangler: "
                f"{problem_regioner}"
            )

    # ---------------------------------------------------------
    # Fjern hjelpekolonner før original + aggregert kombineres
    # ---------------------------------------------------------
    df_agg = df_agg.drop(columns=temp_cols)

    original_columns = (
        alle_klassifikasjonsvariable + statistikkvariable
    )
    df_original = df[original_columns].copy()

    df_combined = pd.concat(
        [df_original, df_agg],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Dtype-håndtering
    # ---------------------------------------------------------
    post_op_dtypes: dict[str, Any] = cast(
        dict[str, Any],
        df_combined[gjennomsnittskolonner].dtypes.to_dict(),
    )

    if restore_original_dtype:
        for col in gjennomsnittskolonner:
            df_combined[col] = _restore_dtype(
                df_combined[col],
                original_dtypes[col],
            )

    final_dtypes: dict[str, Any] = cast(
        dict[str, Any],
        df_combined[gjennomsnittskolonner].dtypes.to_dict(),
    )

    # ---------------------------------------------------------
    # Samme etterbehandling som hierarki()
    # ---------------------------------------------------------
    df_combined = _postprocess_combined(
        df_combined,
        post_filter,
        rename_cols,
        alle_klassifikasjonsvariable,
        add_region_names,
    )

    if return_report:
        excluded_df = pd.DataFrame(excluded_records)
        aggregerte_problemer_df = pd.DataFrame(
            aggregerte_problemer
        )

        report: dict[str, Any] = {
            "variabelbehandling": {
                "klassifikasjonsvariable": (
                    alle_klassifikasjonsvariable
                ),
                "statistikkvariable": statistikkvariable,
                "vektede_variable": vektede_variable,
                "gjennomsnittsvariable": (
                    gjennomsnittsvariable
                ),
                "summeres": summeres,
            },
            "utelatte_observasjoner": excluded_df,
            "aggregerte_problemer": aggregerte_problemer_df,
            "dtype": {
                "original": original_dtypes,
                "post_op": post_op_dtypes,
                "final": final_dtypes,
            },
        }

        return df_combined, report

    return df_combined


