# from ssb_kostra_python.hente_data_folkemengde import hente_data_folkemengde

import pandas as pd
from fagfunksjoner.fagfunksjoner_logger import logger
from klass import KlassClassification

from ssb_kostra_python import hjelpefunksjoner
from ssb_kostra_python import regionshierarki

INPUT_PATCH_TARGET = "builtins.input"
from unittest.mock import patch

statistikkaar = 2020
regionsnivaa = "bydel"
testdata = True

mapping = mapping_mellom_aar(statistikkaar, regionsnivaa)
display(mapping)


# +
def mapping_mellom_aar(
    statistikkaar: int | str,
    regionsnivaa: str = "kommune",
) -> pd.DataFrame:
    statistikkaar_int = int(statistikkaar)
    kildeaar = str(statistikkaar_int - 1)
    regionsnivaa = regionsnivaa.lower()

    kolonner = [
        "kildeaar",
        "oldCode",
        "oldName",
        "oldShortName",
        "statistikkaar",
        "newCode",
        "newName",
        "newShortName",
        "changeOccurred",
    ]

    if regionsnivaa == "kommune":
        klass_kode = 131
        endringstekst = "kommuneendringer"
    elif regionsnivaa == "bydel":
        klass_kode = 103
        endringstekst = "bydelsendringer"
    elif regionsnivaa == "fylkeskommune":
        klass_kode = 127
        endringstekst = "fylkeskommuneendringer"
    else:
        raise ValueError(
            "regionsnivaa må være 'bydel', 'kommune' eller 'fylkeskommune'."
        )

    try:
        klassifikasjon = KlassClassification(
            klass_kode,
            language="nb",
            include_future=True,
        )

        df_endringer = klassifikasjon.get_changes(
            f"{kildeaar}-12-31",
            f"{statistikkaar_int}-12-31",
        )

    except Exception:
        logger.warning(
            f"Ingen endringslogg funnet for {kildeaar}–{statistikkaar_int}. "
            "Returnerer tom mapping."
        )
        return pd.DataFrame(columns=kolonner)

    if df_endringer is None or df_endringer.empty:
        logger.info(f"Ingen {endringstekst} funnet for {kildeaar}–{statistikkaar_int}.")
        return pd.DataFrame(columns=kolonner)

    df_endringer = df_endringer.copy()

    if regionsnivaa == "bydel":
        df_endringer = df_endringer[
            df_endringer["oldCode"].astype(str).str.startswith("03")
            & df_endringer["newCode"].astype(str).str.startswith("03")
        ].copy()

        if df_endringer.empty:
            logger.info(
                f"Ingen Oslo-bydelsendringer funnet for "
                f"{kildeaar}–{statistikkaar_int}."
            )
            return pd.DataFrame(columns=kolonner)

    df_endringer["kildeaar"] = kildeaar
    df_endringer["statistikkaar"] = str(statistikkaar_int)

    manglende_kolonner = [col for col in kolonner if col not in df_endringer.columns]

    if manglende_kolonner:
        logger.warning(
            f"Endringsloggen mangler forventede kolonner: {manglende_kolonner}"
        )
        return pd.DataFrame(columns=kolonner)

    return df_endringer[kolonner]


# mapping = mapping_mellom_aar(statistikkaar)
# display(mapping)


def _regionkolonne(regionsnivaa: str) -> str:
    if regionsnivaa == "kommune":
        return "kommuneregion"
    if regionsnivaa == "bydel":
        return "bydelsregion"
    if regionsnivaa == "fylkeskommune":
        return "fylkesregion"

    raise ValueError("regionsnivaa må være 'bydel', 'kommune' eller 'fylkeskommune'.")


def _normaliser_regionkode(verdi: object, regionsnivaa: str) -> str:
    kode = str(verdi).strip()

    if not kode.isdigit():
        return kode

    if regionsnivaa == "bydel":
        return kode.zfill(6)

    return kode.zfill(4)


def anvende_kommunereform(
    inputfil: pd.DataFrame,
    mapping: pd.DataFrame | None,
    statistikkvariable: list[str],
    statistikkaar: int | str,
    regionsnivaa: str,
) -> pd.DataFrame:
    df = inputfil.copy()
    regionsnivaa = regionsnivaa.lower()
    regionkolonne = _regionkolonne(regionsnivaa)

    df[regionkolonne] = df[regionkolonne].map(
        lambda x: _normaliser_regionkode(x, regionsnivaa)
    )
    df["periode"] = str(statistikkaar)

    kildeaar = str(int(statistikkaar) - 1)
    logger.info(f"Kildeåret er {kildeaar}.")

    if mapping is None or mapping.empty:
        logger.info(
            f"Ingen regionendringer funnet mellom {kildeaar} og {statistikkaar}. "
            "Kopierer region uendret og oppdaterer periode."
        )
        return df

    mapping = mapping.copy()
    mapping["oldCode"] = mapping["oldCode"].map(
        lambda x: _normaliser_regionkode(x, regionsnivaa)
    )
    mapping["newCode"] = mapping["newCode"].map(
        lambda x: _normaliser_regionkode(x, regionsnivaa)
    )

    df_mapped = df.merge(
        mapping[["oldCode", "newCode"]],
        left_on=regionkolonne,
        right_on="oldCode",
        how="left",
    )

    df_mapped[regionkolonne] = df_mapped["newCode"].fillna(df_mapped[regionkolonne])

    df_mapped = df_mapped.drop(columns=["oldCode", "newCode"])

    klassifikasjonsvariable = [
        col for col in df_mapped.columns if col not in statistikkvariable
    ]

    duplicated_keys = df_mapped.duplicated(subset=klassifikasjonsvariable, keep=False)

    if duplicated_keys.any():
        logger.info(
            "Dupliserte klassifikasjonsnøkler funnet etter regionendring. "
            "Aggregerer statistikkvariable."
        )

        df_mapped = df_mapped.groupby(klassifikasjonsvariable, as_index=False)[
            statistikkvariable
        ].sum()
    else:
        logger.info("Ingen aggregering nødvendig etter regionendring.")

    return df_mapped


def hente_data_folkemengde(
    aar: int, regionsnivaa: str, testdata: bool = False
) -> pd.DataFrame:
    """Henter folkemengdedata 31.12 for valgt år og regionsnivå.

    Hvis testdata=True, hentes data fra året før, periode settes til aar,
    og for kommuner anvendes eventuell kommunereform mellom kildeåret og aar.
    """
    regionsnivaa = regionsnivaa.lower()
    statistikkaar = int(aar)

    if testdata:
        kildeaar = statistikkaar - 1
        logger.info(
            f"Lager testdata for {statistikkaar} basert på data fra {kildeaar}."
        )
    else:
        kildeaar = statistikkaar
        logger.info(f"Henter reelle data for {statistikkaar}.")

    if regionsnivaa == "bydel":
        folkemengde_31_12 = hjelpefunksjoner._hent_folkemengde_bydeler_31_12(kildeaar)

    elif regionsnivaa == "kommune":
        _, folkemengde_kommune = hjelpefunksjoner._hent_folkemengde_kommune_31_12(
            kildeaar
        )

        if folkemengde_kommune is None:
            raise RuntimeError(
                f"Klarte ikke å lage KOSTRA-aggregert folkemengdefil for {kildeaar}."
            )

        folkemengde_31_12 = folkemengde_kommune

    elif regionsnivaa == "fylkeskommune":
        folkemengde_31_12 = hjelpefunksjoner._hent_folkemengde_fylkeskommune_31_12(
            kildeaar
        )

    else:
        raise ValueError(
            "Du må angi regionsnivå som 'bydel', 'kommune' eller 'fylkeskommune'."
        )

    folkemengde_31_12_data = folkemengde_31_12.copy()

    if testdata and regionsnivaa == "kommune":
        mapping = mapping_mellom_aar(statistikkaar, regionsnivaa)

        folkemengde_31_12_data = anvende_kommunereform(
            inputfil=folkemengde_31_12_data,
            mapping=mapping,
            statistikkvariable=["personer"],
            statistikkaar=statistikkaar,
        )

        print("✅Fjerner KOSTRA-grupperingene før de legges på igjen.")
        folkemengde_31_12_data_uten_agg = folkemengde_31_12_data[
            ~folkemengde_31_12_data["kommuneregion"]
            .astype(str)
            .str.match(r"^(EKG\d{2}|EKA\d{2}|EAK|EAKUO)$")
        ].copy()

        predefined_input = "alder"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            folkemengde_31_12_data = regionshierarki.hierarki(
                folkemengde_31_12_data_uten_agg
            )
        # display(folkemengde_31_12_data)

    elif testdata:
        folkemengde_31_12_data["periode"] = str(statistikkaar)

    return folkemengde_31_12_data


# -

testdatasett = hente_data_folkemengde(statistikkaar, regionsnivaa, testdata)
display(testdatasett)

# +
# if testdata and regionsnivaa in ("kommune", "bydel"):
#     mapping = mapping_mellom_aar(statistikkaar, regionsnivaa)

#     folkemengde_31_12_data = anvende_kommunereform(
#         inputfil=folkemengde_31_12_data,
#         mapping=mapping,
#         statistikkvariable=["personer"],
#         statistikkaar=statistikkaar,
#     )

#     if regionsnivaa in ("kommune", "bydel"):
#         print("✅Fjerner KOSTRA-grupperingene før de legges på igjen.")

#     if regionsnivaa == "kommune":
#         mask = (
#             folkemengde_31_12_data["kommuneregion"]
#             .astype(str)
#             .str.match(r"^(EKG\d{2}|EKA\d{2}|EAK|EAKUO)$")
#         )
#     else:  # bydel
#         mask = (
#             folkemengde_31_12_data["kommuneregion"]
#             .astype(str)
#             == "EAB"
#         )

#     folkemengde_31_12_data_uten_agg = (
#         folkemengde_31_12_data.loc[~mask].copy()
#     )

#     predefined_input = "alder"
#     with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
#         folkemengde_31_12_data = regionshierarki.hierarki(
#             folkemengde_31_12_data_uten_agg
#         )

# elif testdata:
#     folkemengde_31_12_data["periode"] = str(statistikkaar)
