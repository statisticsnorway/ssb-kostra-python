from ssb_kostra_python.hente_data_folkemengde import hente_data_folkemengde

import pandas as pd
from fagfunksjoner.fagfunksjoner_logger import logger
from klass import KlassClassification

from ssb_kostra_python import hjelpefunksjoner
from ssb_kostra_python import regionshierarki

INPUT_PATCH_TARGET = "builtins.input"
from unittest.mock import patch

statistikkaar = 2020
regionsnivaa = "kommune"
testdata = True


# +
def mapping_mellom_aar(statistikkaar: int | str) -> pd.DataFrame:
    statistikkaar_int = int(statistikkaar)
    kildeaar = str(statistikkaar_int - 1)

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

    try:
        klassifikasjon = KlassClassification(
            131,
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
        logger.info(
            f"Ingen kommuneendringer funnet for {kildeaar}–{statistikkaar_int}."
        )
        return pd.DataFrame(columns=kolonner)

    df_endringer = df_endringer.copy()
    df_endringer["kildeaar"] = kildeaar
    df_endringer["statistikkaar"] = str(statistikkaar_int)

    manglende_kolonner = [col for col in kolonner if col not in df_endringer.columns]

    if manglende_kolonner:
        logger.warning(
            f"Endringsloggen mangler forventede kolonner: {manglende_kolonner}"
        )
        return pd.DataFrame(columns=kolonner)

    return df_endringer[kolonner]


mapping = mapping_mellom_aar(statistikkaar)
display(mapping)


def _normaliser_kommunekode(verdi: object) -> str:
    kode = str(verdi).strip()

    if kode.isdigit():
        return kode.zfill(4)

    return kode


def anvende_kommunereform(
    inputfil: pd.DataFrame,
    mapping: pd.DataFrame | None,
    statistikkvariable: list[str],
    statistikkaar: int | str,
) -> pd.DataFrame:
    df = inputfil.copy()

    df["kommuneregion"] = df["kommuneregion"].map(_normaliser_kommunekode)
    df["periode"] = str(statistikkaar)

    kildeaar = str(statistikkaar - 1)
    logger.info(f"Kildeåret er {kildeaar}.")

    if mapping is None or mapping.empty:
        logger.info(
            f"Ingen kommunereform funnet. Det har ikke vært noen endringer mellom {kildeaar} og {statistikkaar}. Kopierer kommuneregion uendret og oppdaterer periode."
        )
        return df

    mapping = mapping.copy()
    mapping["oldCode"] = mapping["oldCode"].map(_normaliser_kommunekode)
    mapping["newCode"] = mapping["newCode"].map(_normaliser_kommunekode)

    old_counts = mapping.groupby("oldCode")["newCode"].nunique()
    new_counts = mapping.groupby("newCode")["oldCode"].nunique()

    splits = old_counts[old_counts > 1]
    mergers = new_counts[new_counts > 1]

    if not splits.empty:
        logger.info(f"Kommunesplittinger funnet: {splits.to_dict()}")

    if not mergers.empty:
        logger.info(f"Kommunesammenslåinger funnet: {mergers.to_dict()}")

    changed_old_codes = set(mapping["oldCode"])
    input_codes = set(df["kommuneregion"])

    unmapped_codes = sorted(input_codes - changed_old_codes)
    if unmapped_codes:
        logger.info(
            f"{len(unmapped_codes)} kommuneregioner finnes ikke i endringsloggen "
            "og kopieres uendret."
        )

    df_mapped = df.merge(
        mapping[["oldCode", "newCode"]],
        left_on="kommuneregion",
        right_on="oldCode",
        how="left",
    )

    df_mapped["kommuneregion"] = df_mapped["newCode"].fillna(df_mapped["kommuneregion"])

    df_mapped = df_mapped.drop(columns=["oldCode", "newCode"])

    klassifikasjonsvariable = [
        col for col in df_mapped.columns if col not in statistikkvariable
    ]

    duplicated_keys = df_mapped.duplicated(subset=klassifikasjonsvariable, keep=False)

    if duplicated_keys.any():
        logger.info(
            "Dupliserte klassifikasjonsnøkler funnet etter kommunereform. "
            "Aggregerer statistikkvariable."
        )

        df_mapped = df_mapped.groupby(klassifikasjonsvariable, as_index=False)[
            statistikkvariable
        ].sum()
    else:
        logger.info("Ingen aggregering nødvendig etter kommunereform.")

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
        mapping = mapping_mellom_aar(statistikkaar)

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


# This is how you run the code
# testdatasett = hente_data_folkemengde(statistikkaar, "kommune", False)
# display(testdatasett)
# -

testdatasett = hente_data_folkemengde(statistikkaar, "kommune", False)
display(testdatasett)

klassifikasjon = KlassClassification(
    103,
    language="nb",
    include_future=True,
)
display(klassifikasjon)
df_endringer = klassifikasjon.get_changes(
    f"{2003}-12-31",
    f"{2004}-12-31",
)
display(df_endringer)


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
    else:
        raise ValueError("regionsnivaa må være 'kommune' eller 'bydel'.")

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


mapping = mapping_mellom_aar(statistikkaar)
display(mapping)


def _normaliser_kommunekode(verdi: object) -> str:
    kode = str(verdi).strip()

    if kode.isdigit():
        return kode.zfill(4)

    return kode


def anvende_kommunereform(
    inputfil: pd.DataFrame,
    mapping: pd.DataFrame | None,
    statistikkvariable: list[str],
    statistikkaar: int | str,
) -> pd.DataFrame:
    df = inputfil.copy()

    df["kommuneregion"] = df["kommuneregion"].map(_normaliser_kommunekode)
    df["periode"] = str(statistikkaar)

    kildeaar = str(int(statistikkaar - 1))
    logger.info(f"Kildeåret er {kildeaar}.")

    if mapping is None or mapping.empty:
        logger.info(
            f"Ingen kommunereform funnet. Det har ikke vært noen endringer mellom {kildeaar} og {statistikkaar}. Kopierer kommuneregion uendret og oppdaterer periode."
        )
        return df

    mapping = mapping.copy()
    mapping["oldCode"] = mapping["oldCode"].map(_normaliser_kommunekode)
    mapping["newCode"] = mapping["newCode"].map(_normaliser_kommunekode)

    old_counts = mapping.groupby("oldCode")["newCode"].nunique()
    new_counts = mapping.groupby("newCode")["oldCode"].nunique()

    splits = old_counts[old_counts > 1]
    mergers = new_counts[new_counts > 1]

    if not splits.empty:
        logger.info(f"Kommunesplittinger funnet: {splits.to_dict()}")

    if not mergers.empty:
        logger.info(f"Kommunesammenslåinger funnet: {mergers.to_dict()}")

    changed_old_codes = set(mapping["oldCode"])
    input_codes = set(df["kommuneregion"])

    unmapped_codes = sorted(input_codes - changed_old_codes)
    if unmapped_codes:
        logger.info(
            f"{len(unmapped_codes)} kommuneregioner finnes ikke i endringsloggen "
            "og kopieres uendret."
        )

    df_mapped = df.merge(
        mapping[["oldCode", "newCode"]],
        left_on="kommuneregion",
        right_on="oldCode",
        how="left",
    )

    df_mapped["kommuneregion"] = df_mapped["newCode"].fillna(df_mapped["kommuneregion"])

    df_mapped = df_mapped.drop(columns=["oldCode", "newCode"])

    klassifikasjonsvariable = [
        col for col in df_mapped.columns if col not in statistikkvariable
    ]

    duplicated_keys = df_mapped.duplicated(subset=klassifikasjonsvariable, keep=False)

    if duplicated_keys.any():
        logger.info(
            "Dupliserte klassifikasjonsnøkler funnet etter kommunereform. "
            "Aggregerer statistikkvariable."
        )

        df_mapped = df_mapped.groupby(klassifikasjonsvariable, as_index=False)[
            statistikkvariable
        ].sum()
    else:
        logger.info("Ingen aggregering nødvendig etter kommunereform.")

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
