# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: ssb-kostra-python
#     language: python
#     name: ssb-kostra-python
# ---

# +

# import dapla as dp
# from dapla import FileClient
from unittest.mock import patch

import pandas as pd

INPUT_PATCH_TARGET = "builtins.input"

import duckdb
from fagfunksjoner import latest_version_path
from fagfunksjoner import logger
from IPython.display import display  # for nice tables in notebooks

from ssb_kostra_python import regionshierarki
from ssb_kostra_python import summere_kjonn
from ssb_kostra_python import summere_til_aldersgrupperinger

# -


def hent_folkemengde_bydeler_31_12(statistikkaar: str | int) -> pd.DataFrame:
    """Henter, bearbeider og aggregerer folkemengdedata for bydeler per 31.12 for et gitt statistikkår.

    Funksjonen henter først bydeldata fra bøtte, og kjører deretter følgende
    operasjoner:

    1. Leser inn folkemengdefil for bydeler
    2. Summerer til KLASS-aldersgrupperinger
    3. Summerer over kjønn
    4. Grupperer over Oslo-bydelene
    5. Fjerner ettårige alderskoder 105-120

    Parametre
    ----------
    statistikkaar : int eller str
        Statistikkåret det skal hentes data for, for eksempel 2024.

    Returverdi
    ----------
    pandas.DataFrame
        Ferdig aggregert folkemengdedatasett for bydeler per 31.12.

    Eksempel
    --------
    >>> folkemengde_31_12_b = hent_folkemengde_bydeler_31_12(2024)
    >>> display(folkemengde_31_12_b)

    Feilhåndtering
    ---------------
    Hvert trinn har egen feilhåndtering. Ved suksess skrives en bekreftelse.
    Ved feil gis en advarsel og funksjonen stopper, fordi hvert trinn er
    avhengig av resultatet fra forrige trinn.
    """
    INPUT_PATCH_TARGET = "builtins.input"

    # ---------- Hent filsti ----------
    try:
        statistikkaar = str(statistikkaar)
        bucket_inndata = f"/buckets/delt-kostra-befolkning-delt/bydeler/{statistikkaar}"

        bucket_inndata_data_path = latest_version_path(
            f"{bucket_inndata}/folkmengde_bydeler_p{statistikkaar}-12-31"
        )

        print(f"✅Filsti funnet: {bucket_inndata_data_path}")

    except Exception as e:
        msg = f"❌Operasjonen latest_version_path for bydeldata feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    # ---------- Les parquet ----------
    try:
        folketall_bydeler = pd.read_parquet(bucket_inndata_data_path)

        print("✅Operasjonen innlesing av folkemengdefil for bydeler ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen innlesing av folkemengdefil for bydeler feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    # ---------- Summer til KLASS-aldersgrupperinger ----------
    try:
        predefined_input = "kjonn, alder, to"

        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            _rename_variabel, _groupby_variable, df_sum_med_kjonn = (
                summere_til_aldersgrupperinger.summere_til_aldersgrupperinger(
                    folketall_bydeler,
                    hierarki_path="/buckets/produkt/befolkning/_config/mapping_aldershierarki.parquet",
                )
            )

        print(
            "✅Operasjonen summere_til_aldersgrupperinger for bydeler ble gjennomført."
        )

    except Exception as e:
        msg = f"❌Operasjonen summere_til_aldersgrupperinger for bydeler feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    # ---------- Summer over kjønn ----------
    try:
        predefined_input = "kjonn, alder"

        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            df_sum_kjonn = summere_kjonn.summere_over_kjonn(df_sum_med_kjonn)

        print("✅Operasjonen summere_over_kjonn for bydeler ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen summere_over_kjonn for bydeler feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    # ---------- Grupper over Oslo-bydelene ----------
    try:
        predefined_input = "alder"

        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            folkemengde_31_12_b = regionshierarki.hierarki(df_sum_kjonn)

        print(
            "✅Operasjonen regionshierarki.hierarki for Oslo-bydeler ble gjennomført."
        )

    except Exception as e:
        msg = f"❌Operasjonen regionshierarki.hierarki for Oslo-bydeler feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    # ---------- Filtrer bort ettårige alderskoder ----------
    try:
        aldre_som_fjernes = [
            "105",
            "106",
            "107",
            "108",
            "109",
            "110",
            "111",
            "112",
            "113",
            "114",
            "115",
            "116",
            "117",
            "118",
            "119",
            "120",
        ]

        folkemengde_31_12_b = folkemengde_31_12_b[
            ~folkemengde_31_12_b["alder"].isin(aldre_som_fjernes)
        ]

        print("✅Operasjonen filtrering av alder 105-120 for bydeler ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen filtrering av alder 105-120 for bydeler feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    return folkemengde_31_12_b


folkemengde_31_12_b = hent_folkemengde_bydeler_31_12(2024)
display(folkemengde_31_12_b)


def hent_folkemengde_kommune_31_12(statistikkaar: str | int) -> pd.DataFrame:
    """Henter og bearbeider befolkningsdata per 31.12 for et gitt statistikkår.

    Funksjonen forsøker å hente data fra to kilder:
    - Kommunedata (per 31.12.<år>)
    - Svalbarddata (per 01.01.<år+1>, justert tilbake til <år>)

    Dersom én av kildene mangler, vil funksjonen fortsatt returnere et gyldig
    datasett basert på tilgjengelig kilde. Dersom begge kilder mangler, kastes
    en FileNotFoundError.

    Videre forsøker funksjonen å kjøre en sekvens av transformasjoner:
    1. summere_til_aldersgrupperinger
    2. summere_over_kjonn
    3. hierarki
    4. filtrering av alder (fjerner 105-120)

    Hver operasjon håndteres individuelt:
    - Ved suksess skrives en bekreftelsesmelding
    - Ved feil gis en advarsel, og videre prosessering stoppes
    - Grunnlagsdata returneres uansett dersom disse er tilgjengelige

    Parametre
    ----------
    statistikkaar : int eller str
        Året det skal hentes befolkningsdata for (f.eks. 2024)

    Returverdier
    ------------
    tuple (df_folkemengde_31_12, df_folkemengde_31_12_kostra_agg_filtrert)

    - df_folkemengde_31_12 : pandas.DataFrame
        Aggregert befolkningsdata per kommune, kjønn og alder

    - df_folkemengde_31_12_kostra_agg_filtrert : pandas.DataFrame eller None
        Ferdig prosessert og filtrert datasett etter KOSTRA-hierarki.
        Returneres kun dersom alle transformasjonstrinn lykkes, ellers None.

    Unntak
    -------
    FileNotFoundError
        Dersom ingen av inputfilene (kommuner eller Svalbard) er tilgjengelige

    Advarsler
    ---------
    warnings.warn brukes dersom én eller flere inputfiler mangler,
    eller dersom en transformasjonsoperasjon feiler.

    Eksempel på bruk
    ----------------
    >>> df_base, df_final = hent_folkemengde_kommune_31_12(2024)

    >>> if df_final is not None:
    ...     display(df_final)
    ... else:
    ...     print("Videre prosessering feilet - viser kun grunnlagsdata.")
    ...     display(df_base)

    Notater
    -------
    - Funksjonen bruker duckdb for lesing av parquet-filer
    - mapping_hierarki-funksjonene benytter input(), som midlertidig patches
      internt i funksjonen
    - Outputmeldinger skrives til konsoll for transparens i kjøringen
    """
    statistikkaar = str(statistikkaar)
    bef_buckets_shared = "/buckets/shared/bef-statistikk/folketall"

    dataframes = []
    kilder_brukt = []
    mangler = []

    # ---------- Hent kommunedata ----------
    try:
        df_kommuner_data_path = latest_version_path(
            f"{bef_buckets_shared}/bosatte/{statistikkaar}/bosatte_p{statistikkaar}-12-31_v1.parquet"
        )

        df_kommuner = duckdb.query(f"""
            SELECT kjoenn AS kjonn,
                   komm_nr AS kommuneregion,
                   alder
            FROM '{df_kommuner_data_path}'
        """).to_df()

        df_kommuner["periode"] = statistikkaar
        df_kommuner["personer"] = 1

        dataframes.append(df_kommuner)
        kilder_brukt.append("kommuner")
        # print(f"Kommunedata hentet for {statistikkaar}.")
        logger.info(f"✅Kommunedata hentet for {statistikkaar}. \n")

    except Exception as e:
        msg = f"❌Kommunefil mangler eller kunne ikke leses: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        mangler.append(msg)

    # ---------- Hent Svalbarddata ----------
    try:
        aar_inputmappe = int(statistikkaar) + 1
        filsti_input_svalbard = f"{bef_buckets_shared}/svalbard/{aar_inputmappe}"

        df_svalbard_data_path = latest_version_path(
            f"{filsti_input_svalbard}/svalbardbosatte_p{aar_inputmappe}-01-01"
        )

        df_svalbard_data = duckdb.query(f"""
            SELECT kjoenn AS kjonn,
                   alder
            FROM '{df_svalbard_data_path}'
        """).to_df()

        df_svalbard_data["periode"] = statistikkaar
        df_svalbard_data["personer"] = 1
        df_svalbard_data["kommuneregion"] = "2111"

        dataframes.append(df_svalbard_data)
        kilder_brukt.append("svalbard")
        # print(f"Svalbarddata hentet for statistikkår {statistikkaar}.")
        logger.info(f"✅Svalbarddata hentet for statistikkår {statistikkaar}. \n")

    except Exception as e:
        msg = f"❌Svalbardfil mangler eller kunne ikke leses: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        mangler.append(msg)

    # ---------- Ingen grunnlagsdata ----------
    if not dataframes:
        raise FileNotFoundError(
            f"❌Ingen data tilgjengelig for statistikkår {statistikkaar}.\n"
            + "\n".join(mangler)
        )

    # ---------- Kombiner grunnlagsdata ----------
    if len(dataframes) == 1:
        df_folkemengde = dataframes[0]
    else:
        df_folkemengde = pd.concat(dataframes, ignore_index=True)

    df_folkemengde_31_12 = df_folkemengde.groupby(
        ["periode", "kommuneregion", "kjonn", "alder"], as_index=False
    )["personer"].sum()

    print(f"Datagrunnlag for {statistikkaar}: {', '.join(kilder_brukt)} brukt.")

    if mangler:
        print("Merk: Ett eller flere input manglet.")
        for m in mangler:
            print(f" - {m}")

    # ---------- Videre prosessering ----------
    df_folkemengde_31_12_kostra_agg_filtrert = None

    try:
        predefined_input = "kjonn, alder, to"
        with patch("builtins.input", return_value=predefined_input):
            (
                _rename_variabel,
                _groupby_variable,
                df_folkemengde_31_12_agg_alder,
            ) = summere_til_aldersgrupperinger.summere_til_aldersgrupperinger(
                df_folkemengde_31_12,
                hierarki_path="/buckets/produkt/befolkning/_config/mapping_aldershierarki.parquet",
            )

        print("✅Operasjonen summere_til_aldersgrupperinger ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen summere_til_aldersgrupperinger feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        print(msg)
        return df_folkemengde_31_12, None

    try:
        predefined_input = "kjonn, alder"
        with patch("builtins.input", return_value=predefined_input):
            df_folkemengde_31_12_agg_kjonn = summere_kjonn.summere_over_kjonn(
                df_folkemengde_31_12_agg_alder
            )

        print("✅Operasjonen summere_over_kjonn ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen summere_over_kjonn feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        print(msg)
        return df_folkemengde_31_12, None

    try:
        predefined_input = "alder"
        with patch("builtins.input", return_value=predefined_input):
            df_folkemengde_31_12_agg_kostra = regionshierarki.hierarki(
                df_folkemengde_31_12_agg_kjonn
            )

        print("✅Operasjonen hierarki ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen hierarki feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        print(msg)
        return df_folkemengde_31_12, None

    try:
        df_folkemengde_31_12_kostra_agg_filtrert = df_folkemengde_31_12_agg_kostra[
            ~df_folkemengde_31_12_agg_kostra["alder"].isin(
                [
                    "105",
                    "106",
                    "107",
                    "108",
                    "109",
                    "110",
                    "111",
                    "112",
                    "113",
                    "114",
                    "115",
                    "116",
                    "117",
                    "118",
                    "119",
                    "120",
                ]
            )
        ]

        print("✅Operasjonen filtrering av alder 105-120 ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen filtrering av alder 105-120 feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        print(msg)
        return df_folkemengde_31_12, None

    if df_folkemengde_31_12_kostra_agg_filtrert is not None:
        if len(kilder_brukt) == 2:
            print(
                "\n✅Sluttresultat (KOSTRA-aggregert) er basert på begge datakilder: kommuner og svalbard."
            )
        else:
            print(
                f"\nℹ️Sluttresultat (KOSTRA-aggregert) er kun basert på én datakilde: {kilder_brukt[0]}."
            )

    return df_folkemengde_31_12, df_folkemengde_31_12_kostra_agg_filtrert


# +
df_folkemengde_31_12, df_folkemengde_31_12_kostra_agg_filtrert = (
    hent_folkemengde_kommune_31_12(2025)
)

display(df_folkemengde_31_12)
display(df_folkemengde_31_12_kostra_agg_filtrert)


# -


def hent_folkemengde_31_12_fk(statistikkaar: str | int) -> pd.DataFrame:
    """Henter, aggregerer og grupperer folkemengdedata per 31.12 for et gitt statistikkår.

    Funksjonen bygger på `hent_folkemengde_kommune_31_12`, og returnerer et
    ferdig aggregert datasett på EAFK/KOSTRA-fylkesregionnivå.

    Operasjoner:
    1. Henter folkemengde per kommune, kjønn og alder
    2. Summerer opp til KLASS-aldersgrupperinger
    3. Summerer over kjønn
    4. Aggregerer kommuner til fylkeskommuner
    5. Aggregerer fylkeskommuner til KOSTRA-fylkesregioner inkludert EAFK(UO)
    6. Fjerner ettårige KLASS-alderskoder 105-120 og F025-029

    Parametre
    ----------
    statistikkaar : int eller str
        Statistikkåret det skal hentes og bearbeides data for.

    Returverdi
    ----------
    pandas.DataFrame
        Ferdig aggregert folkemengdedatasett på EAFK/KOSTRA-fylkesregionnivå.

    Eksempel
    --------
    >>> folkemengde_31_12_eafk = hent_folkemengde_31_12_eafk(2024)
    >>> display(folkemengde_31_12_eafk)

    Feilhåndtering
    ---------------
    Hvert trinn har egen feilhåndtering. Ved feil gis en advarsel som navngir
    operasjonen som feilet, og funksjonen stopper fordi senere trinn er avhengige
    av tidligere trinn.
    """
    INPUT_PATCH_TARGET = "builtins.input"

    try:
        df_folkemengde_31_12, _ = hent_folkemengde_kommune_31_12(statistikkaar)
        print("✅Operasjonen hent_folkemengde_kommune_31_12 ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen hent_folkemengde_kommune_31_12 feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    try:
        predefined_input = "kjonn, alder, to"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            _rename_variabel, _groupby_variable, df_sum_med_kjonn = (
                summere_til_aldersgrupperinger.summere_til_aldersgrupperinger(
                    df_folkemengde_31_12,
                    hierarki_path="/buckets/produkt/befolkning/_config/mapping_aldershierarki.parquet",
                )
            )

        print("✅Operasjonen summere_til_aldersgrupperinger ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen summere_til_aldersgrupperinger feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    try:
        predefined_input = "kjonn, alder"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            df_sum_kjonn = summere_kjonn.summere_over_kjonn(df_sum_med_kjonn)

        print("✅Operasjonen summere_over_kjonn ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen summere_over_kjonn feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    try:
        predefined_input = "alder"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            folkemengde_31_12_fk = regionshierarki.hierarki(
                df_sum_kjonn, aggregeringstype="kommune_til_fylkeskommune"
            )

        print(
            "✅Operasjonen hierarki med aggregeringstype='kommune_til_fylkeskommune' ble gjennomført."
        )

    except Exception as e:
        msg = (
            "Operasjonen hierarki med aggregeringstype='kommune_til_fylkeskommune' "
            f"feilet: {e}"
        )
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    try:
        predefined_input = "alder"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            folkemengde_31_12_eafk = regionshierarki.hierarki(folkemengde_31_12_fk)

        print("✅Operasjonen hierarki til KOSTRA-fylkesregioner/EAFK ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen hierarki til KOSTRA-fylkesregioner/EAFK feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    try:
        aldre_som_fjernes = [
            "105",
            "106",
            "107",
            "108",
            "109",
            "110",
            "111",
            "112",
            "113",
            "114",
            "115",
            "116",
            "117",
            "118",
            "119",
            "120",
            "F025-029",
        ]

        folkemengde_31_12_eafk = folkemengde_31_12_eafk[
            ~folkemengde_31_12_eafk["alder"].isin(aldre_som_fjernes)
        ]

        print("✅Operasjonen filtrering av alder 105-120 og F025-029 ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen filtrering av alder 105-120 og F025-029 feilet: {e}"
        logger.warning(msg)
        # warnings.warn(msg)
        raise RuntimeError(msg) from e

    return folkemengde_31_12_eafk


folkemengde_31_12_eafk = hent_folkemengde_31_12_fk(2025)
display(folkemengde_31_12_eafk)
