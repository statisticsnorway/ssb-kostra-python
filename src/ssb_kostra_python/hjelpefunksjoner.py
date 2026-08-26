INPUT_PATCH_TARGET = "builtins.input"
ALDERSHIERARKI_PATH = (
    "/buckets/delt-kostra-befolkning-delt/aldershierarki/mapping_aldershierarki.parquet"
)
import pandas as pd

from unittest.mock import patch

import duckdb
from fagfunksjoner import latest_version_path
from fagfunksjoner.fagfunksjoner_logger import logger

from klass import KlassClassification

from ssb_kostra_python import regionshierarki
from ssb_kostra_python import summere_kjonn
from ssb_kostra_python import summere_til_aldersgrupperinger


def format_fil(
    df_uformatert: pd.DataFrame,
) -> pd.DataFrame:
    """Formatering av periode- og regionsvariabelen.

    Dette er en funksjon du kan bruke til å formatere periode- og regionsvariabelen din. Funksjonen forutsetter at periodevariabelen er kalt 'periode'. Den forutsetter også at regionsvariabelen
    heter enten 'bydelsregion', 'kommuneregion' eller 'fylkesregion'. Ellers får du feilmelding. Den setter
    - periode til 4-sifret string-variabel. Ledende null(er) legges til dersom antallet sifre er lavere enn 4.
    - bydelsregion til 6-sifret string-variabel. Ledende null(er) legges til dersom antallet sifre er lavere enn 4.
    - kommuneregion til 4-sifret string-variabel. Ledende null(er) legges til dersom antallet sifre er lavere enn 4.
    - fylkesregion til 6-sifret string-variabel. Ledende null(er) legges til dersom antallet sifre er lavere enn 4.

    Skriv funksjonen slik:

    df_formatert = format_fil(df_uformatert)

    Her er "df_uformatert" den filen du ønsker å kjøre funksjonen på og rette formatet i. df_formatert er datasettet som spyttes ut, men du kan kalle den det du måtte ønske.

    Args:
        df_uformatert: Dataframe som skal formateres.

    Returns:
        Dataframe med formatert periode og regionvariabler.
    """
    df_formatert = df_uformatert.copy()

    # --- simple fixed-width fields ---
    if "periode" in df_formatert.columns:
        df_formatert["periode"] = df_formatert["periode"].astype("string").str.zfill(4)

    if "alder" in df_formatert.columns:
        df_formatert["alder"] = df_formatert["alder"].astype("string").str.zfill(3)

    # --- conditional padding helper (only digits & too short), dtype-safe ---
    def _conditional_pad(col: str, width: int) -> None:
        if col not in df_formatert.columns:
            return
        # Ensure the actual column (not just a temp Series) is string dtype
        df_formatert[col] = df_formatert[col].astype("string")

        # Mask: digits-only AND length < width
        mask = df_formatert[col].str.fullmatch(r"\d+") & (
            df_formatert[col].str.len() < width
        )

        # Assign using where(...) to avoid dtype-mismatch warnings/errors
        df_formatert[col] = df_formatert[col].where(
            ~mask, other=df_formatert[col].str.zfill(width)
        )

    # Apply to possible region columns (pad only where appropriate)
    region_columns = {"kommuneregion": 4, "fylkesregion": 4, "bydelsregion": 6}

    for region in region_columns:
        _conditional_pad(region, region_columns[region])

    # If none of the region columns are present, warn user
    if not any(c in df_formatert.columns for c in region_columns):
        logger.warning(f"No valid region column ({list(region_columns.keys())}) found.")
    else:
        logger.info("Formatting complete.")
    return df_formatert

def definere_klassifikasjonsvariable(
    inputfil: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Definere klassifikasjonsvariablene i datasettet.

    Dette er en funksjon der du definerer klassifikasjonsvariablene i datasettet ditt. I KOMPIS ble klassifikasjonsvariablene automatisk identifisert fordi de var forhåndsdefinert og koplet til
    en bestemt KLASS-kodeliste. Det er så langt ikke lagt til rette for dette på KOSTRA DAPLA.
    Dette er en funksjon som inngår i en annen, nemlig regionshierarkifunksjonen, så det er ikke meningen at du skal anvende denne direkte på et datasett, men det er mulig.
    For at hierarkifunksjonen skal fungere etter hensikten, er det nødvendig at du angir de klassifikasjonsvariablene som KOMMER I TILLEGG til periode- og regionsvariabelen.
    Det gjør du i et tekstfelt som dukker opp når du kjører hierarkifunksjonen.
    """
    tot_cols = inputfil.columns.tolist()
    logger.info(f"Alle variable i datasettet: {tot_cols}")

    # Always-fixed variables (keep order, include only if present)
    alltid_faste_klassifikasjonsvariable = [
        "periode",
        "kommuneregion",
        "fylkesregion",
        "bydelsregion",
    ]
    felles_klassifikasjonsvariable = [
        c for c in alltid_faste_klassifikasjonsvariable if c in tot_cols
    ]

    # Ask user for additional variables
    andre_klassifikasjonsvariable_input = input(
        f"Datasettet inneholder kostra-klassifikasjonsvariablene felles for alle datasett i kostra {felles_klassifikasjonsvariable}.\n"
        "Skriv inn andre klassifikasjonsvariable UTENOM DE OBLIGATORISKE (da trenger du ikke å skrive inn disse: 'periode', 'kommuneregion', 'fylkesregion' eller 'bydelsregion') \n"
        "som datasettet inneholder, uten anførselstegn og komma mellom hver dersom flere enn 1:\n"
        "Trykk ganske enkelt 'enter' dersom du ikke har flere klassifikasjonsvariable å legge til."
    )

    andre_klassifikasjonsvariable = [
        var.strip()
        for var in andre_klassifikasjonsvariable_input.split(",")
        if var.strip()
    ]
    if len(andre_klassifikasjonsvariable) == 0:
        logger.info("Ingen andre klassifikasjonsvariable valgt.")
    else:
        logger.info(f"Andre klassifikasjonsvariable: {andre_klassifikasjonsvariable}")

    # Helper to deduplicate while preserving order
    def uniq(seq: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # Build lists while keeping order
    klassifikasjonsvariable = uniq(
        felles_klassifikasjonsvariable + andre_klassifikasjonsvariable
    )
    statistikkvariable = [c for c in tot_cols if c not in klassifikasjonsvariable]

    logger.info(f"Klassifikasjonsvariable i datasettet: {klassifikasjonsvariable}")

    inputfil[klassifikasjonsvariable] = inputfil[klassifikasjonsvariable].astype(
        "string"
    )

    logger.info(f"Statistikkvariable i datasettet: {statistikkvariable}")

    logger.info("Oppdaterte datatyper:")
    print(inputfil.dtypes)

    return klassifikasjonsvariable, statistikkvariable

def _konvertere_komma_til_punktdesimal(inputfil: pd.DataFrame) -> pd.DataFrame:
    """Konvertere komma til punktdesimal i datasettet."""
    df = inputfil.copy()
    cols_with_commas = [
        col for col in df.columns if df[col].astype(str).str.contains(",").any()
    ]
    for col in cols_with_commas:
        df[col] = df[col].str.replace(",", ".", regex=False).astype(float)
    return df

def _hent_folkemengde_bydeler_31_12(statistikkaar: str | int) -> pd.DataFrame:
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
    folkemengde_31_12_b = hent_folkemengde_bydeler_31_12(2024)
    display(folkemengde_31_12_b)

    Feilhåndtering
    ---------------
    Hvert trinn har egen feilhåndtering. Ved suksess skrives en bekreftelse.
    Ved feil gis en advarsel og funksjonen stopper, fordi hvert trinn er
    avhengig av resultatet fra forrige trinn.
    """
    # ---------- Hent filsti ----------
    try:
        statistikkaar = str(statistikkaar)
        bucket_inndata = f"/buckets/delt-kostra-befolkning-delt/bydeler/{statistikkaar}"

        bucket_inndata_data_path = latest_version_path(
            f"{bucket_inndata}/folkmengde_bydeler_p{statistikkaar}-12-31"
        )

        logger.info(f"✅Filsti funnet: {bucket_inndata_data_path}")

    except Exception as e:
        msg = f"❌Operasjonen latest_version_path for bydeldata feilet: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    # ---------- Les parquet ----------
    try:
        folketall_bydeler = pd.read_parquet(bucket_inndata_data_path)

        logger.info(
            "✅Operasjonen innlesing av folkemengdefil for bydeler ble gjennomført."
        )

    except Exception as e:
        msg = f"❌Operasjonen innlesing av folkemengdefil for bydeler feilet: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    # ---------- Summer til KLASS-aldersgrupperinger ----------
    try:
        predefined_input = "kjonn, alder, to"

        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            df_sum_med_kjonn = (
                summere_til_aldersgrupperinger.summere_til_aldersgrupperinger(
                    folketall_bydeler,
                    hierarki_path=ALDERSHIERARKI_PATH,
                )
            )

        logger.info(
            "✅Operasjonen summere_til_aldersgrupperinger for bydeler ble gjennomført."
        )

    except Exception as e:
        msg = f"❌Operasjonen summere_til_aldersgrupperinger for bydeler feilet: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    # ---------- Summer over kjønn ----------
    try:
        predefined_input = "kjonn, alder"

        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            df_sum_kjonn = summere_kjonn.summere_over_kjonn(df_sum_med_kjonn)

        logger.info("✅Operasjonen summere_over_kjonn for bydeler ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen summere_over_kjonn for bydeler feilet: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    # ---------- Grupper over Oslo-bydelene ----------
    try:
        predefined_input = "alder"

        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            folkemengde_31_12_b = regionshierarki.hierarki(df_sum_kjonn)

        logger.info(
            "✅Operasjonen regionshierarki.hierarki for Oslo-bydeler ble gjennomført."
        )

    except Exception as e:
        msg = f"❌Operasjonen regionshierarki.hierarki for Oslo-bydeler feilet: {e}"
        logger.error(msg)
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

        logger.info(
            "✅Operasjonen filtrering av alder 105-120 for bydeler ble gjennomført."
        )

    except Exception as e:
        msg = f"❌Operasjonen filtrering av alder 105-120 for bydeler feilet: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    return folkemengde_31_12_b[
        ["periode", "bydelsregion", "alder", "personer"]
    ].reset_index(drop=True)

def _hent_folkemengde_kommune_31_12(
    statistikkaar: str | int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
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
    df_base, df_final = _hent_folkemengde_kommune_31_12(2024)

    if df_final is not None:
         display(df_final)
    else:
        print("Videre prosessering feilet - viser kun grunnlagsdata.")
        display(df_base)

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
            f"{bef_buckets_shared}/bosatte/{statistikkaar}/bosatte_p{statistikkaar}-12-31.parquet"
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
        logger.info(f"✅Kommunedata hentet for {statistikkaar}. \n")

    except Exception as e:
        msg = f"❌Kommunefil mangler eller kunne ikke leses: {e}"
        logger.warning(msg)
        mangler.append(msg)

    # ---------- Hent Svalbarddata ----------
    try:
        aar_inputmappe = int(statistikkaar) + 1
        filsti_input_svalbard = f"{bef_buckets_shared}/svalbard/{aar_inputmappe}"

        if aar_inputmappe == 2018:
            df_svalbard_data_path = latest_version_path(
                f"{filsti_input_svalbard}/svalbardbosatte_p{aar_inputmappe}-07-01"
            )
        else:
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
        logger.info(f"✅Svalbarddata hentet for statistikkår {statistikkaar}. \n")

    except Exception as e:
        msg = f"❌Svalbardfil mangler eller kunne ikke leses: {e}"
        logger.warning(msg)
        mangler.append(msg)

    # ---------- Ingen grunnlagsdata ----------
    if not dataframes:
        error_msg = (
            f"❌Ingen data tilgjengelig for statistikkår {statistikkaar}.\n"
            + "\n".join(mangler)
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # ---------- Kombiner grunnlagsdata ----------
    if len(dataframes) == 1:
        df_folkemengde = dataframes[0]
    else:
        df_folkemengde = pd.concat(dataframes, ignore_index=True)

    df_folkemengde_31_12 = df_folkemengde.groupby(
        ["periode", "kommuneregion", "kjonn", "alder"], as_index=False
    )[["personer"]].sum()

    logger.info(f"Datagrunnlag for {statistikkaar}: {', '.join(kilder_brukt)} brukt.")

    if mangler:
        logger.warning("Merk: Ett eller flere input manglet.")
        for m in mangler:
            print(f" - {m}")

    # ---------- Videre prosessering ----------
    df_folkemengde_31_12_kostra_agg_filtrert = None

    try:
        predefined_input = "kjonn, alder, to"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):

            df_folkemengde_31_12_agg_alder = (
                summere_til_aldersgrupperinger.summere_til_aldersgrupperinger(
                    df_folkemengde_31_12,
                    hierarki_path=ALDERSHIERARKI_PATH,
                )
            )

        logger.info("✅Operasjonen summere_til_aldersgrupperinger ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen summere_til_aldersgrupperinger feilet: {e}"
        logger.warning(msg)
        print(msg)
        return df_folkemengde_31_12, None

    try:
        predefined_input = "kjonn, alder"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            df_folkemengde_31_12_agg_kjonn = summere_kjonn.summere_over_kjonn(
                df_folkemengde_31_12_agg_alder
            )

        logger.info("✅Operasjonen summere_over_kjonn ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen summere_over_kjonn feilet: {e}"
        logger.warning(msg)
        print(msg)
        return df_folkemengde_31_12, None

    try:
        predefined_input = "alder"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            df_folkemengde_31_12_agg_kostra = regionshierarki.hierarki(
                df_folkemengde_31_12_agg_kjonn
            )

        logger.info("✅Operasjonen hierarki ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen hierarki feilet: {e}"
        logger.warning(msg)
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

        logger.info("✅Operasjonen filtrering av alder 105-120 ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen filtrering av alder 105-120 feilet: {e}"
        logger.warning(msg)
        print(msg)
        return df_folkemengde_31_12, None

    if df_folkemengde_31_12_kostra_agg_filtrert is not None:
        if len(kilder_brukt) == 2:
            logger.info(
                "\n✅Sluttresultat (KOSTRA-aggregert) er basert på begge datakilder: kommuner og svalbard."
            )
        else:
            logger.info(
                f"\nℹ️Sluttresultat (KOSTRA-aggregert) er kun basert på én datakilde: {kilder_brukt[0]}."
            )

    return df_folkemengde_31_12, df_folkemengde_31_12_kostra_agg_filtrert

def _hent_folkemengde_fylkeskommune_31_12(statistikkaar: str | int) -> pd.DataFrame:
    """Henter, aggregerer og grupperer folkemengdedata per 31.12 for et gitt statistikkår.

    Funksjonen bygger på `_hent_folkemengde_kommune_31_12`, og returnerer et
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
    folkemengde_31_12_eafk = hent_folkemengde_31_12_eafk(2024)
    display(folkemengde_31_12_eafk)

    Feilhåndtering
    ---------------
    Hvert trinn har egen feilhåndtering. Ved feil gis en advarsel som navngir
    operasjonen som feilet, og funksjonen stopper fordi senere trinn er avhengige
    av tidligere trinn.
    """
    try:
        df_folkemengde_31_12, _ = _hent_folkemengde_kommune_31_12(statistikkaar)
        logger.info("✅Operasjonen _hent_folkemengde_kommune_31_12 ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen _hent_folkemengde_kommune_31_12 feilet: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    try:
        predefined_input = "kjonn, alder, to"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            df_sum_med_kjonn = (
                summere_til_aldersgrupperinger.summere_til_aldersgrupperinger(
                    df_folkemengde_31_12,
                    hierarki_path=ALDERSHIERARKI_PATH,
                )
            )

        logger.info("✅Operasjonen summere_til_aldersgrupperinger ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen summere_til_aldersgrupperinger feilet: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    try:
        predefined_input = "kjonn, alder"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            df_sum_kjonn = summere_kjonn.summere_over_kjonn(df_sum_med_kjonn)

        logger.info("✅Operasjonen summere_over_kjonn ble gjennomført.")

    except Exception as e:
        msg = f"❌Operasjonen summere_over_kjonn feilet: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    try:
        predefined_input = "alder"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            folkemengde_31_12_fk = regionshierarki.hierarki(
                df_sum_kjonn, aggregeringstype="kommune_til_fylkeskommune"
            )

        logger.info(
            "✅Operasjonen hierarki med aggregeringstype='kommune_til_fylkeskommune' ble gjennomført."
        )

    except Exception as e:
        msg = (
            "Operasjonen hierarki med aggregeringstype='kommune_til_fylkeskommune' "
            f"feilet: {e}"
        )
        logger.error(msg)
        raise RuntimeError(msg) from e

    try:
        predefined_input = "alder"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            folkemengde_31_12_eafk = regionshierarki.hierarki(folkemengde_31_12_fk)

        logger.info(
            "✅Operasjonen hierarki til KOSTRA-fylkesregioner/EAFK ble gjennomført."
        )

    except Exception as e:
        msg = f"❌Operasjonen hierarki til KOSTRA-fylkesregioner/EAFK feilet: {e}"
        logger.error(msg)
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

        logger.info(
            "✅Operasjonen filtrering av alder 105-120 og F025-029 ble gjennomført."
        )

    except Exception as e:
        msg = f"❌Operasjonen filtrering av alder 105-120 og F025-029 feilet: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    return folkemengde_31_12_eafk

def _mapping_mellom_aar(
    statistikkaar: int | str,
    regionsnivaa: str = "kommune",
) -> pd.DataFrame:
    statistikkaar_int = int(statistikkaar)
    kildeaar = str(statistikkaar_int - 1)
    regionsnivaa = regionsnivaa.lower()
    """Henter endringsmapping mellom to påfølgende år for valgt regionsnivå.

    Funksjonen henter endringsloggen fra KLASS for overgangen mellom
    kildeåret (t-1) og statistikkåret (t). Endringsloggen brukes ved
    produksjon av testdatasett for å oversette regionkoder fra kildeåret
    til statistikkåret.

    For bydeler returneres kun endringer som gjelder Oslo-bydeler
    (regionkoder som starter med "03"). Dersom det ikke finnes relevante
    endringer, returneres en tom DataFrame med forventede kolonner.

    Parametere
    ----------
    statistikkaar : int | str
        Statistikkåret datasettet skal gjelde for.

    regionsnivaa : str, default="kommune"
        Geografisk nivå det skal hentes endringsmapping for.
        Gyldige verdier er "kommune", "bydel" og "fylkeskommune".

    Returverdi
    ----------
    pd.DataFrame
        En DataFrame med én rad per registrerte regionendring og kolonnene

        - kildeaar
        - oldCode
        - oldName
        - oldShortName
        - statistikkaar
        - newCode
        - newName
        - newShortName
        - changeOccurred

        Returnerer en tom DataFrame med de samme kolonnene dersom ingen
        relevante endringer finnes eller dersom endringsloggen ikke kan
        hentes.
    """

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
        klass_kode = "131"
        endringstekst = "kommuneendringer"
    elif regionsnivaa == "bydel":
        klass_kode = "103"
        endringstekst = "bydelsendringer"
    elif regionsnivaa == "fylkeskommune":
        klass_kode = "127"
        endringstekst = "fylkeskommuneendringer"
    else:
        raise ValueError(
            "❌regionsnivaa må være 'bydel', 'kommune' eller 'fylkeskommune'."
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
            f"ℹ️Ingen endringslogg funnet for {kildeaar}-{statistikkaar_int}. "
            "Returnerer tom mapping."
        )
        return pd.DataFrame(columns=kolonner)

    if df_endringer is None or df_endringer.empty:
        logger.info(
            f"ℹ️Ingen {endringstekst} funnet for {kildeaar}-{statistikkaar_int}."
        )
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
                f"{kildeaar}-{statistikkaar_int}."
            )
            return pd.DataFrame(columns=kolonner)

    df_endringer["kildeaar"] = kildeaar
    df_endringer["statistikkaar"] = str(statistikkaar_int)

    manglende_kolonner = [col for col in kolonner if col not in df_endringer.columns]

    if manglende_kolonner:
        logger.warning(
            f"❌Endringsloggen mangler forventede kolonner: {manglende_kolonner}"
        )
        return pd.DataFrame(columns=kolonner)

    return df_endringer[kolonner]

def _regionkolonne(regionsnivaa: str) -> str:
    """Returnerer navnet på regionkolonnen for et gitt regionsnivå.

    Parametere
    ----------
    regionsnivaa : str
        Geografisk nivå. Gyldige verdier er "kommune", "bydel"
        og "fylkeskommune".

    Returverdi
    ----------
    str
        Navnet på kolonnen som inneholder regionkodene:

        - "kommuneregion" for kommuner
        - "bydelsregion" for bydeler
        - "fylkesregion" for fylkeskommuner

    Reiser
    ------
    ValueError
        Dersom regionsnivaa ikke er en gyldig verdi.
    """
    if regionsnivaa == "kommune":
        return "kommuneregion"
    if regionsnivaa == "bydel":
        return "bydelsregion"
    if regionsnivaa == "fylkeskommune":
        return "fylkesregion"

    raise ValueError("❌regionsnivaa må være 'bydel', 'kommune' eller 'fylkeskommune'.")

def _normaliser_regionkode(verdi: object, regionsnivaa: str) -> str:
    """Normaliserer en regionkode til forventet strengformat.

    Numeriske regionkoder konverteres til strenger med ledende nuller.
    Bydelskoder formateres til seks sifre, mens kommune- og
    fylkeskommunekoder formateres til fire sifre. Ikke-numeriske
    regionkoder returneres uendret.

    Parametere
    ----------
    verdi : object
        Regionkode som skal normaliseres.

    regionsnivaa : str
        Geografisk nivå. Gyldige verdier er "kommune", "bydel"
        og "fylkeskommune".

    Returverdi
    ----------
    str
        Den normaliserte regionkoden.
    """
    regionsnivaa = regionsnivaa.lower()

    if regionsnivaa not in {"kommune", "bydel", "fylkeskommune"}:
        raise ValueError(
            "❌regionsnivaa må være 'bydel', 'kommune' eller 'fylkeskommune'."
        )

    kode = str(verdi).strip()

    if not kode.isdigit():
        return kode

    if regionsnivaa == "bydel":
        return kode.zfill(6)

    return kode.zfill(4)

def _anvende_kommunereform(
    inputfil: pd.DataFrame,
    mapping: pd.DataFrame | None,
    statistikkvariable: list[str],
    statistikkaar: int | str,
    regionsnivaa: str,
) -> pd.DataFrame:
    df = inputfil.copy()
    regionsnivaa = regionsnivaa.lower()
    regionkolonne = _regionkolonne(regionsnivaa)

    """Anvender regionendringer på et testdatasett fra året før.

    Funksjonen tilpasser et datasett fra kildeåret (t-1) til valgt
    statistikkår (t) ved hjelp av en endringsmapping fra KLASS. Den kan
    brukes for kommune-, bydels- og fylkeskommunenivå.

    Regionkodene normaliseres til forventet lengde, og verdien i kolonnen
    ``periode`` oppdateres til statistikkåret. Regioner som finnes i
    endringsmappingen, får regionkoden erstattet med tilhørende ny kode.
    Regioner som ikke finnes i mappingen, kopieres uendret.

    Funksjonen identifiserer og logger både regionsoppdelinger, der én
    tidligere regionkode er koblet til flere nye regionkoder, og
    regionssammenslåinger, der flere tidligere regionkoder er koblet til
    samme nye regionkode.

    Etter at mappingen er anvendt, kontrolleres det om flere rader har fått
    samme kombinasjon av klassifikasjonsvariabler. Slike rader aggregeres
    ved å summere de angitte statistikkvariablene.

    Parametere
    ----------
    inputfil : pd.DataFrame
        Datasettet fra kildeåret som regionendringene skal anvendes på.
        Datasettet må inneholde riktig regionkolonne for valgt regionsnivå
        og en kolonne med navnet ``periode``.

    mapping : pd.DataFrame | None
        Endringsmapping mellom kildeåret og statistikkåret. Mappingen
        forventes å inneholde kolonnene ``oldCode`` og ``newCode``.
        Dersom mappingen er ``None`` eller tom, beholdes regionkodene
        uendret, og bare ``periode`` oppdateres.

    statistikkvariable : list[str]
        Navn på de numeriske statistikkvariablene som skal summeres dersom
        regionendringene fører til dupliserte klassifikasjonsnøkler.

    statistikkaar : int | str
        Året testdatasettet skal gjelde for. Kildeåret beregnes som
        statistikkåret minus ett.

    regionsnivaa : str
        Geografisk nivå som mappingen skal anvendes på. Gyldige verdier er
        ``"kommune"``, ``"bydel"`` og ``"fylkeskommune"``.

    Returverdi
    ----------
    pd.DataFrame
        Et nytt datasett der regionkodene er tilpasset statistikkåret,
        ``periode`` er oppdatert, og eventuelle dupliserte
        klassifikasjonsnøkler er aggregert.

    Reiser
    ------
    ValueError
        Dersom ``regionsnivaa`` ikke er ``"kommune"``, ``"bydel"`` eller
        ``"fylkeskommune"``.

    KeyError
        Dersom forventet regionkolonne, ``oldCode``, ``newCode`` eller en
        angitt statistikkvariabel mangler.
    """

    df[regionkolonne] = df[regionkolonne].map(
        lambda x: _normaliser_regionkode(x, regionsnivaa)
    )
    df["periode"] = str(statistikkaar)

    kildeaar = str(int(statistikkaar) - 1)
    logger.info(f"ℹ️Kildeåret for testdatasettet ditt er {kildeaar}.")

    if mapping is None or mapping.empty:
        logger.info(
            f"ℹ️Ingen regionendringer funnet mellom {kildeaar} og {statistikkaar}. Kopierer region uendret og oppdaterer periode.\n"
            # "Kopierer region uendret og oppdaterer periode."
        )
        return df

    mapping = mapping.copy()
    mapping["oldCode"] = mapping["oldCode"].map(
        lambda x: _normaliser_regionkode(x, regionsnivaa)
    )
    mapping["newCode"] = mapping["newCode"].map(
        lambda x: _normaliser_regionkode(x, regionsnivaa)
    )

    split_details = (
        mapping.groupby("oldCode")["newCode"]
        .apply(lambda x: sorted(set(x)))
        .loc[lambda s: s.str.len() > 1]
        .to_dict()
    )

    merger_details = (
        mapping.groupby("newCode")["oldCode"]
        .apply(lambda x: sorted(set(x)))
        .loc[lambda s: s.str.len() > 1]
        .to_dict()
    )

    if split_details:
        logger.info(
            f"ℹ️Regionsoppdelinger funnet mellom {kildeaar} og {statistikkaar}: {split_details}"
        )

    if merger_details:
        logger.info(
            f"ℹ️Regionssammenslåinger funnet mellom {kildeaar} og {statistikkaar}: {merger_details}"
        )

    changed_old_codes = set(mapping["oldCode"])
    input_codes = set(df[regionkolonne])

    unmapped_codes = sorted(input_codes - changed_old_codes)
    if unmapped_codes:
        logger.info(
            f"ℹ️{len(unmapped_codes)} regioner finnes ikke i endringsloggen "
            "og kopieres uendret."
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
            "ℹ️Dupliserte klassifikasjonsnøkler funnet etter regionendring. "
            "Aggregerer statistikkvariable."
        )

        df_mapped = df_mapped.groupby(klassifikasjonsvariable, as_index=False)[
            statistikkvariable
        ].sum()
    else:
        logger.info("ℹ️Ingen aggregering nødvendig etter regionendring.")

    return df_mapped
