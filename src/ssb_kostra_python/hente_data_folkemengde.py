INPUT_PATCH_TARGET = "builtins.input"
import pandas as pd
from fagfunksjoner.fagfunksjoner_logger import logger
from IPython.display import display

from ssb_kostra_python import hjelpefunksjoner
from ssb_kostra_python import regionshierarki

INPUT_PATCH_TARGET = "builtins.input"
from unittest.mock import patch


def hente_data_folkemengde(
    aar: int, regionsnivaa: str, testdata: bool = False
) -> pd.DataFrame:
    """Henter eller lager folkemengdedata per 31.12.

    Funksjonen henter et folkemengdedatasett for angitt år og regionsnivå.
    Den kan enten hente reelle data for året eller lage et testdatasett for
    året basert på de nyeste tilgjengelige dataene fra foregående år.

    Hvis ``testdata=False``, hentes reelle data for året angitt i ``aar``.

    Hvis ``testdata=True``, hentes data fra år t-1, mens datasettet tilpasses
    år t. Verdien i kolonnen ``periode`` oppdateres til år t, og eventuell
    geografisk reform mellom t-1 og t anvendes på datasettet.

    Parametere
    ----------
    aar : int
        Året det returnerte datasettet skal gjelde for.

    regionsnivaa : str
        Geografisk nivå for datasettet.

        Gyldige verdier er:

        - ``"bydel"``
        - ``"kommune"``
        - ``"fylkeskommune"``

    testdata : bool, default False
        Angir om funksjonen skal hente reelle data eller lage testdata.

        Hvis ``False``, hentes reelle data for året angitt i ``aar``.

        Hvis ``True``, hentes data fra året før ``aar``. Deretter oppdateres
        periodeverdien og datasettet tilpasses den geografiske strukturen
        som gjelder i ``aar``.

    Returnerer
    ----------
    pandas.DataFrame
        Et folkemengdedatasett for valgt år og regionsnivå.

        Hvis ``testdata=False``, inneholder datasettet reelle data for året
        angitt i ``aar``.

        Hvis ``testdata=True``, er statistikkverdiene basert på data fra
        foregående år, men perioden og den geografiske inndelingen er
        tilpasset året angitt i ``aar``.

        Eksisterende KOSTRA-grupper fjernes før regionshierarkiet bygges opp
        på nytt.

    Merknader
    ----------
    Endringsmappingen som brukes ved ``testdata=True``, er en endringslogg
    og ikke en fullstendig korrespondansetabell. Den inneholder derfor bare
    regioner som faktisk har blitt endret mellom t-1 og t.

    Regioner som ikke finnes i endringsloggen, kopieres uendret til
    testdatasettet.

    Hvis det ikke har skjedd noen geografiske endringer mellom årene,
    returneres den samme geografiske strukturen som i kildeåret, mens
    ``periode`` oppdateres til året angitt i ``aar``.

    Geografiske endringer behandles etter følgende regler:

    En-til-en-endring
        Hvis én region har én etterfølger, overføres observasjonene direkte
        til den nye regionkoden. Dette gjelder både rene kodeendringer,
        navneendringer og kombinerte kode- og navneendringer.

    Oppdeling
        Hvis en region deles opp i flere, kopieres alle
        observasjonene fra den tidligere regionen til hver av de nye
        regionene. Statistikkverdiene fordeles altså ikke mellom etterfølgerne.


    Sammenslåing
        Hvis flere tidligere regioner får samme nye regionkode, erstattes de
        gamle kodene med den nye koden. Observasjoner som deretter har samme
        kombinasjon av klassifikasjonsvariabler, aggregeres ved å summere
        statistikkvariablene.

    Mange-til-mange-endring
        Hvis flere tidligere regioner inngår i flere nye regioner, behandles
        dette som en kompleks reform. Funksjonen logger en tydelig advarsel,
        fordi resultatet ikke nødvendigvis representerer en entydig
        statistisk fordeling.

    Testdataene skal først og fremst ha korrekt struktur slik at etterfølgende
    programmer kan kjøres før reelle data for året er tilgjengelige.
    Statistikktallene skal derfor ikke tolkes som anslag for den faktiske
    folkemengden etter en geografisk reform.

    Eksempler
    ---------
    Hent reelle bydelsdata for 2024:

    datasett = hente_data_folkemengde(
         2024,
         "bydel",
    )

    Lag kommunetestdata for 2026 basert på data fra 2025:

    testdatasett = hente_data_folkemengde(
         2026,
         "kommune",
         testdata=True,
    )

    Lag fylkeskommunale testdata for 2026:

    testdatasett = hente_data_folkemengde(
         2026,
         "fylkeskommune",
         testdata=True,
    )

    Reiser
    ------
    ValueError
        Hvis ``regionsnivaa`` ikke er ``"bydel"``, ``"kommune"`` eller
        ``"fylkeskommune"``.

    RuntimeError
        Hvis funksjonen ikke klarer å opprette det KOSTRA-aggregerte
        kommunedatasettet for kildeåret.
    """
    print(
        "ℹ️Med denne funksjonen henter du et folkemengdedatasett 31.12 for året og regionsnivået du angir. \n"
    )
    print(
        "ℹ️Du kan hente et cloud-lagret datasett for år t der befolkningsfilene ligger klare.  Om du trenger å lage testdata for t+1 før befolkningsfilene er klare i skyen, som KOSTRA bruker å gjøre i november for det påfølgende året, kan du også gjøre det. \n"
    )
    print(
        "ℹ️Husk å angi året datasettet skal gjelde og også regionsnivå med bydel, kommune eller fylkeskommune for at funksjonen skal hente riktig folkemengdefil."
    )
    print(
        "ℹ️Du må også angi om du skal hente et datasett for t eller et testdatasett for t+1. \033[1mFalse\033[0m betyr at du ønsker å hente et eksisterende datasett for t. Med \033[1mTrue\033[0m henter du et datasett for t, og deretter settes perioden til t+1.\n"
    )

    print(
        "ℹ️Om det har vært kommunereform mellom t og t+1, vil du få befolkningsdata på gammel inndeling i t."
    )
    print(
        "ℹ️Da er det kanskje ikke hensiktmessig å generere dette datasettet om du skal slå det sammen med andre datasett på ny kommunestruktur."
    )
    print(
        "ℹ️Dette problemet gjelder foreløpig bare kommuner og fylkeskommuner. For Oslo-bydelene har det ikke vært reform i KOSTRA-tidsserien.\n"
    )
    print(
        f"ℹ️Om du har skrevet funksjonen som dette - \033[1mhente_data_folkemengde({aar}, {regionsnivaa}, True)\033[0m - har du ikke lagret datasettet i et objekt selv om det vises nederst."
    )
    print(
        f"ℹ️Du har i så fall et {aar}-datasett for {regionsnivaa}, og det siste argumentet \033[1mTrue\033[0m gjør at du får et testdatasett der befolkningsdataene fra året før er brukt.\n"
    )
    print(
        f"ℹ️Men om du har skrevet funksjonen som dette - \033[1mtestdatasett = hente_testdata_folkemengde({aar}, {regionsnivaa})\033[0m - har du lagret datasettet som en dataframe."
    )
    print(
        f"Da har du her et kommunedatasett for {aar} med inputdata for {aar}. Å skrive \033[1mdatasett = hente_data_folkemengde({aar}, {regionsnivaa})\033[0m og \033[1mdatasett = hente_data_folkemengde({aar}, {regionsnivaa}, False)\033[0m  er likestilt."
    )
    print(
        "ℹ️Da vil derimot ikke testdatasettet vises. Skriv ganske enkelt \033[1mdisplay(datasett)\033[0m i tillegg, og så vil du få en visning."
    )
    print(
        "ℹ️Navnet til venstre for likhetstegnet er det du kaller den lagrede dataframen. Den kan du kalle det du vil. \n"
    )

    regionsnivaa = regionsnivaa.lower()
    statistikkaar = int(aar)

    if testdata:
        kildeaar = statistikkaar - 1
        logger.info(
            f"ℹ️Lager testdata for {statistikkaar} basert på data fra {kildeaar}."
        )
    else:
        kildeaar = statistikkaar
        logger.info(f"ℹ️Henter reelle data for {statistikkaar}.")

    if regionsnivaa == "bydel":
        folkemengde_31_12 = hjelpefunksjoner._hent_folkemengde_bydeler_31_12(kildeaar)

    elif regionsnivaa == "kommune":
        _, folkemengde_kommune = hjelpefunksjoner._hent_folkemengde_kommune_31_12(
            kildeaar
        )

        if folkemengde_kommune is None:
            error_msg = (
                f"Klarte ikke å lage KOSTRA-aggregert folkemengdefil for {kildeaar}."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        folkemengde_31_12 = folkemengde_kommune

    elif regionsnivaa == "fylkeskommune":
        folkemengde_31_12 = hjelpefunksjoner._hent_folkemengde_fylkeskommune_31_12(
            kildeaar
        )
    else:
        error_msg = (
            "❌Du må angi regionsnivå som 'bydel', 'kommune' eller 'fylkeskommune'."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    folkemengde_31_12_data = folkemengde_31_12.copy()

    if testdata:
        mapping = hjelpefunksjoner._mapping_mellom_aar(statistikkaar, regionsnivaa)

        folkemengde_31_12_data = hjelpefunksjoner._anvende_kommunereform(
            inputfil=folkemengde_31_12_data,
            mapping=mapping,
            statistikkvariable=["personer"],
            statistikkaar=statistikkaar,
            regionsnivaa=regionsnivaa,
        )
        # logger.info("ℹ️Dette er endringsmappingen:\n")
        # display(mapping)

        # if mapping is None:
        # if mapping.empty:
        if mapping is None or mapping.empty:
            logger.info(
                "ℹ️Under vises endringsmappingen. ⇩ ⇩ Kun kolonneoverskrifter betyr at det ikke har funnet sted endringer mellom periodene.\n"
            )
            display(mapping)
            logger.info(
                f"ℹ️ ⇧ Tomt mappingdatasett ⇧. Altså ingen regionsendringer mellom {kildeaar} og {statistikkaar}.\n"
            )
        else:
            logger.info(
                "ℹ️Under vises endringsmappingen. ⇩ ⇩ Kun kolonneoverskrifter betyr at det ikke har funnet sted endringer mellom periodene.\n"
            )
            display(mapping)
            logger.info(
                f"ℹ️ ⇧ Mappingdatasettet ovenfor ⇧ viser endringer mellom {kildeaar} og {statistikkaar}.\n"
            )

    if regionsnivaa == "kommune":
        regionkolonne = "kommuneregion"
        fjern_mask = (
            folkemengde_31_12_data[regionkolonne]
            .astype(str)
            .str.match(r"^(EKG\d{2}|EKA\d{2}|EAK|EAKUO)$")
        )

    elif regionsnivaa == "bydel":
        regionkolonne = "bydelsregion"
        fjern_mask = folkemengde_31_12_data[regionkolonne].astype(str).eq("EAB")

    elif regionsnivaa == "fylkeskommune":
        regionkolonne = "fylkesregion"
        fjern_mask = (
            folkemengde_31_12_data[regionkolonne].astype(str).isin(["EAFK", "EAFKUO"])
        )

    logger.info("✅Fjerner KOSTRA-grupperingene før de legges på igjen.")

    folkemengde_31_12_data_uten_agg = folkemengde_31_12_data.loc[~fjern_mask].copy()

    predefined_input = "alder"
    with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
        folkemengde_31_12_data = regionshierarki.hierarki(
            folkemengde_31_12_data_uten_agg
        )

    logger.info(
        f"\nℹ️Om du har kjørt funksjonen for eksempel slik - \033[1mhente_data_folkemengde({aar}, {regionsnivaa}, False)\033[0m - vil dataene vises under, med årgang {aar} i periodekolonnen, hentet fra et {kildeaar}-datasett."
    )
    logger.info(
        "ℹ️Siden du da har satt \033[1mFalse\033[0m for testdatasett, har du hentet befolkningsdata fra sky for det samme året som tabellen du har generert gjelder.\n"
    )
    logger.info(
        f"ℹ️Om du har kjørt den slik - \033[1mdatasett = hente_data_folkemengde({aar}, {regionsnivaa}, True)\033[0m - vil ikke dataene vises under, men datasettet er i dette eksemplet lagret med navnet \033[1mdatasett\033[0m, det vil si objektet til venstre for likhetstegnet. Bare legg til \033[1mdisplay(datasett)\033[0m for å se datasettet."
    )
    logger.info(
        f"ℹ️Du har i dette tilfellet satt \033[1mTrue\033[0m for testdatasett. Det vil si at dataene er hentet fra fila for året før {aar} for regionsnivået {regionsnivaa}, men at datasettet skal vise {aar} i periodekolonnen.\n"
    )

    return folkemengde_31_12_data
