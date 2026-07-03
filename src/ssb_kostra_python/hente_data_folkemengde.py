INPUT_PATCH_TARGET = "builtins.input"
import pandas as pd
from fagfunksjoner import logger

from ssb_kostra_python import hjelpefunksjoner


def hente_data_folkemengde(
    aar: int, regionsnivaa: str, testdata: bool = False
) -> pd.DataFrame:
    """Henter et folkemengdedatasett 31.12 for angitt år og regionsnivå.

    Funksjonen kan enten hente reelle folkemengdedata for året som er angitt,
    eller lage et testdatasett for samme år basert på data fra foregående år.
    Hvis testdata=True, hentes data fra år t-1, og verdien i kolonnen
    'periode' endres til år t. Hvis testdata=False, hentes data fra år t
    uten at periodeverdien endres.

    Parametere
    ----------
    aar : int
        Året datasettet skal gjelde for.

    regionsnivaa : str
        Geografisk nivå som datasettet skal hentes for.
        Gyldige verdier er:
        - 'bydel'
        - 'kommune'
        - 'fylkeskommune'

    testdata : bool, default False
        Angir om funksjonen skal lage testdata.
        Hvis False, hentes reelle data for året angitt i aar.
        Hvis True, hentes data fra foregående år, mens periodekolonnen
        oppdateres til året angitt i aar.

    Returnerer
    ----------
    pandas.DataFrame
        Et folkemengdedatasett for valgt år og regionsnivå. Hvis testdata=True,
        er observasjonene hentet fra år t-1, men periodeverdien er endret til
        år t. Hvis testdata=False, er observasjonene hentet fra år t.

    Merknader
    ----------
    Ved kommunereformer eller fylkesreformer mellom år t-1 og t vil et
    testdatasett være basert på den gamle geografiske inndelingen. Dette kan
    gi problemer ved sammenslåing med datasett som benytter ny
    inndelingsstruktur.

    For Oslo-bydeler er dette foreløpig ikke et problem i KOSTRA-tidsserien.

    Eksempler
    ---------
    Hent reelle data for bydeler i 2024:

    datasett = hente_data_folkemengde(2024, "bydel")

    Lag testdata for kommuner i 2026 basert på 2025-data:

    testdatasett = hente_data_folkemengde(2026, "kommune", testdata=True)
    display(testdatasett)

    Reiser
    ------
    ValueError
        Hvis regionsnivaa ikke er 'bydel', 'kommune' eller 'fylkeskommune'.
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

    if testdata:
        kildeaar = str(aar - 1)
        print(
            f"ℹ️Du har satt regionsnivået til \033[1m{regionsnivaa}\033[0m. Du har satt årgang til \033[1m{aar}\033[0m. Befolkningsdataene hentes fra den foregående årgangen \033[1m{kildeaar}\033[0m. \033[1m{kildeaar}\033[0m byttes ut med \033[1m{aar}\033[0m i periodekolonnen.\n"
        )
    else:
        kildeaar = str(aar)
        print(
            f"ℹ️Du har satt regionsnivået til \033[1m{regionsnivaa}\033[0m. Du har satt årgang til \033[1m{aar}\033[0m. Befolkningsdataene hentes fra den samme årgangen \033[1m{kildeaar}\033[0m. \n"
        )

    if regionsnivaa.lower() == "bydel":
        folkemengde_31_12 = hjelpefunksjoner._hent_folkemengde_bydeler_31_12(
            int(kildeaar)
        )

    elif regionsnivaa.lower() == "kommune":
        _, folkemengde_kommune = hjelpefunksjoner._hent_folkemengde_kommune_31_12(
            int(kildeaar)
        )

        if folkemengde_kommune is None:
            error_msg = (
                f"Klarte ikke å lage KOSTRA-aggregert folkemengdefil for {kildeaar}."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        folkemengde_31_12 = folkemengde_kommune

    elif regionsnivaa.lower() == "fylkeskommune":
        folkemengde_31_12 = hjelpefunksjoner._hent_folkemengde_fylkeskommune_31_12(
            int(kildeaar)
        )
    else:
        error_msg = "Du må angi regionsnivå som 'bydel', 'kommune' eller 'fylkeskommune'. Eksempel: hente_data_folkemengde(2024, 'bydel') eller datasett = hente_data_folkemengde(2024, 'bydel')"
        logger.error(error_msg)
        raise ValueError(error_msg)
    folkemengde_31_12_data = folkemengde_31_12.copy()
    if testdata:
        folkemengde_31_12_data.loc[
            folkemengde_31_12_data["periode"] == kildeaar, "periode"
        ] = str(aar)
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
