from collections.abc import Callable
from typing import Any
from typing import cast
from IPython.display import display

import pandas as pd
from fagfunksjoner.fagfunksjoner_logger import logger
from klass import KlassClassification
from klass import KlassCorrespondence
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_float_dtype
from pandas.api.types import is_integer_dtype

from ssb_kostra_python import hjelpefunksjoner
from ssb_kostra_python.titler_til_klasskoder import mapping_regionsnavn


def _select_mapping(
    aggregeringstype: str | None, region_col: str, periode: str | int
) -> tuple[
    pd.DataFrame,
    str,
    str,
    Callable[[pd.DataFrame], pd.DataFrame] | None,
    dict[str, str],
]:
    valid_by_region = {
        "kommuneregion": {"kommune_til_landet", "kommune_til_fylkeskommune"},
        "fylkesregion": {"fylkeskommune_til_kostraregion"},
        "bydelsregion": {"bydeler_til_EAB"},
    }
    if aggregeringstype is None:
        if region_col == "kommuneregion":
            aggregeringstype = "kommune_til_landet"
        elif region_col == "fylkesregion":
            aggregeringstype = "fylkeskommune_til_kostraregion"
        else:
            aggregeringstype = "bydeler_til_EAB"
    allowed = valid_by_region.get(region_col, set())
    if aggregeringstype not in allowed:
        raise ValueError(
            f"Inkonsekvent valg: aggregeringstype='{aggregeringstype}' passer ikke med regionkolonne '{region_col}'. "
            f"Tillatte valg for {region_col}: {sorted(allowed)}."
        )
    if aggregeringstype == "kommune_til_landet":
        return (
            mapping_fra_kommune_til_landet(periode),
            "kommuneregion",
            "kommuneregion",
            None,
            {},
        )
    if aggregeringstype == "kommune_til_fylkeskommune":

        def _post_filter_kommuner_til_fylke(df: pd.DataFrame) -> pd.DataFrame:
            return df[df["kommuneregion"].str.endswith("00")]

        return (
            mapping_fra_kommune_til_fylkeskommune(periode),
            "kommuneregion",
            "kommuneregion",
            _post_filter_kommuner_til_fylke,
            {"kommuneregion": "fylkesregion"},
        )
    if aggregeringstype == "fylkeskommune_til_kostraregion":
        return (
            mapping_fra_fylkeskommune_til_kostraregion(periode),
            "fylkesregion",
            "fylkesregion",
            None,
            {},
        )
    if aggregeringstype == "bydeler_til_EAB":
        return mapping_bydeler_oslo(periode), "bydelsregion", "bydelsregion", None, {}
    raise ValueError(
        f"Ukjent aggregeringstype: {aggregeringstype}. "
        "Gyldige: 'kommune_til_landet', 'kommune_til_fylkeskommune', "
        "'fylkeskommune_til_kostraregion', 'bydeler_til_EAB'."
    )


def _validate_and_normalize_region_col(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    region_cols = [
        c for c in ["kommuneregion", "fylkesregion", "bydelsregion"] if c in df.columns
    ]
    if len(region_cols) != 1:
        if len(region_cols) == 0:
            raise ValueError(
                "Fant ingen gyldig regionkolonne ('kommuneregion', 'fylkesregion', 'bydelsregion'). Datasettet ditt må inneholde minst én."
            )
        else:
            raise ValueError(
                f"Fant flere regionskolonner {region_cols}. Det skal være nøyaktig én."
            )
    col = region_cols[0]
    if col == "kommuneregion":
        df[col] = df[col].astype(str).str.zfill(4)
    elif col == "fylkesregion":
        df[col] = df[col].astype(str).str.zfill(4)
    else:
        df[col] = df[col].astype(str).str.zfill(6)
    return col, df


def _postprocess_combined(
    df: pd.DataFrame,
    post_filter: Callable[[pd.DataFrame], pd.DataFrame] | None,
    rename_cols: dict[str, str],
    klassifikasjonsvariable: list[str],
    add_region_names: bool,
) -> pd.DataFrame:
    df[klassifikasjonsvariable] = df[klassifikasjonsvariable].astype(str)
    if post_filter:
        df = post_filter(df)
    if rename_cols:
        df = df.rename(columns=rename_cols)
    if add_region_names:
        return mapping_regionsnavn(df.reset_index(drop=True))
    else:
        return df.reset_index(drop=True)


def _print_dtype_report(
    original: dict[str, Any],
    post_op: dict[str, Any],
    final: dict[str, Any],
    cols: list[str],
) -> None:
    print("\nOriginal dtypes:")
    for c, dt in original.items():
        print(f"  {c}: {dt}")
    print("\nPost-op (pre-restore) dtypes:")
    for c, dt in post_op.items():
        print(f"  {c}: {dt}")
    print("\nFinal dtypes (after restore):")
    for c, dt in final.items():
        print(f"  {c}: {dt}")
    post_changes = {
        c: (original[c], post_op[c]) for c in cols if original[c] != post_op[c]
    }
    final_changes = {
        c: (original[c], final[c]) for c in cols if original[c] != final[c]
    }
    if post_changes:
        print("\nDtype changes caused by the operation (original -> post-op):")
        for c, (o, p) in post_changes.items():
            print(f"  {c}: {o} -> {p}")
    else:
        print("\nNo dtype changes caused by the operation.")
    if final_changes:
        print("\nDtype changes remaining after restoration (original -> final):")
        for c, (o, f) in final_changes.items():
            print(f"  {c}: {o} -> {f}")
    else:
        print("\nNo dtype changes remain after restoration.")


def _restore_dtype(result: Any, orig: Any) -> Any:
    if is_integer_dtype(orig):
        rounded = result.round(0)
        return (
            rounded.astype(_nullable_int_for(orig))
            if rounded.isna().any()
            else rounded.astype(orig)
        )
    if is_float_dtype(orig):
        return result.astype(orig)
    if is_bool_dtype(orig):
        try:
            return (result != 0).astype("boolean")
        except TypeError:
            return (result != 0).astype(bool)
    return result


def mapping_bydeler_oslo(year: str | int = "2015") -> pd.DataFrame:
    """Mapping av bydelene i Oslo.

    Denne funksjonen er ikke en funksjon du skal anvende direkte på et datasett.
    Her lages kun en mappingfil som viser hvordan Oslos bydeler inngår i samlebydelen "EAB".
    Dette er altså bare en hjelpefunksjon som inngår i en annen funksjon, hierarkifunksjonen, som aggregerer opp bydelsdata til "EAB" kun dersom inputfilen er en bydelsfil.
    """
    nus = KlassClassification("241", language="nb", include_future=True)
    nuskoder = nus.get_codes(f"{year}-01-01")
    ### Oppretter en korrespondansetabell for bydelene i Oslo som tilbyr tjenester
    klass_bydeler_oslo = nuskoder.pivot_level()
    ### Skiller ut bydeler som ikke tilbyr tjenester pluss samlebydelen 'EAB'
    # klass_bydeler_oslo = klass_bydeler_oslo[~klass_bydeler_oslo["code_1"].isin(['030116', '030117', '030199', 'EAB'])]
    klass_bydeler_oslo = klass_bydeler_oslo[
        ~klass_bydeler_oslo["code_1"].isin(["030116", "030117", "EAB"])
    ]
    klass_bydeler_oslo = klass_bydeler_oslo[["code_1"]].rename(
        columns={"code_1": "from"}
    )
    klass_bydeler_oslo["to"] = "EAB"
    return klass_bydeler_oslo


def mapping_fra_kommune_til_landet(year: str | int) -> pd.DataFrame:
    """Mapping av kommunene til landet.

    Denne funksjonen er ikke en funksjon du skal anvende direkte på et datasett.
    Her lages kun en mappingfil som viser hvordan kommunene inngår i fylker, Kostra-grupper og landet et bestemt år.
    Dette er altså bare en hjelpefunksjon som inngår i annen funksjon, hierarkifunksjonen, som aggregerer opp kommunedata til de forskjellige regionsgrupperingene kun dersom inputfilen er en kommunefil.
    """
    komm_fylk_korr_corr: KlassCorrespondence = KlassCorrespondence(
        source_classification_id="131",
        target_classification_id="104",
        from_date=f"{year}-01-01",
        to_date=f"{year}-12-31",
    )

    komm_fylk_korr_df: pd.DataFrame = komm_fylk_korr_corr.data
    komm_fylk_korr_df = komm_fylk_korr_df[
        ~komm_fylk_korr_df["sourceCode"].isin(["9999"])
    ]
    komm_fylk_korr_df = komm_fylk_korr_df.rename(
        columns={
            "sourceCode": "from",
            "targetCode": "to",
        }
    )
    komm_fylk_korr_df = komm_fylk_korr_df[["from", "to"]]

    komm_fylk_korr_df["to"] = "EKA" + komm_fylk_korr_df["to"].str[:2]

    komm_kostra_gr_corr: KlassCorrespondence = KlassCorrespondence(
        source_classification_id="131",
        target_classification_id="112",
        from_date=f"{year}-01-01",
        to_date=f"{year}-12-31",
    )

    komm_kostra_gr_df: pd.DataFrame = komm_kostra_gr_corr.data
    komm_kostra_gr_df = komm_kostra_gr_df[
        ~komm_kostra_gr_df["sourceCode"].isin(["9999"])
    ]
    komm_kostra_gr_df = komm_kostra_gr_df.rename(
        columns={
            "sourceCode": "from",
            "targetCode": "to",
        }
    )
    komm_kostra_gr_df = komm_kostra_gr_df[["from", "to"]]

    nus: KlassClassification = KlassClassification(
        "131", language="nb", include_future=True
    )
    nuskoder: Any = nus.get_codes(f"{year}-01-01")
    ### Oppretter en korrespondansetabell for bydelene i Oslo som tilbyr tjenester
    klass_kommuner_landet: pd.DataFrame = nuskoder.pivot_level()
    klass_kommuner_landet = klass_kommuner_landet[
        ~klass_kommuner_landet["code_1"].isin(["9999"])
    ]
    klass_kommuner_landet = klass_kommuner_landet[["code_1"]].rename(
        columns={"code_1": "from"}
    )
    klass_kommuner_landet["to"] = "EAK"

    nus = KlassClassification("131", language="nb", include_future=True)
    nuskoder = nus.get_codes(f"{year}-01-01")
    ### Oppretter en korrespondansetabell for bydelene i Oslo som tilbyr tjenester
    klass_kommuner_u_oslo: pd.DataFrame = nuskoder.pivot_level()
    klass_kommuner_u_oslo = klass_kommuner_u_oslo[
        ~klass_kommuner_u_oslo["code_1"].isin(["0301", "9999"])
    ]
    klass_kommuner_u_oslo = klass_kommuner_u_oslo[["code_1"]].rename(
        columns={"code_1": "from"}
    )
    klass_kommuner_u_oslo["to"] = "EAKUO"

    mapping_kommuner: pd.DataFrame = pd.concat(
        [
            komm_fylk_korr_df,
            komm_kostra_gr_df,
            klass_kommuner_landet,
            klass_kommuner_u_oslo,
        ],
        ignore_index=True,
    )
    mapping_kommuner["from"] = mapping_kommuner["from"].astype(str).str.zfill(4)
    return mapping_kommuner


def mapping_fra_kommune_til_fylkeskommune(year: str | int) -> pd.DataFrame:
    """Mapping fra kommune til fylkeskommune.

    Denne funksjonen er ikke en funksjon du skal anvende direkte på et datasett.
    Her lages kun en mappingfil som viser hvordan kommunene inngår i de ulike fylkeskommunene et bestemt år.
    Dette er altså bare en hjelpefunksjon som inngår i annen funksjon, hierarkifunksjonen, som aggregerer opp kommunedata til de forskjellige regionsgrupperingene kun dersom inputfilen er en kommunefil.
    """
    komm_fylkeskommune_korr_corr: KlassCorrespondence = KlassCorrespondence(
        source_classification_id="131",
        target_classification_id="127",
        from_date=f"{year}-01-01",
        to_date=f"{year}-12-31",
    )

    komm_fylkeskommune_korr_df: pd.DataFrame = komm_fylkeskommune_korr_corr.data
    komm_fylkeskommune_korr_df = komm_fylkeskommune_korr_df[
        ~komm_fylkeskommune_korr_df["sourceCode"].isin(["9999"])
    ]
    komm_fylkeskommune_korr_df = komm_fylkeskommune_korr_df.rename(
        columns={
            "sourceCode": "from",
            "targetCode": "to",
        }
    )

    komm_fylkeskommune_korr_df = komm_fylkeskommune_korr_df[["from", "to"]]
    komm_fylkeskommune_korr_df["from"] = (
        komm_fylkeskommune_korr_df["from"].astype(str).str.zfill(4)
    )
    komm_fylkeskommune_korr_df["to"] = (
        komm_fylkeskommune_korr_df["to"].astype(str).str.zfill(4)
    )
    return komm_fylkeskommune_korr_df


def mapping_fra_fylkeskommune_til_kostraregion(year: str | int) -> pd.DataFrame:
    """Mapping fra fylkeskommune til KOSTRA-region (EAFK).

    Denne funksjonen er ikke en funksjon du skal anvende direkte på et datasett.
    Her lages kun en mappingfil som viser hvordan fylkeskommunene inngår i de ulike KOSTRA-fylkesgruppene et bestemt år.
    Dette er altså bare en hjelpefunksjon som inngår i annen funksjon, hierarkifunksjonen, som aggregerer opp fylkeskommunedata til de forskjellige regionsgrupperingene kun dersom inputfilen er       en fylkeskommunefil.
    """
    fylkeskomm_kostraregion_corr: KlassCorrespondence = KlassCorrespondence(
        source_classification_id="127",
        target_classification_id="152",
        from_date=f"{year}-01-01",
        to_date=f"{year}-12-31",
    )

    fylkeskomm_kostraregion_df: pd.DataFrame = fylkeskomm_kostraregion_corr.data

    fylkeskomm_kostraregion_df = fylkeskomm_kostraregion_df.rename(
        columns={
            "sourceCode": "from",
            "targetCode": "to",
        }
    )

    fylkeskomm_kostraregion_df = fylkeskomm_kostraregion_df[["from", "to"]]

    nus: KlassClassification = KlassClassification(
        "127", language="nb", include_future=True
    )
    nuskoder: Any = nus.get_codes(f"{year}-01-01")
    ### Oppretter en korrespondansetabell for bydelene i Oslo som tilbyr tjenester
    klass_fylkeskommuner_landet: pd.DataFrame = nuskoder.pivot_level()
    klass_fylkeskommuner_landet = klass_fylkeskommuner_landet[
        ~klass_fylkeskommuner_landet["code_1"].isin(["9900"])
    ]
    klass_fylkeskommuner_landet = klass_fylkeskommuner_landet[["code_1"]].rename(
        columns={"code_1": "from"}
    )
    klass_fylkeskommuner_landet["to"] = "EAFK"

    nus = KlassClassification("127", language="nb", include_future=True)
    nuskoder = nus.get_codes(f"{year}-01-01")
    ### Oppretter en korrespondansetabell for bydelene i Oslo som tilbyr tjenester
    klass_fylkeskommuner_landet_u_oslo: pd.DataFrame = nuskoder.pivot_level()
    klass_fylkeskommuner_landet_u_oslo = klass_fylkeskommuner_landet_u_oslo[
        ~klass_fylkeskommuner_landet_u_oslo["code_1"].isin(["0300", "9900"])
    ]
    klass_fylkeskommuner_landet_u_oslo = klass_fylkeskommuner_landet_u_oslo[
        ["code_1"]
    ].rename(columns={"code_1": "from"})
    klass_fylkeskommuner_landet_u_oslo["to"] = "EAFKUO"

    fylkeskomm_kostraregion_korr: pd.DataFrame = pd.concat(
        [
            fylkeskomm_kostraregion_df,
            klass_fylkeskommuner_landet,
            klass_fylkeskommuner_landet_u_oslo,
        ],
        ignore_index=True,
    )
    return fylkeskomm_kostraregion_korr


def hierarki(
    inputfil: pd.DataFrame,
    aggregeringstype: str | None = None,
    add_region_names: bool = False,
) -> pd.DataFrame:
    """Hierarkisk aggregering.

    Utfører hierarkisk regionsaggregering av inputfilen brukeren angir. Funksjonen fastslår
    regionsnivået basert på kolonnetittelen for regionsvariabelen ("kommuneregion",
    "fylkesregion" eller "bydelsregion").

    Regler:

    - «kommuneregion» aggregeres automatisk til «EKA», «EKG» og «EAK(UO)».
      Det anbefales normalt ikke å aggregere fra «kommuneregion» til «fylkesregion» (fylkeskommuner),
      men det er mulig ved å overstyre parameteren.

    - «fylkesregion» aggregeres automatisk til «EAFKXX» og «EAFK(UO)».

    - «bydelsregion» aggregeres automatisk til «EAB».

    Eksempler::

        # La funksjonen velge aggregeringstype automatisk
        df_agg = regionshierarki.hierarki(df)

        # Overstyring i kommunedata (ikke anbefalt, men mulig)
        df_agg = regionshierarki.hierarki(df, aggregeringstype="kommune_til_fylkeskommune")

    For at aggregeringen skal bli korrekt, må du angi klassifikasjonsvariabler i datasettet
    utover periode- og regionsvariabelen. Disse identifiseres automatisk hvis de er riktig navngitt.
    I Jupyter vil du få et tekstfelt der du kan skrive inn klassifikasjonsvariablene.

    Forhåndsdefinert input i notebook::

        from unittest.mock import patch
        INPUT_PATCH_TARGET = "builtins.input"
        predefined_input = "alder"
        with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
            df_aggregert = mapping_hierarki.hierarki(df_ikke_aggregert)
        display(df_aggregert)

    Merk:

    - Denne funksjonen aggregerer ikke regionsnavn. Unngå derfor datasett med egen kolonne
      for regionsnavn under aggregering. Fest eventuelle regionsnavn etterpå i en egen prosess.

    Parametre
    ---------
    inputfil : pandas.DataFrame
        Må inneholde «periode» og nøyaktig én av regionskolonnene:
        «kommuneregion» (4 sifre), «fylkesregion» (4 sifre, slutter på «00») eller «bydelsregion» (6 sifre).

    aggregeringstype : str | None
        Valgfritt. Dersom ``None``, bestemmes automatisk av regionkolonnen:

        - kommuneregion -> ``"kommune_til_landet"`` (kan overstyres til ``"kommune_til_fylkeskommune"``)
        - fylkesregion  -> ``"fylkeskommune_til_kostraregion"``
        - bydelsregion  -> ``"bydeler_til_EAB"``

    Returnerer
    ----------
    pandas.DataFrame
        Opprinnelige rader + aggregerte rader. Eventuell kolonnenavnendring (``kommuneregion`` -> ``fylkesregion``) anvendes.

    Kaster
    ------
    KeyError, ValueError
    """
    inputfil_copy = inputfil.copy()
    region_names_list = ["kommuneregion_navn", "bydelsregion_navn", "fylkesregion_navn"]
    if any(col in inputfil_copy.columns for col in region_names_list):
        logger.info(
            f"Datasettet ditt inneholder en kolonne for regionsnavn i tillegg til selve regionskodene. For at hierarkifunksjonen skal aggregere riktig, fjernes region_navn-kolonnene {region_names_list} fra datasettet."
        )
        inputfil_copy.drop(columns=region_names_list, inplace=True, errors="ignore")
    if inputfil_copy["periode"].nunique() > 1:
        raise KeyError("Mer enn 1 periode i datasettet")
    inputfil_copy["periode"] = inputfil_copy["periode"].astype(str)
    periode = inputfil_copy["periode"].unique()[0]
    print("Periode:")
    print(periode)

    region_col, inputfil_copy = _validate_and_normalize_region_col(inputfil_copy)

    mappingfil, join_col, replace_col, post_filter, rename_cols = _select_mapping(
        aggregeringstype, region_col, periode
    )
    df_merged = inputfil_copy.merge(
        mappingfil, left_on=join_col, right_on="from", how="inner"
    )
    klassifikasjonsvariable, statistikkvariable = (
        hjelpefunksjoner.definere_klassifikasjonsvariable(inputfil_copy)
    )
    df_merged[replace_col] = df_merged["to"]
    df_agg = df_merged.groupby(
        klassifikasjonsvariable,
        as_index=False,
        observed=True,
    )[
        statistikkvariable
    ].sum(min_count=1)
    df_combined = pd.concat([inputfil_copy, df_agg], ignore_index=True)
    return _postprocess_combined(
        df_combined, post_filter, rename_cols, klassifikasjonsvariable, add_region_names
    )


def overfore_data_fra_fk_til_k(inputfil: pd.DataFrame) -> pd.DataFrame:
    """Legge fylkeskommunedata over på alle tilhørende kommuner.

    Denne funksjonen legger data som kun finnes på fylkes- eller fylkeskommunenivå over på kommunenivå.
    Eksempel: Hvis forventet levealder for kvinner er 85.3 år i Vestland fylkeskommune (4600) i 2024,
    kan funksjonen legge 85.3 som forventet levealder for kvinner i alle kommuner i Vestland (46XX).

    Også her må du angi klassifikasjonsvariabler utover periode- og regionsvariabelen.

    Enkel bruk::

        df_kommune = mapping_hierarki.overfore_data_fra_fk_til_k(df_fylke)
        display(df_kommune)  # valgfritt

    Forhåndsdefinerte klassifikasjonsvariabler::

        from unittest.mock import patch
        INPUT_PATCH_TARGET = "builtins.input"
        predefined_input = "ekstra_klassifikasjonsv_1, ekstra_klassifikasjonsv_2"
        with patch("builtins.input", return_value=predefined_input):
            df_kommune = mapping_hierarki.overfore_data_fra_fk_til_k(df_fylke)
        display(df_kommune)
    """
    year: Any = inputfil["periode"].unique()[0]
    hjelpefunksjoner._konvertere_komma_til_punktdesimal(inputfil)
    hjelpefunksjoner.format_fil(inputfil)
    mappingfil: pd.DataFrame = mapping_fra_kommune_til_fylkeskommune(year)
    mappingfil[["from", "to"]] = mappingfil[["to", "from"]]
    mappingfil = mappingfil.copy()
    mappingfil["from"] = mappingfil["from"].astype(str).str.zfill(4)
    mappingfil["to"] = mappingfil["to"].astype(str).str.zfill(4)

    df_merged: pd.DataFrame = mappingfil.merge(
        inputfil, left_on="from", right_on="fylkesregion", how="left"
    )

    df_merged = df_merged.rename(columns={"to": "kommuneregion"})
    df_merged = df_merged.rename(columns={"to": "kommuneregion"}).drop(
        columns=["fylkesregion", "from"]
    )
    logger.info(
        "ℹ️ Funksjonen identifiserer nå 'kommuneregion' og ikke 'fylkesregion' som regionsvariabelen i dette datasettet. Dette er ikke en feil.\n"
        "Det skjer fordi datasettet gjøres om fra et fylkesregionsdatasett til et kommuneregionsdatasett."
    )

    klassifikasjonsvariable: list[str]
    statistikkvariable: list[str]
    klassifikasjonsvariable, statistikkvariable = (
        hjelpefunksjoner.definere_klassifikasjonsvariable(df_merged)
    )

    df_merged = df_merged[klassifikasjonsvariable + statistikkvariable]
    return df_merged


def _nullable_int_for(dtype: Any) -> Any:
    """Return a pandas nullable integer dtype matching the given dtype name.

    Falls back to Int64 if the specific bit width is not recognized.
    """
    name: str = str(dtype).lower()
    signed_map = {
        "int64": pd.Int64Dtype(),
        "int32": pd.Int32Dtype(),
        "int16": pd.Int16Dtype(),
        "int8": pd.Int8Dtype(),
    }
    unsigned_map = {
        "uint64": pd.UInt64Dtype(),
        "uint32": pd.UInt32Dtype(),
        "uint16": pd.UInt16Dtype(),
        "uint8": pd.UInt8Dtype(),
    }
    if name.startswith("int"):
        for key in signed_map:
            if key in name:
                return signed_map[key]
        return pd.Int64Dtype()
    if name.startswith("uint"):
        for key in unsigned_map:
            if key in name:
                return unsigned_map[key]
        return pd.UInt64Dtype()
    return pd.Int64Dtype()


def gjennomsnitt_aggregerte_regioner(
    df: pd.DataFrame,
    cols: list[str],
    denom_col: str = "teller",
    decimals: int | None = None,  # None => round to integer; e.g. 2 => round to 2 dp
    restore_original_dtype: bool = True,
    print_types: bool = True,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Aggregerer regioner og beregner gjennomsnitt.

    Funksjonen tar et datasett på kommune-, fylkeskommune- eller bydelsnivå og aggregerer det
    til regionsgrupperinger. Deretter beregnes gjennomsnitt for angitte kolonner, mens øvrige
    kolonner summeres. Merk at funksjonen ikke er egnet for andeler (f.eks. «andel_skilte»):
    en enkel snittberegning kan bli misvisende.

    Du må angi:

    1) Klassifikasjonsvariablene i datasettet (utenom periode- og regionsvariabelen). Periode og region
       blir alltid automatisk registrert som klassifikasjonsvariabler.

    2) Kolonnene det skal beregnes gjennomsnitt for.

    Eksempel uten forhåndsdefinerte klassifikasjonsvariabler::

        gjennomsnittskolonner = ["skilte_separerte"]
        df_gjennomsnitt = mapping_hierarki.gjennomsnitt_aggregerte_regioner(
            utvalgte_nokkeltall_kommuner_2024,
            cols=gjennomsnittskolonner,
            denom_col="teller",
            decimals=2,
            restore_original_dtype=False,
            print_types=True,
        )
        display(df_gjennomsnitt)

    Eksempel med forhåndsdefinerte klassifikasjonsvariabler::

        from unittest.mock import patch
        predefined_input = ""
        gjennomsnittskolonner = ["andel_skilte_separerte"]
        with patch("builtins.input", return_value=predefined_input):
            df_gjennomsnitt = mapping_hierarki.gjennomsnitt_aggregerte_regioner(
                utvalgte_nokkeltall_kommuner_2024,
                cols=gjennomsnittskolonner,
                denom_col="teller",
                decimals=2,
                restore_original_dtype=False,
                print_types=True,
            )
        display(df_gjennomsnitt)

    Args:
        df: Input-dataframe for aggregering og gjennomsnittsberegning.
        cols: Liste over kolonner som skal gjennomsnittsberegnes.
        denom_col: Kolonne som fungerer som nevner ved aggregering. Standard er "teller".
        decimals: Antall desimaler å runde til. ``None`` runder til nærmeste heltall.
        restore_original_dtype: Hvis ``True``, gjenopprettes opprinnelig dtype etter beregning.
        print_types: Hvis ``True``, skrives dtypene ut for debug.
        return_report: Hvis ``True``, returneres også en rapport over dtype-endringer.

    Returns:
        DataFrame, eventuelt sammen med en rapport over dtype-endringer.
    """
    df = df.copy()
    df["teller"] = 1
    df = hierarki(df)

    original: dict[str, Any] = cast(dict[str, Any], df[cols].dtypes.to_dict())
    post_op: dict[str, Any] = {}

    for c in cols:
        result = df[c] / df[denom_col]
        result = result.round(0) if decimals is None else result.round(decimals)
        post_op[c] = result.dtype
        if restore_original_dtype:
            df[c] = _restore_dtype(result, original[c])
        else:
            df[c] = result

    final: dict[str, Any] = cast(dict[str, Any], df[cols].dtypes.to_dict())

    if print_types:
        _print_dtype_report(original, post_op, final, cols)

    df = df.drop(columns=["teller"])
    if return_report:
        report: dict[str, dict[str, Any]] = {
            "original": original,
            "post_op": post_op,
            "final": final,
        }
        return df, report
    return df


def vektet_gjennomsnitt_aggregerte_regioner(
    inputfil: pd.DataFrame,
    klassifikasjonsvariable: list[str] | None = None,
    statistikkvariable: list[str] | None = None,
    vektede_variable: dict[str, str] | None = None,
    gjennomsnittsvariable: list[str] | None = None,
    aggregeringstype: str | None = None,
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

    add_region_names:
        Dersom ``True``, legges regionsnavn til gjennom eksisterende
        etterbehandling.
    
    return_report:
        Dersom ``True``, returneres også en rapport som dictionary.
        Rapporten inneholder informasjon om variabelbehandling,
        samt DataFrame-objektene ``utelatte_observasjoner`` og
        ``aggregerte_problemer``.

    Returnerer
    ----------
    pandas.DataFrame
        Opprinnelige rader og aggregerte rader.

    eller

    tuple[pandas.DataFrame, dict[str, Any]]
        DataFrame og rapport dersom ``return_report=True``.
        ``rapport["utelatte_observasjoner"]`` inneholder observasjoner
        som er utelatt fra beregning av vektede gjennomsnitt, mens
        ``rapport["aggregerte_problemer"]`` inneholder problemer som
        oppstår på aggregert nivå.
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
        raise KeyError(
            "Datasettet mangler obligatorisk kolonne 'periode'."
        )
    
    if df["periode"].isna().any():
        raise ValueError(
            "Kolonnen 'periode' inneholder manglende verdier."
        )
    
    antall_perioder = df["periode"].nunique()
    
    if antall_perioder == 0:
        raise ValueError(
            "Datasettet inneholder ingen observasjoner med gyldig periode."
        )
    
    if antall_perioder > 1:
        raise KeyError("Mer enn 1 periode i datasettet")
    
    df["periode"] = df["periode"].astype(str)
    periode = df["periode"].iloc[0]

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

    samme_variabel_og_vekt = sorted(
        target
        for target, weight in vektede_variable.items()
        if target == weight
    )
    
    if samme_variabel_og_vekt:
        raise ValueError(
            "En variabel kan ikke brukes som sin egen vekt. "
            "Følgende variable er oppgitt både som variabel og vekt: "
            f"{samme_variabel_og_vekt}"
        )

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
                "aggregerte regioner fordi summen av gyldige "
                f"vekter er 0 eller mangler: {problem_regioner}"
            )

            for idx in df_agg.index[ugyldig_nevner]:
                aggregerte_problemer.append(
                    {
                        "variabel": target_col,
                        "region": df_agg.at[idx, region_col],
                        "grunn": "sum av gyldige vekter er 0 eller mangler",
                    }
                )

    # ---------------------------------------------------------
    # Beregn vanlige gjennomsnitt
    # ---------------------------------------------------------
    for target_col, count_col in mean_temp.items():
        result = df_agg[target_col] / df_agg[count_col]
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

            for idx in df_agg.index[ingen_observasjoner]:
                aggregerte_problemer.append(
                    {
                        "variabel": target_col,
                        "region": df_agg.at[idx, region_col],
                        "grunn": "alle observasjoner mangler",
                    }
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
    # Samme etterbehandling som hierarki()
    # ---------------------------------------------------------
    df_combined = _postprocess_combined(
        df_combined,
        post_filter,
        rename_cols,
        alle_klassifikasjonsvariable,
        add_region_names,
    )

    excluded_df = pd.DataFrame(
        excluded_records,
        columns=[
            "variabel",
            "vektvariabel",
            "region",
            "grunn",
            "verdi",
            "vekt",
            *rapport_klassifikasjonsvariable,
        ],
    )

    aggregerte_problemer_df = pd.DataFrame(
        aggregerte_problemer,
        columns=[
            "variabel",
            "region",
            "grunn",
        ],
    )


    if excluded_df.empty and aggregerte_problemer_df.empty:
        print(
            "\n✅ Ingen utelatte observasjoner eller "
            "aggregerte problemer å rapportere."
        )

    if not excluded_df.empty:
        print("\n⚠️ Utelatte observasjoner:")
        display(excluded_df)

    if not aggregerte_problemer_df.empty:
        print("\n⚠️ Aggregerte problemer:")
        display(aggregerte_problemer_df)
    
    if return_report:
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
        }
    
        return df_combined, report
    
    return df_combined
