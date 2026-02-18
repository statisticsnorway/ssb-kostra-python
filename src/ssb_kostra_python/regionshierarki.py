from collections.abc import Callable
from typing import Any

import pandas as pd


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
) -> pd.DataFrame:
    df[klassifikasjonsvariable] = df[klassifikasjonsvariable].astype(str)
    if post_filter:
        df = post_filter(df)
    if rename_cols:
        df = df.rename(columns=rename_cols)
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


# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: kostra-fellesfunksjoner
#     language: python
#     name: kostra-fellesfunksjoner
# ---

# %%
import logging
from typing import Any
from typing import cast

import pandas as pd
from klass import KlassClassification
from klass import KlassCorrespondence
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_float_dtype
from pandas.api.types import is_integer_dtype

from ssb_kostra_python import hjelpefunksjoner

logger = logging.getLogger(__name__)
# %% [markdown]
# ### Innhenting av filer til bruk


# %%
# def mapping_bydeler_oslo(year: str | int = "2015"):
def mapping_bydeler_oslo(year: str | int = "2015") -> pd.DataFrame:
    """Mapping av bydelene i Oslo.

    Denne funksjonen er ikke en funksjon du skal anvende direkte på et datasett.
    Her lages kun en mappingfil som viser hvordan Oslos bydeler inngår i samlebydelen "EAB".
    Dette er altså bare en hjelpefunksjon som inngår i en annen funksjon, hierarkifunksjonen, som aggregerer opp bydelsdata til "EAB" kun dersom inputfilen er en bydelsfil.
    """
    nus = KlassClassification(241, language="nb", include_future=True)
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


# %%
# def hierarki_fra_kommune_til_landet(year : str | int):
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
    # display(komm_fylk_korr)

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
    # display(komm_kostra_gr)

    nus: KlassClassification = KlassClassification(
        131, language="nb", include_future=True
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
    # display(klass_kommuner_landet)

    nus = KlassClassification(131, language="nb", include_future=True)
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
    # display(klass_kommuner_u_oslo)

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


# %%
# def hierarki_fra_kommune_til_fylkeskommune(year : str | int):
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


# %%
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
        127, language="nb", include_future=True
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

    nus = KlassClassification(127, language="nb", include_future=True)
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


# %%
# def hierarki_mapping(inputfil: pd.DataFrame, aggregeringstype: str | None = None) -> pd.DataFrame:
def hierarki(
    inputfil: pd.DataFrame, aggregeringstype: str | None = None
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
        df_agg = mapping_hierarki.hierarki(df)

        # Overstyring i kommunedata (ikke anbefalt, men mulig)
        df_agg = mapping_hierarki.hierarki(df, aggregeringstype="kommune_til_fylkeskommune")

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
    df_agg = df_merged.groupby(klassifikasjonsvariable, as_index=False, observed=True)[
        statistikkvariable
    ].sum()
    df_combined = pd.concat([inputfil_copy, df_agg], ignore_index=True)
    return _postprocess_combined(
        df_combined, post_filter, rename_cols, klassifikasjonsvariable
    )


# %%
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
    hjelpefunksjoner.konvertere_komma_til_punktdesimal(inputfil)
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


# %%
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
