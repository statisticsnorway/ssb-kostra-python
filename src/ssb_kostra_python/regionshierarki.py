def _select_mapping(aggregeringstype, region_col, periode):
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

        def _post_filter_kommuner_til_fylke(df):
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


def _validate_and_normalize_region_col(df):
    region_cols = [
        c for c in ["kommuneregion", "fylkesregion", "bydelsregion"] if c in df.columns
    ]
    if len(region_cols) != 1:
            if len(region_cols) == 0:
                raise ValueError("Fant ingen gyldig regionkolonne ('kommuneregion', 'fylkesregion', 'bydelsregion'). Datasettet ditt må inneholde minst én.")
            else:
                raise ValueError(f"Fant flere regionskolonner {region_cols}. Det skal være nøyaktig én.")
    col = region_cols[0]
    if col == "kommuneregion":
        df[col] = df[col].astype(str).str.zfill(4)
    elif col == "fylkesregion":
        df[col] = df[col].astype(str).str.zfill(4)
    else:
        df[col] = df[col].astype(str).str.zfill(6)
    return col, df


def _postprocess_combined(df, post_filter, rename_cols, klassifikasjonsvariable):
    df[klassifikasjonsvariable] = df[klassifikasjonsvariable].astype(str)
    if post_filter:
        df = post_filter(df)
    if rename_cols:
        df = df.rename(columns=rename_cols)
    return df.reset_index(drop=True)


def _print_dtype_report(original, post_op, final, cols):
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


def _restore_dtype(result, orig):
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

    Utfører hierarkisk regionsaggregering av inputfilen brukeren angir. Det er forsøkt å gjøre funksjonen så lik hierarkifunksjonen i KOMPIS som mulig. Denne funksjonen dessverre krever litt
    ekstra arbeid, men ikke mye.

    Funksjonen fastslår regionsnivået i datasettet basert på kolonnetittelen for regionsvariabelen (som MÅ være "kommuneregion", "fylkesregion" eller "bydelsregion").
    - Dersom regionsnivået er "kommuneregion" (regionsvariabelen må hete "kommuneregion"), vil funksjonen automatisk aggregere opp til "EKA", "EKG" og "EAK(UO)".
        - Det anbefales ikke å aggregere opp fra "kommuneregion" til "fylkesregion", som angir fylkeskommuner (f.eks: 0300), og ikke fylker (f.eks: EKA03), men det går an.
          Da må du angi det i funksjonen.
    - Dersom regionsnivået er "fylkesregion" (regionsvariabelen må hete "fylkesregion"), vil funksjonen automatisk aggregere opp til "EAFKXX" og "EAFK(UO)".
    - Dersom regionsnivået er "bydelsregion" (regionsvariabelen må hete "bydelsregion"), vil funksjonen automatisk aggregere opp til "EAB".

    Slik skriver du funksjonen:
        - Vanligst: la funksjonen velge aggregeringstype automatisk:
            df_agg = mapping_hierarki.hierarki(df)

        - Sjelden override i kommunedata, som kommentert over er dette ikke anbefalt, men er nødvendig i noen tilfeller (kommune -> fylkeskommune):
            df_agg = mapping_hierarki.hierarki(df, aggregeringstype="kommune_til_fylkeskommune")

    For at den hierarkiske aggregeringen skal skje på riktig måte, er det nødvendig å angi de klassifikasjonsvariablene som finnes i datasettet UTOVER periode- og regionsvariabelen. Som kommentert
    over, blir periode- og regionsvariabelen automatisk identifisert, så lenge de er navngitt riktig.
    Når du kjører funksjonen vil det automatisk dukke opp et tekstfelt der du må skrive inn disse ektra klassifikasjonsvariablene. Det er muligens irriterende å måtte føre inn de ekstra
    klassifikasjonsvariablene hver gang du kjører funksjonen i en ferdiglaget og mer eller mindre fast notebook som skal behandle de samme datasettene gang etter gang, særlig hvis de ekstra
    klassifikasjonsvariablene er de samme hver gang. Da kan du gjøre slik:

    from unittest.mock import patch
    INPUT_PATCH_TARGET = "builtins.input"

    predefined_input = "alder"

    with patch(INPUT_PATCH_TARGET, return_value=predefined_input):
        df_aggregert = mapping_hierarki.hierarki(df_ikke_aggregert)

    display(df_aggregert)

    I eksempelet over er "alder" den ene klassifikasjonsvariabelen som finnes i datasettet i tillegg til periode- og regionsvariabelen. Du forhåndsdefinerer inputen 'alder' og kjører funksjonen.
    Funksjonen vil bruke den forhåndsdefinerte inputen til tekstpromtet, og du slipper å bli spurt.


    Denne funksjonen fungerer IKKE til å aggregere regionsnavn til aggregerte regionsnavn. Du unngår rot i datasettet ditt hvis du aggregerer et datasett uten en kolonne for regionsnavnet.
    Prøver du å aggregere et datasett som også inneholder regionsnavnene vil det likevel skje en aggregering, men regionsnavnene blir ikke aggregert riktig.
    Det finnes en annen funksjon du kan bruke til å feste regionsnavn etter at hierarkioperasjonen er utført.
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Parametre
    ---------
    inputfil : pandas.DataFrame
        Må inneholde 'periode' og ikke flere enn én av regionskolonnene:
        'kommuneregion' (4 sifre), 'fylkesregion' (4 sifre, slutter på '00') eller 'bydelsregion' (6 sifre).

    aggregeringstype : str | None
        Valgfritt. Dersom None, bestemmes automatisk av hvilken regionkolonne som finnes:
            - kommuneregion -> "kommune_til_landet" (default; kan overstyres til "kommune_til_fylkeskommune")
            - fylkesregion  -> "fylkeskommune_til_kostraregion"
            - bydelsregion  -> "bydeler_til_EAB"

    Returnerer
    ---------
    pandas.DataFrame
        Opprinnelige rader + aggregerte rader. Ev. kolonnenavnendring (kommuneregion -> fylkesregion) anvendes.

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

    Denne funksjonen brukes til å legge data som bare finnes på fylkes- eller fylkeskommunenivå over på enkeltkommunenivå. Dersom du har et fylkes(-kommune)datasett, kan dette
    gjøres om til et kommunedatasett der verdiene for fylket/fylkeskommunen gjør seg gjeldende for hver kommune i fylket/fylkeskommunen.

    Det vil si at dersom forventet levealder for kvinner er
    85.3 år i Vestland fylkeskommune (4600) i 2024, kan du bruke denne funksjonen til å legge 85.3 år som forventet levealder for kvinner i alle 46XX-kommuner som inngår i
    Vestland fylke.

    Også i denne funksjonen er det nødvendig å føre inn klassifikasjonsvariable utover periode- og regionsvariabelen.
    Du kan skrive inn funksjon så enkelt som under:

    df_kommune = mapping_hierarki.overfore_data_fra_fk_til_k(df_fylke)
    display(df_kommune) <------ om du trenger å se datasettet etterpå

    I tilfellet over har du ikke forhåndsdefinert de øvrige klassifikasjonsvariablene i datasettet, og det vil dukke opp et tekstfelt der du må føre den/dem inn.


    Du kan også forhåndsdefinere de(n) øvrige klassifikasjonsvariablene. Da skriver du slik:

    from unittest.mock import patch
    INPUT_PATCH_TARGET = "builtins.input"

    predefined_input = "ekstra_klassifikasjonsv_1, ekstra_klassifikasjonsv_2 "
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
    """Performs region aggregation and computes region averages.

    Denne funksjonen tar et datasett på kommune-, fylkeskommune- eller bydelsnivå, og aggregerer det til regionsgrupperinger.
    Den beregner så gjennomnittsverdier for de kolonnene brukeren velger ut. De øvrige kolonnene blir summert. Funksjonen kan brukes til å
    beregne gjennomsnittet av summerbare variable (f.eks. folkemengde, antall_menn, antall_barn), men den passer IKKE til å beregne
    gjennomsnittet av andeler (f.eks. andel_skilte). Om andelen er 0.51 i en stor kommune og 0.49 i en liten kommune, vil denne
    funksjonen beregne gjennomsnittet til 0.50, og det blir ikke riktig. Funksjonen duger altså kun til å beregne gjennomsnittet av summerbare størrelser.

    For at funksjonen skal fungere, må du angi
    1) klassifikasjonsvariablene i datasettet utenom periode- og regionsvariabelen. Periode og region vil alltid bli automatisk registrert som klassifikasjonsvariable.
    og
    2) kolonnene det skal utføres gjennomsnittsberegninger på.

    * Slik bruker du funksjonen dersom du ikke har forhåndsdefinert de øvrige klassifikasjonsvariablene. Du vil bli bedt om å angi disse i tekstfeltet som dukker opp underveis.

    gjennomsnittskolonner = ['skilte_separerte']

    df_gjennomsnitt = mapping_hierarki.gjennomsnitt_aggregerte_regioner(
    utvalgte_nokkeltall_kommuner_2024,
    cols=gjennomsnittskolonner,
    denom_col="teller",
    decimals=2,
    restore_original_dtype=False,
    print_types=True
    )

    display(df_gjennomsnitt)

    * Du KAN OGSÅ FORHÅNDSDEFINERE de øvrige klassifikasjonsvariablene for å slippe tekstpromptet. Da gjør du som under. Du lager et objekt som du for eksempel kan kalle predefined_input
    og i denne skriver du inn variablene. Det skal være komma mellom hver av dem og anførselstegn på utsiden. Du kan også lage en predefined input uten noen ting i, som vist i
    eksempelet. Da gir du beskjed om at det ikke finnes andre klassifikasjonsvariable utenom 'periode' og 'region' og også da slipper du også tekstpromptet.
    Skriv slik:

    predefined_input = ""
    gjennomsnittskolonner = ['andel_skilte_separerte']

    with patch("builtins.input", return_value=predefined_input):

        df_gjennomsnitt = mapping_hierarki.gjennomsnitt_aggregerte_regioner( <------ df_gjennomsnitt er datasettet som blir spyttet ut.
        utvalgte_nokkeltall_kommuner_2024, <------ datasettet som skal behandles.
        cols=gjennomsnittskolonner, <------ kolonnene det skal beregnes gjennomsnitt på.
        denom_col="teller",
        decimals=2, <------ antall desimaler du ønsker på det beregnede gjennomsnittet.
        restore_original_dtype=False, <------ skal formatet på de utregnede verdiene være det samme som formatet på verdien gjennomsnittet blir beregnet av?
        print_types=True <------ skal du ha utskrift av formatet på variablene i datasettet etter at beregningen er utført?
        )

    display(df_gjennomsnitt)

    Args:
        df: Input DataFrame on which regional aggregation and average computation will be applied.
        cols: List of columns to perform aggregation and calculations on.
        denom_col: Column serving as the denominator for aggregation. Defaults to "teller".
        decimals: Number of decimal points to round to. None rounds to the nearest integer.
        restore_original_dtype: If True, restores the original dtype of the columns after computation.
        print_types: If True, prints the dtypes at different stages for debugging purposes.
        return_report: If True, returns a tuple containing the DataFrame and a report of dtype changes.

    Returns:
        Modified DataFrame, optionally along with a report of dtype changes.
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
