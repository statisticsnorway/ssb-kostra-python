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

import pandas as pd

INPUT_PATCH_TARGET = "builtins.input"


logger = logging.getLogger(__name__)


# %%
def format_fil(file: pd.DataFrame) -> pd.DataFrame:
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
    """
    # --- simple fixed-width fields ---
    if "periode" in file.columns:
        file["periode"] = file["periode"].astype("string").str.zfill(4)

    if "alder" in file.columns:
        file["alder"] = file["alder"].astype("string").str.zfill(3)

    # --- conditional padding helper (only digits & too short), dtype-safe ---
    def _conditional_pad(col: str, width: int):
        if col not in file.columns:
            return
        # Ensure the actual column (not just a temp Series) is string dtype
        file[col] = file[col].astype("string")

        # Mask: digits-only AND length < width
        mask = file[col].str.fullmatch(r"\d+") & (file[col].str.len() < width)

        # Assign using where(...) to avoid dtype-mismatch warnings/errors
        file[col] = file[col].where(~mask, other=file[col].str.zfill(width))

    # Apply to possible region columns (pad only where appropriate)
    _conditional_pad("kommuneregion", 4)
    _conditional_pad("fylkesregion", 4)
    _conditional_pad("bydelsregion", 6)  # 6 for bydelsregion

    # If none of the region columns are present, raise
    if not any(
        c in file.columns for c in ("kommuneregion", "fylkesregion", "bydelsregion")
    ):
        raise ValueError(
            "No valid region column ('kommuneregion', 'fylkesregion', or 'bydelsregion') found."
        )

    return file


# %%
# def definere_klassifikasjonsvariable(inputfil: pd.DataFrame):


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
    print("Alle variable i datasettet:")
    print(tot_cols)

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
    andre_klassifikasjonsvariable = input(
        f"Datasettet inneholder kostra-klassifikasjonsvariablene felles for alle datasett i kostra {felles_klassifikasjonsvariable}.\n"
        "Skriv inn andre klassifikasjonsvariable UTENOM DE OBLIGATORISKE (da trenger du ikke å skrive inn disse: 'periode', 'kommuneregion', 'fylkesregion' eller 'bydelsregion') \n"
        "som datasettet inneholder, uten anførselstegn og komma mellom hver dersom flere enn 1:\n"
        "Trykk ganske enkelt 'enter' dersom du ikke har flere klassifikasjonsvariable å legge til."
    )

    andre_klassifikasjonsvariable = [
        var.strip() for var in andre_klassifikasjonsvariable.split(",") if var.strip()
    ]
    if len(andre_klassifikasjonsvariable) == 0:
        print("Ingen andre klassifikasjonsvariable valgt.")
    else:
        print("Andre klassifikasjonsvariable:")
        print(andre_klassifikasjonsvariable)

    # Helper to deduplicate while preserving order
    # def uniq(seq):
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

    print("Klassifikasjonsvariable i datasettet:")
    print(klassifikasjonsvariable)

    inputfil[klassifikasjonsvariable] = inputfil[klassifikasjonsvariable].astype(
        "string"
    )

    print("Statistikkvariable i datasettet:")
    print(statistikkvariable)

    print("Oppdaterte datatyper:")
    print(inputfil.dtypes)

    return klassifikasjonsvariable, statistikkvariable


# %%
def konvertere_komma_til_punktdesimal(inputfil: pd.DataFrame) -> pd.DataFrame:
    """Konvertere komma til punktdesimal i datasettet."""
    df = inputfil.copy()
    cols_with_commas = [
        col for col in df.columns if df[col].astype(str).str.contains(",").any()
    ]
    for col in cols_with_commas:
        df[col] = df[col].str.replace(",", ".", regex=False).astype(float)
    return df
