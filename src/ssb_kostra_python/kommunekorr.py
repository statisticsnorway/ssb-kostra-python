import pandas as pd
from klass import KlassClassification
from klass import KlassCorrespondence
from requests.exceptions import HTTPError


def kostra_kommunekorr(year: str) -> pd.DataFrame:
    """Fetches and compiles data on correspondences between municipalities and related classifications for a given year.

    The function retrieves the following:
      - Municipality classification (KLASS 131) and manually adds Longyearbyen.
      - The correspondence between municipality (KLASS 131) and KOSTRA group (KLASS 112). This request is wrapped in a try-except block to catch HTTP 404 errors and raise a descriptive ValueError.
      - The correspondence between municipality (KLASS 131) and county (KLASS 104).

    The retrieved data is merged into a single DataFrame containing information on:
      - Municipality number (komm_nr) and name (komm_navn)
      - County number (fylke_nr) and name (fylke_navn)
      - KOSTRA group number (kostra_gr) and name (kostra_gr_navn)
      - Validity start and end dates for both KOSTRA group and county classifications.
      - Additional columns:
          - 'fylke_nr_eka': county number prefixed with "EKA".
          - 'fylke_nr_eka_m_tekst': concatenation of 'fylke_nr_eka' and the county name.
          - 'landet': a static label "EAK Landet".
          - 'landet_u_oslo': a static label "EAKUO Landet uten Oslo" (set to NaN for Oslo, municipality code "0301").

    Args:
        year (str): The year (format "YYYY") for which data should be fetched.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - komm_nr: Municipality number.
            - komm_navn: Municipality name.
            - fylke_nr: County number.
            - fylke_navn: County name.
            - fylke_nr_eka: County number prefixed with "EKA".
            - fylke_nr_eka_m_tekst: Combination of fylke_nr_eka and fylke_navn.
            - fylke_validFrom: Start date for county classification validity.
            - fylke_validTo: End date for county classification validity.
            - kostra_gr: KOSTRA group number.
            - kostra_gr_navn: KOSTRA group name.
            - kostra_validFrom: Start date for KOSTRA group validity.
            - kostra_validTo: End date for KOSTRA group validity.
            - landet: Static label for the nation.
            - landet_u_oslo: Static label for the nation excluding Oslo.

    Raises:
        ValueError: If the correspondence between municipality and KOSTRA group is not found (e.g., HTTP 404),
                    or if duplicates are detected for municipality numbers after merging the data.

    Example:
        >>> df = kostra_kommunekorr("2025")
        >>> df['verdi'] = 1000
        >>> groups = [
        ...     ['komm_nr', 'komm_navn'],
        ...     ['fylke_nr', 'fylke_navn'],
        ...     ['kostra_gr', 'kostra_gr_navn'],
        ...     ['landet_u_oslo'],
        ...     ['landet']
        ... ]
        >>> agg_list = []
        >>> for cols in groups:
        ...     temp = df.groupby(cols)['verdi'].sum().rename('agg_verdi')
        ...     agg_list.append(temp)
        >>> df_agg = pd.DataFrame(pd.concat(agg_list))
    """
    from_date = f"{year}-01-01"
    to_date = f"{year}-12-31"

    kom = (
        KlassClassification("131", language="nb", include_future=False)
        .get_codes(from_date=from_date, to_date=to_date)
        .data[["code", "name"]]
        .rename(columns={"code": "komm_nr", "name": "komm_navn"})
    )

    # Manually add Longyearbyen
    df_longyear = pd.DataFrame({"komm_nr": ["2111"], "komm_navn": ["Longyearbyen"]})
    kom = pd.concat([kom, df_longyear], ignore_index=True)

    # Retrieve the correspondence between municipality and KOSTRA group (KLASS 112)
    try:
        korresp_kostra = KlassCorrespondence(
            source_classification_id="131",
            target_classification_id="112",
            from_date=from_date,
            to_date=to_date,
        )
        kom_kostra_gr = korresp_kostra.data.rename(
            columns={
                "sourceCode": "komm_nr",
                "targetCode": "kostra_gr",
                "targetName": "kostra_gr_navn",
                "validFrom": "kostra_validFrom",
                "validTo": "kostra_validTo",
            }
        ).drop(columns=["sourceName", "sourceShortName", "targetShortName"])
    except HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(
                f"KOSTRA group correspondence (131 → 112) for the period {from_date} to {to_date} was not found."
            ) from e
        else:
            raise

    # Retrieve the correspondence between municipality and county (KLASS 104)
    korresp_fyl = KlassCorrespondence(
        source_classification_id="131",
        target_classification_id="104",
        from_date=from_date,
        to_date=to_date,
    )
    kom_fyl = korresp_fyl.data.rename(
        columns={
            "sourceCode": "komm_nr",
            "targetCode": "fylke_nr",
            "targetName": "fylke_navn",
            "validFrom": "fylke_validFrom",
            "validTo": "fylke_validTo",
        }
    ).drop(columns=["sourceName", "sourceShortName", "targetShortName"])

    # Merge the data
    kom = pd.merge(kom, kom_kostra_gr, on="komm_nr", how="left", validate="1:m")
    kom = pd.merge(kom, kom_fyl, on="komm_nr", how="left", validate="1:m")

    # Check for duplicate municipality numbers and raise an error if found
    if kom.duplicated("komm_nr").sum() > 0:
        duplicates = list(kom[kom.duplicated("komm_nr")]["komm_navn"])
        raise ValueError(
            "Duplicates detected for municipality numbers: " + ", ".join(duplicates)
        )

    kom = kom[kom["komm_nr"] != "9999"].copy()

    # Add extra columns for county data and national categorization
    kom["fylke_nr_eka"] = "EKA" + kom["fylke_nr"].str[:2]
    kom["fylke_nr_eka_m_tekst"] = kom["fylke_nr_eka"] + " " + kom["fylke_navn"]
    kom["landet"] = "EAK Landet"
    kom["landet_u_oslo"] = "EAKUO Landet uten Oslo"
    kom.loc[kom["komm_nr"] == "0301", "landet_u_oslo"] = pd.NA

    return kom[
        [
            "komm_nr",
            "komm_navn",
            "fylke_nr",
            "fylke_navn",
            "fylke_nr_eka",
            "fylke_nr_eka_m_tekst",
            "fylke_validFrom",
            "fylke_validTo",
            "kostra_gr",
            "kostra_gr_navn",
            "kostra_validFrom",
            "kostra_validTo",
            "landet",
            "landet_u_oslo",
        ]
    ]
