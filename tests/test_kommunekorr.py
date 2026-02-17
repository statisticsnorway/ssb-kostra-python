import pandas as pd

from ssb_kostra_python.kommunekorr import kostra_kommunekorr


from typing import Any

def test_kostra_kommunekorr(
    mock_klass_classification: Any, mock_klass_correspondence: Any
) -> None:
    year = "2023"
    result = kostra_kommunekorr(year)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 7

    exp_cols = [
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
    assert result.shape[1] == len(exp_cols)

    for c1, c2 in zip(exp_cols, result.columns, strict=False):
        assert c1 == c2
