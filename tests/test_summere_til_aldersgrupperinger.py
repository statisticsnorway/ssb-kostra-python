import pandas as pd

from ssb_kostra_python.summere_til_aldersgrupperinger import (
    summere_til_aldersgrupperinger,
)


class TestSummereTilAldersgrupperinger:
    """Tests for `summere_til_aldersgrupperinger(df_input, hierarki_path=...)`."""

    from typing import Any

    def test_summere_til_aldersgrupperinger_basic(self, mocker: Any) -> None:
        """Validate a basic end-to-end flow of summing/aggregating rows into age groups."""
        # 1) Arrange: input data
        df_input = pd.DataFrame(
            {
                "periode": ["2025", "2025", "2025"],
                "alder": ["1", "2", "10"],
                "kommuneregion": ["0301", "0301", "0301"],
                "personer": [10, 20, 5],
            }
        )

        # 2) Arrange: hierarchy data
        df_hierarki = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "from": [1, 2],
                "to": ["000-004", "000-004"],
            }
        )
        mock_read_parquet = mocker.patch(
            "pandas.read_parquet", return_value=df_hierarki
        )

        # 3) Arrange: patch format_fil
        def fake_format_fil(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["periode"] = df["periode"].astype(str)
            df["alder"] = df["alder"].astype(str).str.zfill(3)
            return df

        mock_format_fil = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.format_fil", side_effect=fake_format_fil
        )

        # 4) Arrange: patch definere_klassifikasjonsvariable
        klass_vars = ["periode", "kommuneregion", "to"]
        stat_vars = ["personer"]
        mock_definer_klass = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable",
            return_value=(klass_vars, stat_vars),
        )

        mocker.patch("ssb_kostra_python.summere_til_aldersgrupperinger.display")

        # 5) Act: run function
        rename_variabel, groupby_variable, df_out = summere_til_aldersgrupperinger(
            df_input,
            hierarki_path="dummy/path.parquet",
        )

        # 6) Assert: metadata
        assert rename_variabel == ["alder"]
        assert sorted(groupby_variable) == sorted(["periode", "kommuneregion", "to"])

        # 7) Assert: output aggregated row
        assert (df_out["alder"] == "000-004").any()
        aggregated = df_out[df_out["alder"] == "000-004"]
        assert aggregated["personer"].iloc[0] == 30

        # 8) Assert: calls happened
        mock_read_parquet.assert_called_once()
        mock_format_fil.assert_called_once()
        mock_definer_klass.assert_called_once()
