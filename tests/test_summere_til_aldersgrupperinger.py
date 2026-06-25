from typing import Any

import pandas as pd

from ssb_kostra_python.summere_til_aldersgrupperinger import (
    summere_til_aldersgrupperinger,
)


class TestSummereTilAldersgrupperinger:
    def test_summere_til_aldersgrupperinger_basic(self, mocker: Any) -> None:
        df_input = pd.DataFrame(
            {
                "periode": ["2025", "2025", "2025"],
                "alder": ["1", "2", "10"],
                "kommuneregion": ["0301", "0301", "0301"],
                "personer": [10, 20, 5],
            }
        )

        df_hierarki = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "from": [1, 2],
                "to": ["000-004", "000-004"],
            }
        )

        mock_read_parquet = mocker.patch(
            "ssb_kostra_python.summere_til_aldersgrupperinger.pd.read_parquet",
            return_value=df_hierarki,
        )

        def fake_format_fil(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["periode"] = df["periode"].astype(str).str.zfill(4)

            if "alder" in df.columns:
                df["alder"] = df["alder"].astype(str).str.zfill(3)

            if "from" in df.columns:
                df["from"] = df["from"].astype(str).str.zfill(3)

            if "kommuneregion" in df.columns:
                df["kommuneregion"] = df["kommuneregion"].astype(str).str.zfill(4)

            return df

        mock_format_fil = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.format_fil",
            side_effect=fake_format_fil,
        )

        mock_definer_klass = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable",
            return_value=(
                ["periode", "kommuneregion", "to"],
                ["personer"],
            ),
        )

        mock_display = mocker.patch(
            "ssb_kostra_python.summere_til_aldersgrupperinger.display"
        )

        df_out = summere_til_aldersgrupperinger(
            df_input,
            hierarki_path="dummy/path.parquet",
        )

        mock_read_parquet.assert_called_once_with("dummy/path.parquet")
        assert mock_format_fil.call_count == 2
        mock_definer_klass.assert_called_once()
        mock_display.assert_called_once()

        assert isinstance(df_out, pd.DataFrame)

        assert "000-004" in df_out["alder"].tolist()

        aggregated = df_out[df_out["alder"] == "000-004"]

        assert len(aggregated) == 1
        assert aggregated["periode"].iloc[0] == "2025"
        assert aggregated["kommuneregion"].iloc[0] == "0301"
        assert aggregated["personer"].iloc[0] == 30

        original_ages = set(
            df_out[df_out["alder"].isin(["001", "002", "010"])]["alder"]
        )
        assert original_ages == {"001", "002", "010"}

    def test_adds_to_groupby_when_user_does_not_classify_to(self, mocker: Any) -> None:
        df_input = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "alder": ["1", "2"],
                "kommuneregion": ["0301", "0301"],
                "personer": [10, 20],
            }
        )

        df_hierarki = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "from": [1, 2],
                "to": ["000-004", "000-004"],
            }
        )

        mocker.patch(
            "ssb_kostra_python.summere_til_aldersgrupperinger.pd.read_parquet",
            return_value=df_hierarki,
        )

        def fake_format_fil(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["periode"] = df["periode"].astype(str).str.zfill(4)

            if "alder" in df.columns:
                df["alder"] = df["alder"].astype(str).str.zfill(3)

            if "from" in df.columns:
                df["from"] = df["from"].astype(str).str.zfill(3)

            if "kommuneregion" in df.columns:
                df["kommuneregion"] = df["kommuneregion"].astype(str).str.zfill(4)

            return df

        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.format_fil",
            side_effect=fake_format_fil,
        )

        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable",
            return_value=(
                ["periode", "kommuneregion"],
                ["to", "personer"],
            ),
        )

        mocker.patch("ssb_kostra_python.summere_til_aldersgrupperinger.display")

        df_out = summere_til_aldersgrupperinger(
            df_input,
            hierarki_path="dummy/path.parquet",
        )

        aggregated = df_out[df_out["alder"] == "000-004"]

        assert len(aggregated) == 1
        assert aggregated["personer"].iloc[0] == 30
