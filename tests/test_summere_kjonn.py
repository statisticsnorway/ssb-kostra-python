import pandas as pd

from ssb_kostra_python.summere_kjonn import summere_over_kjonn


class TestSummereOverKjonn:
    """Tests for `summere_over_kjonn(df)`."""

    def test_returns_input_unchanged_when_kjonn_missing(self) -> None:
        """Verify behavior when 'kjonn' is missing."""
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "kommuneregion": ["0301", "0301"],
                "personer": [10, 20],
            }
        )

        out = summere_over_kjonn(df)

        assert out.equals(df)

    from typing import Any

    def test_sums_over_kjonn_when_present(self, mocker: Any) -> None:
        """Verify that when 'kjonn' exists, the function sums all statistikkvariable over kjonn."""
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025", "2025"],
                "kommuneregion": ["0301", "0301", "0301"],
                "kjonn": ["1", "2", "1"],
                "personer": [10, 20, 5],
            }
        )

        mock_definer_klass = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable"
        )
        mock_definer_klass.return_value = (
            ["periode", "kommuneregion", "kjonn"],
            ["personer"],
        )

        out = summere_over_kjonn(df)

        assert len(out) == 1
        assert out["periode"].iloc[0] == "2025"
        assert out["kommuneregion"].iloc[0] == "0301"
        assert out["personer"].iloc[0] == 35
        assert "kjonn" not in out.columns

        mock_definer_klass.assert_called_once()

    def test_sums_multiple_stat_columns(self, mocker: Any) -> None:
        """Verify that if the helper returns multiple statistikkvariable, the function sums ALL of them."""
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "kommuneregion": ["0301", "0301"],
                "kjonn": ["1", "2"],
                "personer": [10, 20],
                "inntekt": [100, 200],
            }
        )

        mock_definer_klass = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable"
        )
        mock_definer_klass.return_value = (
            ["periode", "kommuneregion", "kjonn"],
            ["personer", "inntekt"],
        )

        out = summere_over_kjonn(df)

        assert len(out) == 1
        assert out["personer"].iloc[0] == 30
        assert out["inntekt"].iloc[0] == 300
        assert "kjonn" not in out.columns

        mock_definer_klass.assert_called_once()
