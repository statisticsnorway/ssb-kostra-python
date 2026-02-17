import io
from typing import Any

import numpy as np
import pandas as pd

from ssb_kostra_python.avrunding import _round_half_up
from ssb_kostra_python.avrunding import konverter_dtypes


class TestRoundingAndKonverterDtypes:
    """Testing the rounding helper and the DataFrame transformation function."""

    def test_round_half_up_basic(self) -> None:
        """Verify the core rounding rule."""
        s = pd.Series([0.5, 1.5, 2.4, 2.5, -0.5, -1.5, -2.5, -2.4])
        out = _round_half_up(s, decimals=0)

        expected = np.array([1, 2, 2, 3, -1, -2, -3, -2], dtype=float)
        assert np.allclose(out, expected, equal_nan=True)

    def test_round_half_up_with_decimals(self) -> None:
        """Verify rounding behavior when decimals != 0."""
        s = pd.Series([1.25, 1.35, -1.25, -1.35])
        out1 = _round_half_up(s, decimals=1)
        out2 = _round_half_up(s, decimals=2)

        assert np.allclose(out1, [1.3, 1.4, -1.3, -1.4])
        assert np.allclose(out2, [1.25, 1.35, -1.25, -1.35])

    def test_konverter_dtypes_converts_groups_correctly(self, mocker: Any) -> None:
        """Verify that konverter_dtypes applies each conversion group correctly."""
        mocker.patch("ssb_kostra_python.avrunding.logger", autospec=True)
        mocker.patch("ssb_kostra_python.avrunding.display", autospec=True)

        df = pd.DataFrame(
            {
                "h": [0.5, 1.5, np.nan, -2.5],
                "d1": [1.25, 1.35, -1.25, np.nan],
                "d2": [2.345, 2.355, -2.345, np.nan],
                "s": [1, None, "A", 4],
                "b": [1, 0, None, 1],
                "keep": ["x", "y", "z", "w"],
            }
        )

        mapping = {
            "heltall": ["h"],
            "desimaltall_1_des": ["d1"],
            "desimaltall_2_des": ["d2"],
            "stringvar": ["s"],
            "bool_var": ["b"],
        }

        out, dtypes = konverter_dtypes(df, mapping)

        assert out is not df
        assert out["h"].tolist() == [1, 2, pd.NA, -3]
        assert str(out["h"].dtype) == "Int64"
        assert np.allclose(
            out["d1"].to_numpy(), [1.3, 1.4, -1.3, np.nan], equal_nan=True
        )
        assert np.allclose(
            out["d2"].to_numpy(), [2.35, 2.36, -2.35, np.nan], equal_nan=True
        )
        assert pd.api.types.is_string_dtype(out["s"])
        assert out["s"].tolist() == ["1", pd.NA, "A", "4"]
        assert pd.api.types.is_bool_dtype(out["b"])
        assert out["b"].tolist() == [True, False, pd.NA, True]
        assert out["keep"].tolist() == ["x", "y", "z", "w"]
        assert dtypes.equals(out.dtypes)

    def test_konverter_dtypes_warns_for_missing_and_unknown_group(
        self, mocker: Any
    ) -> None:
        """Verify that konverter_dtypes warns for missing columns or unknown groups."""
        mocker.patch("ssb_kostra_python.avrunding.logger", autospec=True)
        mock_display = mocker.patch(
            "ssb_kostra_python.avrunding.display", autospec=True
        )

        df = pd.DataFrame({"x": [1, 2]})

        mapping = {
            "heltall": ["missing_col"],
            "weird_group": ["x"],
        }

        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out, dtypes = konverter_dtypes(df, mapping)

        printed = buf.getvalue()

        assert "Advarsel: Kolonnen 'missing_col' finnes ikke i dataframen." in printed
        assert "Advarsel: Ukjent gruppe 'weird_group' for kolonnen 'x'." in printed
        assert out["x"].tolist() == [1, 2]
        assert dtypes.equals(out.dtypes)
        assert mock_display.call_count == 2
