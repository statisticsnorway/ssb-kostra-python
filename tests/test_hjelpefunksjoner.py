import numpy as np
import pandas as pd
import pytest

from ssb_kostra_python.hjelpefunksjoner import definere_klassifikasjonsvariable
from ssb_kostra_python.hjelpefunksjoner import format_fil
from ssb_kostra_python.hjelpefunksjoner import konvertere_komma_til_punktdesimal


class TestFormatFil:
    """Tests for `format_fil(df)`."""

    def test_formats_periode_and_alder_fixed_width(self):
        """Checking correct conversions."""
        df = pd.DataFrame(
            {
                "periode": [1, "23", "2025"],
                "alder": [7, "45", "123"],
                "kommuneregion": ["301", "0301", "9999"],  # required so no ValueError
            }
        )

        out = format_fil(df.copy())

        # Assert: fixed-width formatting
        assert out["periode"].tolist() == ["0001", "0023", "2025"]
        assert out["alder"].tolist() == ["007", "045", "123"]

        # Assert: output dtypes are string dtype
        assert pd.api.types.is_string_dtype(out["periode"])
        assert pd.api.types.is_string_dtype(out["alder"])

    def test_kommuneregion_pads_only_digits_and_only_when_too_short(self):
        """Verify the rules for kommuneregion formatting."""
        df = pd.DataFrame(
            {
                "periode": [
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                ],
                "kommuneregion": ["301", "0301", "12A", "12345", None],
            }
        )

        out = format_fil(df.copy())

        assert out["kommuneregion"].tolist() == ["0301", "0301", "12A", "12345", pd.NA]
        assert pd.api.types.is_string_dtype(out["kommuneregion"])

    def test_fylkesregion_pads_only_digits_and_only_when_too_short(self):
        """Verify the fylkesregion padding rule."""
        df = pd.DataFrame(
            {
                "fylkesregion": ["3", "03", "0301", "AB", ""],
            }
        )

        out = format_fil(df.copy())

        assert out["fylkesregion"].tolist() == ["0003", "0003", "0301", "AB", ""]

    def test_bydelsregion_pads_only_digits_and_only_when_too_short(self):
        """Verify the bydelsregion padding rule."""
        df = pd.DataFrame(
            {
                "bydelsregion": ["301", "030101", "12A", "1234567"],
            }
        )

        out = format_fil(df.copy())

        assert out["bydelsregion"].tolist() == ["000301", "030101", "12A", "1234567"]

    def test_raises_if_no_valid_region_column_present(self):
        """Checking if at least one valid region column exists."""
        df = pd.DataFrame({"periode": [1, 2], "alder": [10, 20]})

        with pytest.raises(ValueError, match="No valid region column"):
            format_fil(df.copy())


class TestDefinereKlassifikasjonsvariable:
    """Defining klassifikasjonsvariable and statistikkvariable."""

    def test_no_additional_variables(self, mocker):
        """Checking df with no extra klassifikasjonsvariable."""
        df = pd.DataFrame(
            {
                "periode": [2025, 2026],
                "kommuneregion": [301, 302],
                "value": [1.2, 3.4],
            }
        )

        mocker.patch("builtins.input", return_value="")

        klass, stats = definere_klassifikasjonsvariable(df)

        assert klass == ["periode", "kommuneregion"]
        assert stats == ["value"]

        # dtype check: classification columns converted to pandas string dtype
        assert pd.api.types.is_string_dtype(df["periode"])
        assert pd.api.types.is_string_dtype(df["kommuneregion"])

    def test_additional_variables_parsing_dedup_and_order(self, mocker):
        """Verify parsing."""
        df = pd.DataFrame(
            {
                "periode": [2025],
                "kommuneregion": ["0301"],
                "kjonn": ["1"],
                "alder": ["007"],
                "stat": [10],
            }
        )

        mocker.patch("builtins.input", return_value="kjonn, alder , kjonn,  ")

        klass, stats = definere_klassifikasjonsvariable(df)

        assert klass == ["periode", "kommuneregion", "kjonn", "alder"]
        assert stats == ["stat"]

        # dtype check for extras too
        assert pd.api.types.is_string_dtype(df["kjonn"])
        assert pd.api.types.is_string_dtype(df["alder"])

    def test_fixed_vars_only_included_if_present(self, mocker):
        """Verifying that fixed vars are included ONLY if present in the DataFrame."""
        df = pd.DataFrame(
            {
                "fylkesregion": [3],
                "alder": [7],
                "value": [99],
            }
        )

        mocker.patch("builtins.input", return_value="alder")

        klass, stats = definere_klassifikasjonsvariable(df)

        assert klass == ["fylkesregion", "alder"]
        assert stats == ["value"]

        assert pd.api.types.is_string_dtype(df["fylkesregion"])
        assert pd.api.types.is_string_dtype(df["alder"])


class TestKonvertereKommaTilPunktdesimal:
    """Tests for `konvertere_komma_til_punktdesimal(df)`."""

    def test_converts_comma_decimal_to_float(self):
        """Verify that comma-decimal strings are converted to floats."""
        df = pd.DataFrame({"a": ["1,5", "2,0", "3,25"]})
        out = konvertere_komma_til_punktdesimal(df)

        assert np.allclose(out["a"].values, [1.5, 2.0, 3.25])
        assert pd.api.types.is_float_dtype(out["a"])

    def test_leaves_columns_without_commas_unchanged(self):
        """Verifying that columns that do NOT contain comma decimals are unchanged."""
        df = pd.DataFrame(
            {
                "a": ["1,5", "2,0"],
                "b": ["x", "y"],
                "c": [10, 20],
            }
        )
        out = konvertere_komma_til_punktdesimal(df)

        assert out["b"].tolist() == ["x", "y"]
        assert out["c"].tolist() == [10, 20]

    def test_converts_column_if_any_value_contains_comma(self):
        """Verifying the rule: if ANY value in a column contains a comma, convert the whole column."""
        df = pd.DataFrame({"a": ["1,5", "2"]})
        out = konvertere_komma_til_punktdesimal(df)

        assert np.allclose(out["a"].values, [1.5, 2.0])
        assert pd.api.types.is_float_dtype(out["a"])

    def test_does_not_modify_input_dataframe(self):
        """Verify that konvertere_komma_til_punktdesimal does NOT mutate the input DataFrame."""
        df = pd.DataFrame({"a": ["1,5", "2,0"]})
        df_before = df.copy(deep=True)

        _ = konvertere_komma_til_punktdesimal(df)

        pd.testing.assert_frame_equal(df, df_before)
