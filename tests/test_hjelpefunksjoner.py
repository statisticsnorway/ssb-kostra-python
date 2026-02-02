import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ssb_kostra_python.hjelpefunksjoner import definere_klassifikasjonsvariable
from ssb_kostra_python.hjelpefunksjoner import format_fil
from ssb_kostra_python.hjelpefunksjoner import konvertere_komma_til_punktdesimal


class TestFormatFil(unittest.TestCase):
    """Tests for `format_fil(df)`.

    What `format_fil` is expected to do (based on the tests)
    --------------------------------------------------------
    1) Ensure certain identifier columns are *string-typed* and *zero-padded*:
       - periode: width 4 (e.g. 1 -> "0001")
       - alder:   width 3 (e.g. 7 -> "007")
    2) Ensure at least one "region" column exists (otherwise raise ValueError):
       - kommuneregion, fylkesregion, bydelsregion (at least one must be present)
    3) Apply zero-padding for region codes only under specific rules:
       - kommuneregion: pad digits-only values shorter than 4 to width 4
       - fylkesregion:  pad digits-only values shorter than 4 to width 4
       - bydelsregion:  pad digits-only values shorter than 6 to width 6
       - values containing non-digits should not be modified
       - missing values should remain missing (pandas NA)
    """

    def test_formats_periode_and_alder_fixed_width(self):
        """Checking correct conversions.

        Purpose
        -------
        Verify that:
          - `periode` values are converted to 4-character zero-padded strings
          - `alder` values are converted to 3-character zero-padded strings
          - the dtypes of these columns are pandas string dtype

        Steps
        -----
        1) Create a DataFrame where periode and alder contain mixed types
           (ints + strings).
        2) Include a valid region column (kommuneregion) so the function doesn't
           raise due to missing region columns.
        3) Run format_fil.
        4) Assert the formatted values and string dtypes.
        """
        df = pd.DataFrame(
            {
                "periode": [1, "23", "2025"],
                "alder": [7, "45", "123"],
                "kommuneregion": ["301", "0301", "9999"],  # required so no ValueError
            }
        )

        out = format_fil(df.copy())

        # Assert: fixed-width formatting
        self.assertEqual(
            out["periode"].tolist(),
            ["0001", "0023", "2025"],
            msg="periode should be zero-padded to 4 characters",
        )
        self.assertEqual(
            out["alder"].tolist(),
            ["007", "045", "123"],
            msg="alder should be zero-padded to 3 characters",
        )

        # Assert: output dtypes are string dtype
        self.assertTrue(
            pd.api.types.is_string_dtype(out["periode"]),
            msg="periode should be pandas string dtype",
        )
        self.assertTrue(
            pd.api.types.is_string_dtype(out["alder"]),
            msg="alder should be pandas string dtype",
        )

    def test_kommuneregion_pads_only_digits_and_only_when_too_short(self):
        """Verify the rules for kommuneregion formatting.

        Purpose
        -------
        Verify the rules for kommuneregion formatting:

        - Only digits-only values are eligible for padding.
        - Only values with length < 4 are padded to width 4.
        - Non-digit strings (e.g. "12A") are not changed.
        - Values longer than 4 are not truncated (e.g. "12345" stays "12345").
        - Missing values remain missing (pandas NA).
        - Output dtype should be pandas string dtype.

        Steps
        -----
        1) Provide kommuneregion values that cover all cases:
           * "301"    -> should become "0301"
           * "0301"   -> unchanged
           * "12A"    -> unchanged (non-digit)
           * "12345"  -> unchanged (already longer than 4)
           * None     -> becomes <NA>
        2) Run format_fil and assert the final list.
        """
        df = pd.DataFrame(
            {
                "periode": [
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                ],  # length matches kommuneregion rows
                "kommuneregion": ["301", "0301", "12A", "12345", None],
            }
        )

        out = format_fil(df.copy())

        self.assertEqual(
            out["kommuneregion"].tolist(),
            ["0301", "0301", "12A", "12345", pd.NA],
            msg="kommuneregion should pad digits-only values shorter than 4; leave others unchanged",
        )
        self.assertTrue(
            pd.api.types.is_string_dtype(out["kommuneregion"]),
            msg="kommuneregion should be pandas string dtype",
        )

    def test_fylkesregion_pads_only_digits_and_only_when_too_short(self):
        """Verify the fylkesregion padding rule.

        Purpose
        -------
        Verify the fylkesregion padding rule:
          - digits-only values with length < 4 are padded to width 4

        Notes:
        -----
        This test also checks that non-digit strings and empty strings are not altered.

        Steps
        -----
        Provide a mix of values:
          "3"    -> "0003"
          "03"   -> "0003" (still padded to width 4)
          "0301" -> unchanged (already length 4)
          "AB"   -> unchanged (non-digit)
          ""     -> unchanged (empty)
        """
        df = pd.DataFrame(
            {
                "fylkesregion": ["3", "03", "0301", "AB", ""],
            }
        )

        out = format_fil(df.copy())

        self.assertEqual(
            out["fylkesregion"].tolist(),
            ["0003", "0003", "0301", "AB", ""],
            msg="fylkesregion should pad digits-only values shorter than 4; leave non-digits/empty unchanged",
        )

    def test_bydelsregion_pads_only_digits_and_only_when_too_short(self):
        """Verify the bydelsregion padding rule.

        Purpose
        -------
        Verify the bydelsregion padding rule:
          - digits-only values with length < 6 are padded to width 6

        Steps
        -----
        Provide values that cover:
          "301"      -> "000301"
          "030101"   -> unchanged (already length 6)
          "12A"      -> unchanged (non-digit)
          "1234567"  -> unchanged (longer than 6, not truncated)
        """
        df = pd.DataFrame(
            {
                "bydelsregion": ["301", "030101", "12A", "1234567"],
            }
        )

        out = format_fil(df.copy())

        self.assertEqual(
            out["bydelsregion"].tolist(),
            ["000301", "030101", "12A", "1234567"],
            msg="bydelsregion should pad digits-only values shorter than 6; leave others unchanged",
        )

    def test_raises_if_no_valid_region_column_present(self):
        """Checking if at least one valid region column exists.

        Purpose
        -------
        format_fil should require that at least one valid region column exists:
          - kommuneregion OR fylkesregion OR bydelsregion

        If none of these are present, it should raise ValueError.

        Steps
        -----
        1) Create df containing periode and alder but no region columns.
        2) Assert ValueError is raised.
        """
        df = pd.DataFrame({"periode": [1, 2], "alder": [10, 20]})

        with self.assertRaisesRegex(
            ValueError,
            r"No valid region column",
            msg="Expected ValueError when no valid region column is present",
        ):
            format_fil(df.copy())


class TestDefinereKlassifikasjonsvariable(unittest.TestCase):
    """Dfining klassifikasjonsvariable and statistikkvariable.

    Tests for `definere_klassifikasjonsvariable(df)`.

    What the function appears to do (inferred from tests)
    -----------------------------------------------------
    1) Determine "klassifikasjonsvariable" (classification variables) from:
       - a fixed set of known columns, but only if present in df:
           ['periode','kommuneregion','fylkesregion','bydelsregion']
       - plus optional extra variables typed by the user in an input prompt
         as a comma-separated list
    2) Return:
       - klassifikasjonsvariable: list of classification column names
       - statistikkvariable: remaining columns not in klassifikasjonsvariable
    3) Convert all selected classification variables to pandas string dtype in-place.

    Note on patching input()
    ------------------------
    These tests patch builtins.input so the function does not prompt during tests
    and returns deterministic results.
    """

    @patch("builtins.input", return_value="")  # user presses Enter
    def test_no_additional_variables(self, mock_input):
        """Checking df with no extra klassifikasjonsvariable.

        Purpose
        -------
        If user provides no extra variables (empty input):
          - classification variables should be only the fixed ones that exist in df
          - statistikkvariable should be everything else
          - fixed classification columns should be converted to string dtype

        Steps
        -----
        1) Create df with fixed vars: periode, kommuneregion, plus a value column.
        2) Patch input() to return "".
        3) Call definere_klassifikasjonsvariable(df).
        4) Assert:
           - klass == ["periode","kommuneregion"]
           - stats == ["value"]
           - dtype of periode/kommuneregion are string dtype (in-place conversion)
        """
        df = pd.DataFrame(
            {
                "periode": [2025, 2026],
                "kommuneregion": [301, 302],
                "value": [1.2, 3.4],
            }
        )

        klass, stats = definere_klassifikasjonsvariable(df)

        self.assertEqual(
            klass,
            ["periode", "kommuneregion"],
            msg="Should include only present fixed classification variables when no extras are provided",
        )
        self.assertEqual(
            stats,
            ["value"],
            msg="Statistikkvariable should be all columns not in klassifikasjonsvariable",
        )

        # dtype check: classification columns converted to pandas string dtype
        self.assertTrue(
            pd.api.types.is_string_dtype(df["periode"]),
            msg="periode should be string dtype after function",
        )
        self.assertTrue(
            pd.api.types.is_string_dtype(df["kommuneregion"]),
            msg="kommuneregion should be string dtype after function",
        )

    @patch(
        "builtins.input", return_value="kjonn, alder , kjonn,  "
    )  # duplicates + spaces
    def test_additional_variables_parsing_dedup_and_order(self, mock_input):
        """Verify parsing.

        Purpose
        -------
        Verify parsing of extra user-provided variables:
          - split on commas
          - strip whitespace
          - remove empty entries
          - remove duplicates while preserving first occurrence order
          - append extras after the fixed vars

        Also verifies:
          - extras are converted to string dtype (in-place)
          - statistikkvariable excludes ALL classification vars

        Steps
        -----
        1) Create df with fixed vars + two extra vars + one stats var.
        2) Patch input to return duplicates and whitespace.
        3) Assert order: fixed first, then extras deduped.
        """
        df = pd.DataFrame(
            {
                "periode": [2025],
                "kommuneregion": ["0301"],
                "kjonn": ["1"],
                "alder": ["007"],
                "stat": [10],
            }
        )

        klass, stats = definere_klassifikasjonsvariable(df)

        self.assertEqual(
            klass,
            ["periode", "kommuneregion", "kjonn", "alder"],
            msg="Order should be fixed vars first, then extra vars; duplicates removed",
        )
        self.assertEqual(
            stats,
            ["stat"],
            msg="Statistikkvariable should exclude all klassifikasjonsvariable",
        )

        # dtype check for extras too
        self.assertTrue(
            pd.api.types.is_string_dtype(df["kjonn"]),
            msg="Extra classification var 'kjonn' should become string dtype",
        )
        self.assertTrue(
            pd.api.types.is_string_dtype(df["alder"]),
            msg="Extra classification var 'alder' should become string dtype",
        )

    @patch("builtins.input", return_value="alder")
    def test_fixed_vars_only_included_if_present(self, mock_input):
        """Verifying that fixed vars are included ONLY if present in the DataFrame.

        Purpose
        -------
        Verify that fixed vars are included ONLY if present in the DataFrame.
        The fixed set is assumed to be:
          ['periode','kommuneregion','fylkesregion','bydelsregion']
        but df may contain only a subset.

        Steps
        -----
        1) Provide df with only fylkesregion among fixed vars (no periode/kommuneregion/bydelsregion).
        2) Patch input to request "alder" as an extra classification variable.
        3) Assert:
           - klass contains only present fixed vars + the extra
           - remaining columns become statistikkvariable
           - dtype conversion happened for chosen classification variables
        """
        df = pd.DataFrame(
            {
                "fylkesregion": [3],
                "alder": [7],
                "value": [99],
            }
        )

        klass, stats = definere_klassifikasjonsvariable(df)

        self.assertEqual(
            klass,
            ["fylkesregion", "alder"],
            msg="Should include only present fixed vars; then user-provided extras",
        )
        self.assertEqual(
            stats,
            ["value"],
            msg="Remaining non-classification columns should be statistikkvariable",
        )

        self.assertTrue(
            pd.api.types.is_string_dtype(df["fylkesregion"]),
            msg="fylkesregion should be string dtype after function",
        )
        self.assertTrue(
            pd.api.types.is_string_dtype(df["alder"]),
            msg="alder should be string dtype after function",
        )


class TestKonvertereKommaTilPunktdesimal(unittest.TestCase):
    """Tests for `konvertere_komma_til_punktdesimal(df)`.

    What the function appears to do (based on tests)
    ------------------------------------------------
    - Identify columns where at least one value contains a comma decimal separator (e.g. "1,5")
    - Convert those columns to float by replacing comma with dot
    - Leave other columns unchanged
    - Return a NEW DataFrame (do not mutate the input df in-place)
    """

    def test_converts_comma_decimal_to_float(self):
        """Purpose.

        -------
        Verify that comma-decimal strings are converted to floats.

        Steps
        -----
        1) Create df with a column of comma-decimal strings.
        2) Call konvertere_komma_til_punktdesimal.
        3) Assert values are floats with dot decimals and dtype is float.
        """
        df = pd.DataFrame({"a": ["1,5", "2,0", "3,25"]})
        out = konvertere_komma_til_punktdesimal(df)

        self.assertTrue(
            np.allclose(out["a"].values, [1.5, 2.0, 3.25]),
            msg="Comma decimals were not correctly converted to float values",
        )
        self.assertTrue(
            pd.api.types.is_float_dtype(out["a"]),
            msg="Column 'a' should have float dtype after conversion",
        )

    def test_leaves_columns_without_commas_unchanged(self):
        """Verifying that columns that do NOT contain comma decimals are unchanged.

        Purpose
        -------
        Verify that columns that do NOT contain comma decimals are unchanged.

        Steps
        -----
        1) Create df with:
           - a comma-decimal column "a" (should convert)
           - a pure string column "b" (should remain unchanged)
           - a numeric column "c" (should remain unchanged)
        2) Call conversion.
        3) Assert b and c are unchanged.
        """
        df = pd.DataFrame(
            {
                "a": ["1,5", "2,0"],
                "b": ["x", "y"],
                "c": [10, 20],
            }
        )
        out = konvertere_komma_til_punktdesimal(df)

        self.assertEqual(
            out["b"].tolist(),
            ["x", "y"],
            msg="Non-numeric string columns without commas should remain unchanged",
        )
        self.assertEqual(
            out["c"].tolist(),
            [10, 20],
            msg="Numeric columns without commas should remain unchanged",
        )

    def test_converts_column_if_any_value_contains_comma(self):
        """Verifying the rule: if ANY value in a column contains a comma, convert the whole column.

        Purpose
        -------
        Verify the rule: if ANY value in a column contains a comma, convert the whole column.

        Steps
        -----
        1) Create a column with mixed values: ["1,5", "2"].
        2) Call conversion.
        3) Assert both values become floats: [1.5, 2.0] and dtype is float.
        """
        df = pd.DataFrame({"a": ["1,5", "2"]})
        out = konvertere_komma_til_punktdesimal(df)

        self.assertTrue(
            np.allclose(out["a"].values, [1.5, 2.0]),
            msg="Column with mixed comma and non-comma values was not converted correctly",
        )
        self.assertTrue(
            pd.api.types.is_float_dtype(out["a"]),
            msg="Column dtype should be float when any value contains a comma",
        )

    def test_does_not_modify_input_dataframe(self):
        """Verify that konvertere_komma_til_punktdesimal does NOT mutate the input DataFrame.

        Purpose
        -------
        Verify that konvertere_komma_til_punktdesimal does NOT mutate the input DataFrame.

        Steps
        -----
        1) Create df and make a deep copy (df_before).
        2) Call konvertere_komma_til_punktdesimal(df).
        3) Assert df is unchanged compared to df_before.
           If changed, fail with a helpful message.
        """
        df = pd.DataFrame({"a": ["1,5", "2,0"]})
        df_before = df.copy(deep=True)

        _ = konvertere_komma_til_punktdesimal(df)

        try:
            pd.testing.assert_frame_equal(df, df_before)
        except AssertionError as e:
            self.fail(
                "Input DataFrame was modified in-place; "
                "function should operate on a copy instead.\n"
                f"{e}"
            )
