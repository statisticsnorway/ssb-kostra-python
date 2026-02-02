import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ssb_kostra_python.avrunding import _round_half_up
from ssb_kostra_python.avrunding import konverter_dtypes


class TestRoundingAndKonverterDtypes(unittest.TestCase):
    """Testing the rounding helper and the DataFrame transformation function.

    Test suite for:
      - _round_half_up(values, decimals)
      - konverter_dtypes(df, dtype_mapping)

    High-level structure:

    1) The first two tests validate ONLY the pure rounding helper (_round_half_up).
       These tests do not involve DataFrames, logger, or display.

    2) The last two tests validate the DataFrame transformation function (konverter_dtypes):
       - that it applies the right conversions per group
       - that it warns (prints) for missing columns / unknown groups
       - that it calls display and logger without crashing

    Note on patching:
      konverter_dtypes imports `display` and `logger` into its own module namespace,
      so we patch:
        - ssb_kostra_python.avrunding.display
        - ssb_kostra_python.avrunding.logger
      to prevent real notebook display output and to assert calls.
    """

    def test_round_half_up_basic(self):
        """Purpose.

        -------
        Verify the core rounding rule: "commercial rounding" / half-away-from-zero.

        Steps
        -----
        1) Create a Series containing:
           - positive halves (0.5, 1.5, 2.5)
           - a non-half value (2.4)
           - negative halves (-0.5, -1.5, -2.5)
           - a negative non-half (-2.4)
        2) Round with decimals=0 (to nearest integer).
        3) Compare to the expected numeric array:
           halves go away from zero:
             0.5 -> 1
            -0.5 -> -1
           and non-halves round normally:
             2.4 -> 2
            -2.4 -> -2
        """
        s = pd.Series([0.5, 1.5, 2.4, 2.5, -0.5, -1.5, -2.5, -2.4])
        out = _round_half_up(s, decimals=0)

        expected = np.array([1, 2, 2, 3, -1, -2, -3, -2], dtype=float)
        self.assertTrue(
            np.allclose(out, expected, equal_nan=True),
            msg="Half values should round away from zero (commercial rounding).",
        )

    def test_round_half_up_with_decimals(self):
        """Purpose.

        -------
        Verify rounding behavior when decimals != 0.

        Steps
        -----
        1) Create a Series with values that are sensitive to 1-decimal rounding:
              1.25, 1.35, -1.25, -1.35
        2) Round to 1 decimal:
              1.25 -> 1.3
              1.35 -> 1.4
             -1.25 -> -1.3
             -1.35 -> -1.4
           (still half-away-from-zero at the chosen decimal precision)
        3) Round to 2 decimals:
           These values already have two decimals, so output should match input.
        """
        s = pd.Series([1.25, 1.35, -1.25, -1.35])
        out1 = _round_half_up(s, decimals=1)
        out2 = _round_half_up(s, decimals=2)

        self.assertTrue(
            np.allclose(out1, [1.3, 1.4, -1.3, -1.4]),
            msg="decimals=1 rounding incorrect.",
        )
        self.assertTrue(
            np.allclose(out2, [1.25, 1.35, -1.25, -1.35]),
            msg="decimals=2 should preserve these values.",
        )

    @patch("ssb_kostra_python.avrunding.logger", autospec=True)
    @patch("ssb_kostra_python.avrunding.display", autospec=True)
    def test_konverter_dtypes_converts_groups_correctly(
        self, mock_display, mock_logger
    ):
        """Purpose.

        -------
        Verify that konverter_dtypes applies each conversion group correctly and
        preserves other columns.

        What this test checks
        ---------------------
        A) The function returns a COPY (does not mutate the original df object).
        B) For each mapping group, the correct conversion happens:
           - "heltall": round half-up to integer and cast to pandas nullable Int64
           - "desimaltall_1_des": round half-up to 1 decimal
           - "desimaltall_2_des": round half-up to 2 decimals
           - "stringvar": cast to pandas string dtype (missing -> <NA>)
           - "bool_var": cast to pandas boolean dtype (missing -> <NA>)
        C) Columns not mentioned in mapping are unchanged.
        D) The returned `df_dtypes` equals `out.dtypes` (consistency check).
        E) Side effects:
           - display() is called twice (once for df, once for dtypes)
           - logger.info() is called (we don't assert exact messages)

        Why we patch display/logger
        ---------------------------
        The function prints/displays output intended for notebooks.
        In unit tests, we:
          - prevent actual display output
          - assert that the calls happened
        """
        # 1) Arrange: create input data with:
        #    - halves and negatives for rounding
        #    - NaNs/None to test nullable conversions
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

        # 2) Arrange: define the intended conversion behavior
        mapping = {
            "heltall": ["h"],
            "desimaltall_1_des": ["d1"],
            "desimaltall_2_des": ["d2"],
            "stringvar": ["s"],
            "bool_var": ["b"],
        }

        # 3) Act: run conversion
        out, dtypes = konverter_dtypes(df, mapping)

        # 4) Assert: original df object is not returned (copy semantics)
        self.assertIsNot(out, df, msg="Function should return a new DataFrame (copy).")

        # 5) Assert: "heltall" conversion (rounding + dtype)
        self.assertEqual(
            out["h"].tolist(),
            [1, 2, pd.NA, -3],
            msg="'h' should be rounded half-up and cast to Int64.",
        )
        self.assertEqual(
            str(out["h"].dtype),
            "Int64",
            msg="'h' should be pandas nullable Int64 dtype.",
        )

        # 6) Assert: decimal rounding conversions
        self.assertTrue(
            np.allclose(out["d1"].values, [1.3, 1.4, -1.3, np.nan], equal_nan=True),
            msg="'d1' should be half-up rounded to 1 decimal.",
        )
        self.assertTrue(
            np.allclose(out["d2"].values, [2.35, 2.36, -2.35, np.nan], equal_nan=True),
            msg="'d2' should be half-up rounded to 2 decimals.",
        )

        # 7) Assert: string conversion
        self.assertTrue(
            pd.api.types.is_string_dtype(out["s"]),
            msg="'s' should be pandas string dtype.",
        )
        self.assertEqual(
            out["s"].tolist(),
            ["1", pd.NA, "A", "4"],
            msg="'s' should be string-cast with missing as <NA>.",
        )

        # 8) Assert: boolean conversion
        self.assertTrue(
            pd.api.types.is_bool_dtype(out["b"]), msg="'b' should be boolean dtype."
        )
        self.assertEqual(
            out["b"].tolist(),
            [True, False, pd.NA, True],
            msg="'b' should be boolean with NA preserved.",
        )

        # 9) Assert: columns not in mapping remain unchanged
        self.assertEqual(
            out["keep"].tolist(),
            ["x", "y", "z", "w"],
            msg="Columns not in mapping should remain unchanged.",
        )

        # 10) Assert: returned dtypes object matches out.dtypes
        self.assertTrue(
            dtypes.equals(out.dtypes), msg="Returned df_dtypes should equal out.dtypes."
        )

        # 11) Assert: side-effects were invoked
        self.assertEqual(
            mock_display.call_count,
            2,
            msg="Expected display to be called exactly twice (df and dtypes).",
        )
        self.assertTrue(
            mock_logger.info.called, msg="Expected logger.info to be called."
        )

    @patch("ssb_kostra_python.avrunding.logger", autospec=True)
    @patch("ssb_kostra_python.avrunding.display", autospec=True)
    def test_konverter_dtypes_warns_for_missing_and_unknown_group(
        self, mock_display, mock_logger
    ):
        """Purpose.

        -------
        Verify that konverter_dtypes:
          - prints warnings when:
              * a mapped column does not exist
              * a mapping group name is unknown
          - does NOT crash
          - leaves unaffected columns unchanged
          - still returns (df, df_dtypes) consistently
          - still calls display()

        Steps
        -----
        1) Create a minimal DataFrame with only column 'x'.
        2) Provide a mapping that includes:
           - a missing column under a valid group ("heltall": ["missing_col"])
           - an unknown group ("weird_group": ["x"])
        3) Capture stdout while running konverter_dtypes to inspect printed warnings.
        4) Assert that:
           - both warnings appear in printed output
           - 'x' is unchanged
           - returned dtypes matches out.dtypes
           - display() was called twice
        """
        # 1) Arrange
        df = pd.DataFrame({"x": [1, 2]})

        mapping = {
            "heltall": ["missing_col"],  # missing column -> should warn
            "weird_group": ["x"],  # unknown group -> should warn, and do nothing
        }

        # 2) Act (capture printed warnings)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out, dtypes = konverter_dtypes(df, mapping)

        printed = buf.getvalue()

        # 3) Assert: warnings printed
        self.assertIn(
            "Advarsel: Kolonnen 'missing_col' finnes ikke i dataframen.", printed
        )
        self.assertIn(
            "Advarsel: Ukjent gruppe 'weird_group' for kolonnen 'x'.", printed
        )

        # 4) Assert: data unaffected
        self.assertEqual(out["x"].tolist(), [1, 2])

        # 5) Assert: dtype consistency
        self.assertTrue(dtypes.equals(out.dtypes))

        # 6) Assert: display still called twice
        self.assertEqual(mock_display.call_count, 2)
