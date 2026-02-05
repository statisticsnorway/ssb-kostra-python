import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

from functions.funksjoner.titler_til_klasskoder import (
    _attach_one_mapping,
    _fetch_mapping_for_year,
    kodelister_navn,
    _pick_level_columns,
)

# --- Test doubles (fakes) for KLASS ---

class FakeCodes:
    """
    Test double for the object returned by KlassClassification.get_codes(...)

    In the real 'klass' library:
      klass.get_codes(...) returns an object that has .pivot_level()

    Our code under test calls:
      codes = klass.get_codes(...)
      pivot = codes.pivot_level()

    So this fake only needs to implement pivot_level() and return a DataFrame.
    """
    def __init__(self, pivot_df: pd.DataFrame):
        self._pivot_df = pivot_df

    def pivot_level(self):
        """Return the pivot table as if it came from real KLASS."""
        return self._pivot_df


class FakeKlassClassification:
    """
    Test double for klass.KlassClassification.

    The production code does:
      klass = KlassClassification(klass_id, ...)
      codes = klass.get_codes(...)
      pivot = codes.pivot_level()

    This fake:
      - ignores network/API access
      - returns a FakeCodes instance whose pivot_df is provided by the test

    The class attribute `pivot_df` is set per test to control the mapping.
    """
    pivot_df = None  # set per test

    def __init__(self, klass_id, language="nb", include_future=True):
        self.klass_id = klass_id
        self.language = language
        self.include_future = include_future

    def get_codes(self, **kwargs):
        """
        Return a FakeCodes object containing the test-provided pivot table.

        The kwargs are accepted so the signature looks compatible with the real API,
        but they are not used.
        """
        return FakeCodes(self.__class__.pivot_df)


# IMPORTANT:
# The production module imports KlassClassification like:
#   from klass import KlassClassification
# so the code under test references the symbol in *its own module namespace*.
# Therefore we patch the KlassClassification name where it is looked up:
PATCH_TARGET = "functions.funksjoner.titler_til_klasskoder.KlassClassification"


class TestKlassMappingHelpersAndMain(unittest.TestCase):
    """
    Test suite for helper functions + main function in titler_til_klasskoder:

      - _pick_level_columns(pivot_df, level)
      - _fetch_mapping_for_year(klass_id, year, ...)
      - _attach_one_mapping(df_in, year, code_col, klass_id, ...)
      - kodelister_navn(df, mappings, ...)

    Organization:
      1) _pick_level_columns: pure logic (no external KLASS dependency)
      2) _fetch_mapping_for_year: uses KLASS -> patched with FakeKlassClassification
      3) _attach_one_mapping: uses fetch mapping + merge + diagnostics
      4) kodelister_navn: validates periode + applies multiple mappings sequentially
    """

    # -------------------------
    # _pick_level_columns tests
    # -------------------------

    def test_pick_level_columns_selects_smallest_level_when_none(self):
        """
        Purpose
        -------
        If select_level=None, _pick_level_columns should pick the smallest available
        level number from columns like "code_1", "code_3", etc.

        Steps
        -----
        1) Build a pivot DF that contains both level 3 and level 1 columns.
        2) Call _pick_level_columns(..., level=None).
        3) Assert it chose level 1 and returned the correct column names.
        """
        pivot = pd.DataFrame({
            "code_3": ["A"],
            "name_3": ["NameA"],
            "code_1": ["B"],
            "name_1": ["NameB"],
        })

        level, mcode, mname = _pick_level_columns(pivot, level=None)

        self.assertEqual(level, 1)
        self.assertEqual(mcode, "code_1")
        self.assertEqual(mname, "name_1")

    def test_pick_level_columns_raises_if_missing_expected_columns(self):
        """
        Purpose
        -------
        If the pivot DF does NOT contain any 'code_*' or 'name_*' columns,
        the function should raise a RuntimeError.

        Steps
        -----
        1) Provide a pivot DF with unrelated columns only.
        2) Assert a RuntimeError is raised with a helpful message.
        """
        pivot = pd.DataFrame({"something_else": [1]})
        with self.assertRaisesRegex(RuntimeError, "missing expected"):
            _pick_level_columns(pivot, level=None)

    def test_pick_level_columns_raises_if_requested_level_not_present(self):
        """
        Purpose
        -------
        If a specific level is requested (e.g. level=2) but the pivot DF only
        has code_1/name_1, the function should raise a RuntimeError.

        Steps
        -----
        1) Provide a pivot DF containing only code_1/name_1.
        2) Request level=2.
        3) Assert the correct RuntimeError is raised.
        """
        pivot = pd.DataFrame({"code_1": ["1"], "name_1": ["One"]})
        with self.assertRaisesRegex(RuntimeError, "Expected columns 'code_2' and 'name_2'"):
            _pick_level_columns(pivot, level=2)

    # --------------------------------
    # _fetch_mapping_for_year tests
    # --------------------------------

    @patch(PATCH_TARGET, new=FakeKlassClassification)
    def test_fetch_mapping_for_year_returns_two_col_mapping_and_level(self):
        """
        Purpose
        -------
        Verify that _fetch_mapping_for_year:
          - calls into KLASS (here: our fake)
          - pivots the result
          - picks the correct (code_X, name_X) columns
          - returns a 2-column mapping DataFrame:
                ["_map_code", "_map_name"]
          - strips whitespace from codes
          - returns the level used

        Steps
        -----
        1) Configure the fake KLASS pivot table with code_1/name_1.
        2) Call _fetch_mapping_for_year(..., select_level=None).
        3) Assert:
           - level=1 chosen
           - mapping has correct columns and cleaned values
        """
        FakeKlassClassification.pivot_df = pd.DataFrame({
            "code_1": [" 0301 ", "0302"],
            "name_1": ["Oslo", "Bergen"],
        })

        mapping, level = _fetch_mapping_for_year(
            klass_id=231,
            year=2024,
            language="nb",
            include_future=True,
            select_level=None,
        )

        self.assertEqual(level, 1)
        self.assertEqual(list(mapping.columns), ["_map_code", "_map_name"])
        self.assertEqual(mapping["_map_code"].tolist(), ["0301", "0302"])
        self.assertEqual(mapping["_map_name"].tolist(), ["Oslo", "Bergen"])

    # --------------------------------
    # _attach_one_mapping tests
    # --------------------------------

    @patch(PATCH_TARGET, new=FakeKlassClassification)
    def test_attach_one_mapping_inserts_name_column_after_code_and_builds_diagnostics(self):
        """
        Purpose
        -------
        Verify that _attach_one_mapping:
          - merges the mapping onto the input DataFrame
          - inserts the name column immediately after the code column
          - produces missing names for unmapped codes
          - builds diagnostics about invalid data codes

        Steps
        -----
        1) Configure fake mapping:
             0301 -> Oslo
             0302 -> Bergen
        2) Build df_in with codes: 0301, 9999, 0302 (9999 is invalid).
        3) Run _attach_one_mapping(...).
        4) Assert:
           - new column 'kommuneregion_navn' exists right after 'kommuneregion'
           - names match mapping; invalid code yields NaN
           - diagnostics report exactly one invalid code ("9999")
        """
        FakeKlassClassification.pivot_df = pd.DataFrame({
            "code_1": ["0301", "0302"],
            "name_1": ["Oslo", "Bergen"],
        })

        df_in = pd.DataFrame({
            "periode": [2024, 2024, 2024],
            "kommuneregion": ["0301", "9999", "0302"],
            "value": [10, 20, 30],
        })

        out, diag = _attach_one_mapping(
            df_in,
            year=2024,
            code_col="kommuneregion",
            klass_id=231,
            name_col_out=None,   # triggers default: f"{code_col}_navn"
            language="nb",
            include_future=True,
            select_level=1,
        )

        # Assert column placement: inserted immediately after the code column
        cols = list(out.columns)
        idx_code = cols.index("kommuneregion")
        self.assertEqual(cols[idx_code + 1], "kommuneregion_navn")

        # Assert merge results: invalid code should produce missing (NaN/NA)
        self.assertEqual(out["kommuneregion_navn"].tolist(), ["Oslo", np.nan, "Bergen"])

        # Assert diagnostics are correct and stable
        self.assertEqual(diag["invalid_count"], 1)
        self.assertEqual(diag["invalid_sample"], ["9999"])
        self.assertEqual(diag["all_invalid"], ["9999"])
        self.assertEqual(diag["level"], 1)
        self.assertEqual(diag["year"], 2024)
        self.assertEqual(diag["code_col"], "kommuneregion")

    @patch(PATCH_TARGET, new=FakeKlassClassification)
    def test_attach_one_mapping_raises_if_code_col_missing(self):
        """
        Purpose
        -------
        _attach_one_mapping should fail fast if the requested code_col does not
        exist in the input DataFrame.

        Steps
        -----
        1) Create df_in without the 'kommuneregion' column.
        2) Call _attach_one_mapping(..., code_col="kommuneregion", ...).
        3) Assert that a ValueError is raised with a clear message.
        """
        FakeKlassClassification.pivot_df = pd.DataFrame({"code_1": ["1"], "name_1": ["One"]})

        df_in = pd.DataFrame({"periode": [2024], "value": [1]})

        with self.assertRaisesRegex(ValueError, "Column 'kommuneregion' not found"):
            _attach_one_mapping(df_in, year=2024, code_col="kommuneregion", klass_id=231)

    # -----------------------
    # kodelister_navn tests
    # -----------------------

    def test_kodelister_navn_requires_periode(self):
        """
        Purpose
        -------
        kodelister_navn requires a 'periode' column containing the year.
        If it's missing, the function should raise a ValueError.

        Steps
        -----
        1) Create df without 'periode'.
        2) Call kodelister_navn(...)
        3) Assert it raises with a helpful message.
        """
        df = pd.DataFrame({"kommuneregion": ["0301"]})
        with self.assertRaisesRegex(ValueError, "must contain a 'periode'"):
            kodelister_navn(df, mappings=[{"code_col": "kommuneregion", "klass_id": 231}], verbose=False)

    def test_kodelister_navn_requires_exactly_one_unique_year(self):
        """
        Purpose
        -------
        kodelister_navn expects exactly ONE unique year in df['periode'].
        If there are multiple years, it should raise.

        Steps
        -----
        1) Create df with periode containing two different years.
        2) Call kodelister_navn(...)
        3) Assert it raises.
        """
        df = pd.DataFrame({"periode": [2023, 2024], "kommuneregion": ["0301", "0301"]})
        with self.assertRaisesRegex(ValueError, "exactly one unique value"):
            kodelister_navn(df, mappings=[{"code_col": "kommuneregion", "klass_id": 231}], verbose=False)

    @patch(PATCH_TARGET, new=FakeKlassClassification)
    def test_kodelister_navn_applies_multiple_mappings_in_order(self):
        """
        Purpose
        -------
        Verify that kodelister_navn:
          - reads year from df['periode']
          - applies multiple mappings sequentially (the output of the first
            mapping becomes the input to the next)
          - inserts each name column immediately after its code column
          - respects custom output column name via 'name_col_out'
          - returns diagnostics for each mapping entry

        Steps
        -----
        1) Configure one fake mapping (A->Alpha, B->Beta).
           We reuse it for both klass_id values since this is a behavior test,
           not a KLASS correctness test.
        2) Create df with two code columns col1 and col2.
        3) Provide two mapping specs:
           - col1 uses default name col: "col1_navn"
           - col2 uses explicit name col: "col2_name"
        4) Assert column placement, values, and diagnostics keys.
        """
        FakeKlassClassification.pivot_df = pd.DataFrame({
            "code_1": ["A", "B"],
            "name_1": ["Alpha", "Beta"],
        })

        df = pd.DataFrame({
            "periode": [2024, 2024],
            "col1": ["A", "B"],
            "col2": ["B", "A"],
            "value": [1, 2],
        })

        mappings = [
            {"code_col": "col1", "klass_id": 111, "select_level": 1},
            {"code_col": "col2", "klass_id": 222, "name_col_out": "col2_name", "select_level": 1},
        ]

        out, diag = kodelister_navn(df, mappings=mappings, verbose=False)

        cols = list(out.columns)

        # Assert mapping #1 results and placement
        i1 = cols.index("col1")
        self.assertEqual(cols[i1 + 1], "col1_navn")
        self.assertEqual(out["col1_navn"].tolist(), ["Alpha", "Beta"])

        # Assert mapping #2 results and placement
        i2 = cols.index("col2")
        self.assertEqual(cols[i2 + 1], "col2_name")
        self.assertEqual(out["col2_name"].tolist(), ["Beta", "Alpha"])

        # Assert diagnostics exist per mapping
        self.assertIn("col1", diag)
        self.assertIn("col2", diag)
        self.assertEqual(diag["col1"]["klass_id"], 111)
        self.assertEqual(diag["col2"]["klass_id"], 222)
