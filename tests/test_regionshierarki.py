import unittest
from unittest.mock import patch

import pandas as pd

from ssb_kostra_python.regionshierarki import hierarki
from ssb_kostra_python.regionshierarki import mapping_bydeler_oslo
from ssb_kostra_python.regionshierarki import mapping_fra_fylkeskommune_til_kostraregion
from ssb_kostra_python.regionshierarki import mapping_fra_kommune_til_fylkeskommune
from ssb_kostra_python.regionshierarki import mapping_fra_kommune_til_landet

# =============================================================================
# SECTION 1: mapping_bydeler_oslo (BYDELER)
# =============================================================================
#
# This part tests mapping_bydeler_oslo(year) which (based on the tests) should:
#   - fetch codes from KLASS via KlassClassification
#   - filter out certain codes (e.g. "030116", "030117", "EAB")
#   - produce a DataFrame with columns ["from", "to"]
#   - set all "to" values to "EAB"
#
# We patch KlassClassification so we can control which codes appear without
# calling any external services.
# =============================================================================


# ---- Fakes for KLASSClassification used by mapping_bydeler_oslo ----
class FakeCodes_bb_eab:
    """Test double for the object returned by KlassClassification.get_codes(...).

    Production code does:
      codes = klass.get_codes(...)
      pivot = codes.pivot_level()

    So this fake only needs pivot_level().
    """

    def __init__(self, pivot_df: pd.DataFrame):
        """Initialize fake with dataframe returned by pivot_level()."""
        self._pivot_df = pivot_df

    def pivot_level(self):
        return self._pivot_df


class FakeKlassClassification_bb_eab:
    """Test double for KlassClassification used by mapping_bydeler_oslo.

    The function under test likely does something like:
      klass = KlassClassification(...)
      codes = klass.get_codes(...)
      pivot = codes.pivot_level()
      ... filter + build mapping ...

    We control the pivot table via the class attribute `pivot_df`.
    """

    pivot_df = None  # set per test

    def __init__(self, klass_id, language="nb", include_future=True):
        """Initialize fake KlassClassification with expected constructor arguments."""
        self.klass_id = klass_id
        self.language = language
        self.include_future = include_future

    def get_codes(self, *args, **kwargs):
        """Return fake codes object for testing.

        mapping_bydeler_oslo calls get_codes(...) (possibly with a date string).
        We ignore args/kwargs and return our fake codes object.
        """
        return FakeCodes_bb_eab(self.__class__.pivot_df)


class TestMappingBydelerOslo(unittest.TestCase):
    """Tests for mapping_bydeler_oslo(year).

    Main behaviors tested:
      1) Proper output schema: columns ["from","to"]
      2) Filtering: exclude specific bydel codes + exclude "EAB"
      3) Mapping: all remaining rows map to "EAB"
      4) Edge case: if only filtered codes exist, output is empty but schema remains
    """

    @patch(
        "ssb_kostra_python.regionshierarki.KlassClassification",
        new=FakeKlassClassification_bb_eab,
    )
    def test_mapping_bydeler_oslo_filters_and_sets_to_EAB(self):
        """Checks if the function maps and filters correctly.

        Purpose
        -------
        Verify that mapping_bydeler_oslo:
          - filters out specific codes (030116, 030117, and EAB)
          - keeps other codes
          - maps every kept code to 'EAB'
          - returns DataFrame with columns ["from","to"]

        Steps
        -----
        1) Provide a fake KLASS pivot table including:
           - codes to be filtered out
           - codes to keep
        2) Call mapping_bydeler_oslo(year=...)
        3) Assert schema, filtered codes absent, remaining present, to="EAB" for all.
        """
        FakeKlassClassification_bb_eab.pivot_df = pd.DataFrame(
            {
                "code_1": ["030101", "030116", "030117", "030102", "EAB", "030103"],
                "name_1": [
                    "A",
                    "B",
                    "C",
                    "D",
                    "Samlebydel",
                    "E",
                ],  # extra cols shouldn't matter
            }
        )

        out = mapping_bydeler_oslo(year="2024")

        # Assert schema
        self.assertEqual(
            list(out.columns),
            ["from", "to"],
            msg="Output must have columns ['from','to'].",
        )

        # Assert filtered codes are not present in "from"
        self.assertNotIn(
            "030116", out["from"].tolist(), msg="030116 should be filtered out."
        )
        self.assertNotIn(
            "030117", out["from"].tolist(), msg="030117 should be filtered out."
        )
        self.assertNotIn(
            "EAB",
            out["from"].tolist(),
            msg="EAB should be filtered out from 'from' column.",
        )

        # Assert only the expected non-filtered codes remain
        self.assertCountEqual(
            out["from"].tolist(),
            ["030101", "030102", "030103"],
            msg="Output 'from' should contain only the non-filtered codes.",
        )

        # Assert all map to EAB
        self.assertTrue(
            (out["to"] == "EAB").all(), msg="All 'to' values should be 'EAB'."
        )

    @patch(
        "ssb_kostra_python.regionshierarki.KlassClassification",
        new=FakeKlassClassification_bb_eab,
    )
    def test_mapping_bydeler_oslo_returns_empty_if_only_filtered_codes(self):
        """Verify behavior when all available codes are filtered out.

        Purpose
        -------
        Verify behavior when all available codes are filtered out:
          - output should be empty (0 rows)
          - but should still have the correct schema ["from","to"]

        Steps
        -----
        1) Provide pivot table containing only filtered codes.
        2) Call mapping_bydeler_oslo.
        3) Assert len(out) == 0 and correct columns.
        """
        FakeKlassClassification_bb_eab.pivot_df = pd.DataFrame(
            {
                "code_1": ["030116", "030117", "EAB"],
            }
        )

        out = mapping_bydeler_oslo(year=2024)

        self.assertEqual(list(out.columns), ["from", "to"])
        self.assertEqual(
            len(out), 0, msg="If only filtered codes exist, output should be empty."
        )


# =============================================================================
# SECTION 2: mapping_fra_kommune_til_landet (KOMMUNER -> REGIONSGRUPPERINGER)
# =============================================================================
#
# This part tests mapping_fra_kommune_til_landet(year), which (based on the tests)
# likely does:
#   - Fetch municipality codes via KlassClassification (code_1)
#   - Build mappings using KlassCorrespondence:
#       kommune -> fylke (target classification 104)
#       kommune -> KOSTRA-group (target classification 112)
#   - Create derived "landet" groups like:
#       EAK      (landet)
#       EAKUO    (landet uten Oslo)
#       EKAxx    (fylke-based grouping derived from first two digits)
#   - Filter out invalid/placeholder codes like "9999"
#   - Ensure 'from' is zero padded to 4 characters
#
# We patch both KlassCorrespondence and KlassClassification to use deterministic fakes.
# =============================================================================


class FakeKlassCorrespondence_kk_eak:
    """Fake KlassCorrespondence that provides a `.data` DataFrame.

    The production code likely does something like:
      corr = KlassCorrespondence(source_id, target_id, from_date, to_date)
      df_corr = corr.data

    Here we return different deterministic data depending on target_classification_id
    to simulate:
      - kommune -> fylke (104)
      - kommune -> KOSTRA-group (112)
    """

    def __init__(
        self, source_classification_id, target_classification_id, from_date, to_date
    ):
        """Initialize fake KlassCorrespondence with deterministic mapping data.

        The returned mapping depends on target_classification_id to simulate
        kommune-to-fylke and kommune-to-KOSTRA-group correspondences.
        """
        self.source_classification_id = source_classification_id
        self.target_classification_id = target_classification_id
        self.from_date = from_date
        self.to_date = to_date

        if str(target_classification_id) == "104":  # kommune -> fylke
            self.data = pd.DataFrame(
                {
                    "sourceCode": ["0301", "5001", "9999"],
                    "targetCode": ["0300", "5000", "9999"],
                }
            )
        elif str(target_classification_id) == "112":  # kommune -> KOSTRA-gruppe
            self.data = pd.DataFrame(
                {
                    "sourceCode": ["0301", "5001", "9999"],
                    "targetCode": ["K1", "K2", "KX"],
                }
            )
        else:
            self.data = pd.DataFrame(columns=["sourceCode", "targetCode"])


class FakeCodes_kk_eak:
    """Fake codes object providing pivot_level(), used by FakeKlassClassification_kk_eak."""

    def __init__(self, pivot_df: pd.DataFrame):
        """Fake codes object providing pivot_level(), used by FakeKlassClassification_kk_eak."""
        self._pivot_df = pivot_df

    def pivot_level(self):
        return self._pivot_df


class FakeKlassClassification_kk_eak:
    """Fake KlassClassification returning a FakeCodes object.

    The production code likely calls:
      klass = KlassClassification(...)
      codes = klass.get_codes(...)
      pivot = codes.pivot_level()

    We control pivot via FakeKlassClassification_kk_eak.pivot_df.
    """

    pivot_df = None

    def __init__(self, klass_id, language="nb", include_future=True):
        self.klass_id = klass_id
        self.language = language
        self.include_future = include_future

    def get_codes(self, *args, **kwargs):
        return FakeCodes_kk_eak(self.__class__.pivot_df)


class TestMappingFraKommuneTilLandet(unittest.TestCase):
    """Tests for mapping_fra_kommune_til_landet(year).

    Primary behaviors tested:
      - Filtering out code "9999"
      - Transformation kommune->fylke into EKAxx (based on first 2 digits of targetCode)
      - Inclusion of kommune->KOSTRA-group mapping rows
      - Adding EAK and EAKUO rows based on classification codes
      - Enforcing that "from" values are zero-padded to 4 digits
    """

    @patch(
        "ssb_kostra_python.regionshierarki.KlassCorrespondence",
        new=FakeKlassCorrespondence_kk_eak,
    )
    @patch(
        "ssb_kostra_python.regionshierarki.KlassClassification",
        new=FakeKlassClassification_kk_eak,
    )
    def test_mapping_fra_kommune_til_landet_happy_path(self):
        """Validate the main invariants of mapping_fra_kommune_til_landet.

        Purpose
        -------
        Validate the main invariants of mapping_fra_kommune_til_landet:

        - 9999 is filtered out everywhere
        - kommune->fylke produces EKAxx where xx = first 2 digits of fylke code
        - kommune->KOSTRA-group mapping rows are included
        - EAK row exists for each municipality code except filtered ones
        - EAKUO row exists for each non-Oslo municipality code (and non-9999)
        - "from" is always padded to 4 characters

        Steps
        -----
        1) Provide fake municipality classification codes: 0301, 5001, 9999.
        2) Call mapping_fra_kommune_til_landet(year=...).
        3) Assert output schema and that all expected mapping rows exist.
        """
        FakeKlassClassification_kk_eak.pivot_df = pd.DataFrame(
            {
                "code_1": ["0301", "5001", "9999"],
                "name_1": ["Oslo", "Trondheim", "Ugyldig"],
            }
        )

        out = mapping_fra_kommune_til_landet(year="2024")

        self.assertEqual(list(out.columns), ["from", "to"])

        # 9999 must be removed
        self.assertFalse(
            (out["from"] == "9999").any(),
            msg="Row(s) with from=9999 should be filtered out.",
        )

        # All "from" should be 4-digit strings
        self.assertTrue(
            out["from"].str.len().eq(4).all(),
            msg="'from' should be padded to 4 digits.",
        )

        # kommune->fylke -> EKAxx
        self.assertTrue(
            ((out["from"] == "0301") & (out["to"] == "EKA03")).any(),
            msg="Expected kommune 0301 to map to EKA03 (from fylke targetCode 0300).",
        )
        self.assertTrue(
            ((out["from"] == "5001") & (out["to"] == "EKA50")).any(),
            msg="Expected kommune 5001 to map to EKA50 (from fylke targetCode 5000).",
        )

        # kommune->KOSTRA group rows
        self.assertTrue(((out["from"] == "0301") & (out["to"] == "K1")).any())
        self.assertTrue(((out["from"] == "5001") & (out["to"] == "K2")).any())

        # EAK exists for each municipality except filtered ones
        self.assertTrue(((out["from"] == "0301") & (out["to"] == "EAK")).any())
        self.assertTrue(((out["from"] == "5001") & (out["to"] == "EAK")).any())

        # EAKUO excludes Oslo
        self.assertFalse(
            ((out["from"] == "0301") & (out["to"] == "EAKUO")).any(),
            msg="Oslo (0301) should NOT be included in EAKUO mapping.",
        )
        self.assertTrue(
            ((out["from"] == "5001") & (out["to"] == "EAKUO")).any(),
            msg="Non-Oslo municipality should be included in EAKUO mapping.",
        )

    @patch(
        "ssb_kostra_python.regionshierarki.KlassCorrespondence",
        new=FakeKlassCorrespondence_kk_eak,
    )
    @patch(
        "ssb_kostra_python.regionshierarki.KlassClassification",
        new=FakeKlassClassification_kk_eak,
    )
    def test_from_is_zero_padded_when_short(self):
        """Specifically validate the zero-padding behavior.

        Purpose
        -------
        Specifically validate the zero-padding behavior (.zfill(4)) for "from"
        when municipality codes are shorter than 4 characters.

        Steps
        -----
        1) Make classification return a short code "301" (should become "0301")
           plus "9999" (should be filtered).
        2) Call mapping_fra_kommune_til_landet.
        3) Assert "0301" exists and "301" does not.
        """
        FakeKlassClassification_kk_eak.pivot_df = pd.DataFrame(
            {
                "code_1": ["301", "9999"],
                "name_1": ["ShortCode", "Ugyldig"],
            }
        )

        out = mapping_fra_kommune_til_landet(year=2024)

        self.assertTrue(
            (out["from"] == "0301").any(),
            msg="Short kommune code '301' should be zero-padded to '0301'.",
        )
        self.assertFalse(
            (out["from"] == "301").any(),
            msg="Unpadded '301' should not appear in output.",
        )


# =============================================================================
# SECTION 3: mapping_fra_kommune_til_fylkeskommune (KOMMUNER -> FYLKESKOMMUNER)
# =============================================================================
#
# This function (based on test expectations) likely:
#   - calls KlassCorrespondence to obtain sourceCode->targetCode pairs
#   - filters out invalid codes (9999)
#   - renames columns to ["from","to"]
#   - pads both columns to 4 digits
# =============================================================================


class FakeKlassCorrespondence_kk_fk:
    """Minimal fake for KlassCorrespondence(...).data used in mapping_fra_kommune_til_fylkeskommune.

    Provides a mixture of:
      - already-4-digit codes
      - short codes (e.g. "301", "11") to exercise padding
      - invalid "9999" rows to exercise filtering
    """

    def __init__(self, *args, **kwargs):
        self.data = pd.DataFrame(
            {
                "sourceCode": ["0301", "9999", "1103", "301"],
                "targetCode": ["0300", "9999", "1100", "11"],
            }
        )


class TestMappingFraKommuneTilFylkeskommune(unittest.TestCase):
    """Tests for mapping_fra_kommune_til_fylkeskommune(year).

    Main expectations:
      - output columns are exactly ["from","to"]
      - rows with from == "9999" are filtered out
      - both from and to are padded to 4-character strings
      - specific padding transformation examples hold
    """

    @patch(
        "ssb_kostra_python.regionshierarki.KlassCorrespondence",
        new=FakeKlassCorrespondence_kk_fk,
    )
    def test_mapping_filters_renames_and_pads(self):
        """Steps.

        -----
        1) Patch KlassCorrespondence to return deterministic correspondence data.
        2) Call mapping_fra_kommune_til_fylkeskommune("2024").
        3) Assert:
           - schema is ["from","to"]
           - "9999" removed from "from"
           - padding to 4 chars for all entries in both columns
           - spot-check padding behavior with known values
        """
        out = mapping_fra_kommune_til_fylkeskommune("2024")

        self.assertEqual(
            list(out.columns),
            ["from", "to"],
            msg="Output should contain exactly ['from','to'] columns",
        )

        self.assertFalse(
            (out["from"] == "9999").any(),
            msg="Rows with sourceCode '9999' should be filtered out",
        )

        # Assert padding invariants
        self.assertTrue(
            out["from"].map(lambda x: isinstance(x, str) and len(x) == 4).all(),
            msg="'from' values should be 4-character strings",
        )
        self.assertTrue(
            out["to"].map(lambda x: isinstance(x, str) and len(x) == 4).all(),
            msg="'to' values should be 4-character strings",
        )

        # Spot-check expected rows
        self.assertTrue(
            ((out["from"] == "0301") & (out["to"] == "0300")).any(),
            msg="Expected mapping row for 0301 -> 0300 to be present",
        )
        self.assertTrue(
            ((out["from"] == "0301") & (out["to"] == "0011")).any(),
            msg="Expected padding behavior for '301' -> '11' to become '0301' -> '0011'",
        )


# =============================================================================
# SECTION 4: mapping_fra_fylkeskommune_til_kostraregion (FYLKE -> KOSTRA-GRUPPER)
# =============================================================================
#
# This function (as inferred) likely:
#   - uses KlassCorrespondence to map fylkeskommune codes -> KOSTRA-group codes
#   - adds derived rows for:
#       EAFK   (landet)
#       EAFKUO (landet uten Oslo)
#   - excludes some sentinel codes like "9900" from these derived groups
# =============================================================================


class FakeKlassCorrespondence_fk_eafk:
    """Fake KlassCorrespondence that returns deterministic mapping rows for fylkes codes."""

    def __init__(
        self, source_classification_id, target_classification_id, from_date, to_date
    ):
        self.data = pd.DataFrame(
            {
                "sourceCode": ["0300", "4200", "9900"],
                "targetCode": ["R1", "R2", "R9"],
            }
        )


class FakeCodes_fk_eafk:
    def __init__(self, pivot_df: pd.DataFrame):
        self._pivot_df = pivot_df

    def pivot_level(self):
        return self._pivot_df


class FakeKlassClassification_fk_eafk:
    """Fake KlassClassification providing fylkes codes via pivot_level()."""

    pivot_df = None

    def __init__(self, klass_id, language="nb", include_future=True):
        self.klass_id = klass_id

    def get_codes(self, *args, **kwargs):
        return FakeCodes_fk_eafk(self.__class__.pivot_df)


class TestMappingFraFylkeskommuneTilKostraregion(unittest.TestCase):
    """Tests for mapping_fra_fylkeskommune_til_kostraregion(year).

    Expected behavior:
      - include correspondence rows (fylke -> KOSTRA group)
      - add EAFK rows for all fylkeskommune codes except 9900
      - add EAFKUO rows for all fylkeskommune codes except Oslo (0300) and 9900
    """

    @patch(
        "ssb_kostra_python.regionshierarki.KlassCorrespondence",
        new=FakeKlassCorrespondence_fk_eafk,
    )
    @patch(
        "ssb_kostra_python.regionshierarki.KlassClassification",
        new=FakeKlassClassification_fk_eafk,
    )
    def test_mapping_fra_fylkeskommune_til_kostraregion_happy_path(self):
        """Steps.

        -----
        1) Provide fake classification codes including:
           - "0300" (Oslo-like code)
           - "4200" (another valid)
           - "9900" (sentinel code to exclude from derived groups)
        2) Provide fake correspondence rows mapping to R1/R2/R9.
        3) Call mapping_fra_fylkeskommune_til_kostraregion(year="2024")
        4) Assert:
           - schema ["from","to"]
           - correspondence rows exist
           - EAFK rows exist for 0300 and 4200 but not 9900
           - EAFKUO exists only for 4200 (excludes 0300 and 9900)
           - total row count matches expectation
        """
        FakeKlassClassification_fk_eafk.pivot_df = pd.DataFrame(
            {
                "code_1": ["0300", "4200", "9900"],
                "name_1": ["Oslo", "Agder", "Ugyldig"],
            }
        )

        out = mapping_fra_fylkeskommune_til_kostraregion(year="2024")

        self.assertEqual(
            list(out.columns),
            ["from", "to"],
            msg="Output must have columns ['from','to'].",
        )

        # Correspondence rows
        self.assertTrue(((out["from"] == "0300") & (out["to"] == "R1")).any())
        self.assertTrue(((out["from"] == "4200") & (out["to"] == "R2")).any())
        self.assertTrue(((out["from"] == "9900") & (out["to"] == "R9")).any())

        # Derived EAFK rows (excluding 9900)
        self.assertTrue(((out["from"] == "0300") & (out["to"] == "EAFK")).any())
        self.assertTrue(((out["from"] == "4200") & (out["to"] == "EAFK")).any())
        self.assertFalse(((out["from"] == "9900") & (out["to"] == "EAFK")).any())

        # Derived EAFKUO rows (excluding 0300 and 9900)
        self.assertFalse(((out["from"] == "0300") & (out["to"] == "EAFKUO")).any())
        self.assertTrue(((out["from"] == "4200") & (out["to"] == "EAFKUO")).any())
        self.assertFalse(((out["from"] == "9900") & (out["to"] == "EAFKUO")).any())

        # Expected counts: 3 correspondence + 2 EAFK + 1 EAFKUO = 6
        self.assertEqual(
            len(out),
            6,
            msg="Expected 6 rows total: 3 correspondence + 2 EAFK + 1 EAFKUO.",
        )


# =============================================================================
# SECTION 5: hierarki(df, aggregeringstype=...)
# =============================================================================
#
# hierarki appears to be a "main" function that:
#   - validates the input DataFrame (periode uniqueness, region column presence)
#   - chooses which mapping function to use based on which region column is present
#     or based on aggregeringstype
#   - uses definere_klassifikasjonsvariable to decide grouping keys vs statistics columns
#   - appends aggregated rows (mapped region codes) to the output
#   - sometimes renames region column (e.g. kommuneregion -> fylkesregion)
# =============================================================================


class TestHierarki(unittest.TestCase):
    """Tests for the main `hierarki` function.

    Split into:
      - validation tests (it should fail fast on bad inputs)
      - "happy path" tests verifying it:
          * selects correct mapping function
          * aggregates and appends expected rows
          * renames columns when appropriate
    """

    # ---- validation tests ----

    def test_raises_if_more_than_one_periode(self):
        """Hierarki expects exactly one unique periode.

        If more than one periode exists, it should raise.
        """
        df = pd.DataFrame(
            {
                "periode": ["2024", "2025"],
                "kommuneregion": ["0301", "0301"],
                "personer": [1, 2],
            }
        )
        with self.assertRaisesRegex(KeyError, "Mer enn 1 periode"):
            hierarki(df)

    def test_raises_if_no_region_column(self):
        """Hierarki expects exactly one valid region column to exist.

        If none exists, it should raise.
        """
        df = pd.DataFrame({"periode": ["2025"], "personer": [1]})
        with self.assertRaisesRegex(ValueError, "Fant ingen gyldig regionkolonne"):
            hierarki(df)

    def test_raises_if_multiple_region_columns(self):
        """Hierarki expects exactly one region column.

        If multiple region columns are present, it should raise.
        """
        df = pd.DataFrame(
            {
                "periode": ["2025"],
                "kommuneregion": ["0301"],
                "fylkesregion": ["0300"],
                "personer": [1],
            }
        )
        with self.assertRaisesRegex(ValueError, "Fant flere regionskolonner"):
            hierarki(df)

    def test_raises_if_inconsistent_aggregeringstype(self):
        """If the caller specifies an aggregeringstype that doesn't match the actual region column in the DataFrame, hierarki should raise an error."""
        df = pd.DataFrame(
            {
                "periode": ["2025"],
                "fylkesregion": ["0300"],
                "personer": [1],
            }
        )
        with self.assertRaisesRegex(ValueError, "Inkonsekvent valg"):
            hierarki(df, aggregeringstype="kommune_til_landet")

    # ---- happy path: kommune -> landet (default) ----

    @patch(
        "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable",
        autospec=True,
    )
    @patch("ssb_kostra_python.regionshierarki.mapping_fra_kommune_til_landet")
    def test_kommune_til_landet_default_appends_aggregated_rows(
        self, mock_map, mock_definer
    ):
        """Purpose.

        -------
        When df has kommuneregion and aggregeringstype is None, hierarki should:
          - choose mapping_fra_kommune_til_landet
          - use definere_klassifikasjonsvariable to decide grouping keys
          - append aggregated rows where kommuneregion is replaced by mapped 'to' code

        Steps
        -----
        1) Create df with kommuneregion + alder + a stats column (personer).
        2) Patch mapping to return 0301 -> EAK.
        3) Patch definere_klassifikasjonsvariable so grouping includes alder.
        4) Call hierarki(df).
        5) Assert output includes original rows + aggregated rows for each alder.
        """
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "kommuneregion": ["0301", "0301"],
                "alder": ["001", "002"],
                "personer": [10, 20],
            }
        )

        mock_map.return_value = pd.DataFrame({"from": ["0301"], "to": ["EAK"]})
        mock_definer.return_value = (
            ["periode", "kommuneregion", "alder"],
            ["personer"],
        )

        out = hierarki(df)

        # Original 2 rows + 2 aggregated rows (one per alder)
        self.assertEqual(len(out), 4)
        self.assertTrue(
            (
                (out["kommuneregion"] == "EAK")
                & (out["alder"] == "001")
                & (out["personer"] == 10)
            ).any()
        )
        self.assertTrue(
            (
                (out["kommuneregion"] == "EAK")
                & (out["alder"] == "002")
                & (out["personer"] == 20)
            ).any()
        )

        mock_map.assert_called_once()
        mock_definer.assert_called_once()

    # ---- happy path: kommune -> fylkeskommune (override) ----

    @patch(
        "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable",
        autospec=True,
    )
    @patch("ssb_kostra_python.regionshierarki.mapping_fra_kommune_til_fylkeskommune")
    def test_kommune_til_fylkeskommune_filters_to_endswith_00_and_renames(
        self, mock_map, mock_definer
    ):
        """Purpose.

        -------
        If aggregeringstype="kommune_til_fylkeskommune", hierarki should:
          - use mapping_fra_kommune_til_fylkeskommune
          - produce a fylkesregion column (rename kommuneregion -> fylkesregion)
          - filter output to only rows where the resulting region codes end with "00"
            (i.e., fylkes codes)

        Steps
        -----
        1) Create df with kommuneregion.
        2) Patch mapping to produce 0301->0300 and 5001->5000 (end with "00").
        3) Patch classification vars so kommuneregion is used as the region key.
        4) Call hierarki(..., aggregeringstype="kommune_til_fylkeskommune").
        5) Assert kommuneregion is renamed and all remaining rows endwith "00".
        """
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "kommuneregion": ["0301", "5001"],
                "personer": [10, 20],
            }
        )

        mock_map.return_value = pd.DataFrame(
            {
                "from": ["0301", "5001"],
                "to": ["0300", "5000"],
            }
        )
        mock_definer.return_value = (["periode", "kommuneregion"], ["personer"])

        out = hierarki(df, aggregeringstype="kommune_til_fylkeskommune")

        self.assertIn("fylkesregion", out.columns)
        self.assertNotIn("kommuneregion", out.columns)
        self.assertTrue(out["fylkesregion"].str.endswith("00").all())

        mock_map.assert_called_once()
        mock_definer.assert_called_once()

    # ---- happy path: fylkesregion -> kostraregion ----

    @patch(
        "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable",
        autospec=True,
    )
    @patch(
        "ssb_kostra_python.regionshierarki.mapping_fra_fylkeskommune_til_kostraregion"
    )
    def test_fylkeskommune_til_kostraregion(self, mock_map, mock_definer):
        """Purpose.

        -------
        If df contains fylkesregion, hierarki should auto-select the mapping
        fylkeskommune -> kostraregion and apply it.

        Steps
        -----
        1) Create df with fylkesregion and stats column.
        2) Patch mapping to return simple replacements (0300->KFK1, 4200->KFK2).
        3) Patch definere_klassifikasjonsvariable to group by periode+fylkesregion.
        4) Call hierarki(df).
        5) Assert output contains the mapped region codes with same stats values.
        """
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "fylkesregion": ["0300", "4200"],
                "personer": [5, 7],
            }
        )

        mock_map.return_value = pd.DataFrame(
            {"from": ["0300", "4200"], "to": ["KFK1", "KFK2"]}
        )
        mock_definer.return_value = (["periode", "fylkesregion"], ["personer"])

        out = hierarki(df)

        self.assertTrue(
            ((out["fylkesregion"] == "KFK1") & (out["personer"] == 5)).any()
        )
        self.assertTrue(
            ((out["fylkesregion"] == "KFK2") & (out["personer"] == 7)).any()
        )

        mock_map.assert_called_once()
        mock_definer.assert_called_once()

    # ---- happy path: bydelsregion -> EAB ----

    @patch(
        "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable",
        autospec=True,
    )
    @patch("ssb_kostra_python.regionshierarki.mapping_bydeler_oslo")
    def test_bydeler_til_EAB(self, mock_map, mock_definer):
        """Purpose.

        -------
        If df contains bydelsregion, hierarki should auto-select the bydel->EAB mapping
        and aggregate all bydel codes into a single EAB total (per grouping keys).

        Steps
        -----
        1) Create df with bydelsregion and stats column.
        2) Patch mapping to map each bydel code -> EAB.
        3) Patch definere_klassifikasjonsvariable so grouping is periode+bydelsregion.
        4) Call hierarki(df).
        5) Assert a single aggregated EAB row exists and stats are summed.
        """
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "bydelsregion": ["030101", "030102"],
                "personer": [3, 4],
            }
        )

        mock_map.return_value = pd.DataFrame(
            {"from": ["030101", "030102"], "to": ["EAB", "EAB"]}
        )
        mock_definer.return_value = (["periode", "bydelsregion"], ["personer"])

        out = hierarki(df)

        eab_rows = out[out["bydelsregion"] == "EAB"]
        self.assertEqual(
            len(eab_rows),
            1,
            msg="Expected exactly one aggregated EAB row (grouped by periode+bydelsregion).",
        )
        self.assertEqual(
            eab_rows["personer"].iloc[0],
            7,
            msg="Expected persons to be summed across bydel codes into EAB.",
        )

        mock_map.assert_called_once()
        mock_definer.assert_called_once()
