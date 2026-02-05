import unittest
from unittest.mock import patch
import pandas as pd

from functions.funksjoner.hjelpefunksjoner import definere_klassifikasjonsvariable, format_fil
from functions.funksjoner.enkel_editering import display
from functions.funksjoner.summere_til_aldersgrupperinger import summere_til_aldersgrupperinger


class TestSummereTilAldersgrupperinger(unittest.TestCase):
    """
    Tests for `summere_til_aldersgrupperinger(df_input, hierarki_path=...)`.

    What this function appears to do (inferred from the test)
    ---------------------------------------------------------
    1) Load an "age hierarchy" mapping from a parquet file using pandas.read_parquet.
       The hierarchy contains at least:
         - a key to join on (often "from" representing a specific age code)
         - a target/group code (often "to" representing an age bucket like "000-004")
         - a periode column that must match the input periode type/format

    2) Format the input DataFrame (and/or relevant columns) using `format_fil`
       so that join keys match:
         - 'alder' is expected to be zero-padded (e.g. "1" -> "001")
         - 'periode' is expected to be comparable to the hierarchy periode

    3) Merge input data with the hierarchy so each input row gets an age-group label.

    4) Use `definere_klassifikasjonsvariable` to decide:
         - classification variables (groupby keys)
         - statistics variables (columns to sum)
       and then aggregate the stats into age groups.

    5) Rename the final age-group column so the output uses 'alder' as the age-group label
       (the test comments suggest 'to' is eventually renamed to 'alder').

    Outputs (based on this test)
    ----------------------------
    The function returns a tuple:
      (rename_variabel, groupby_variable, df_out)

    where:
      - rename_variabel is expected to be ["alder"]
      - groupby_variable is the list of grouping keys used
      - df_out is the aggregated DataFrame (should include aggregated rows by age groups)

    Why we patch things
    -------------------
    - pandas.read_parquet: to avoid filesystem I/O and supply a deterministic hierarchy.
    - format_fil: to avoid running real formatting logic and instead enforce the minimal
      formatting required for the merge to work in the test.
    - definere_klassifikasjonsvariable: to fully control grouping keys and stats columns
      (so the aggregation behavior is deterministic and easy to assert).
    - display: because the function likely displays intermediate results in notebooks;
      tests should not produce UI output.
    """

    @patch("functions.funksjoner.enkel_editering.display")
    @patch("functions.funksjoner.hjelpefunksjoner.definere_klassifikasjonsvariable")
    @patch("functions.funksjoner.hjelpefunksjoner.format_fil")
    @patch("pandas.read_parquet")
    def test_summere_til_aldersgrupperinger_basic(
        self,
        mock_read_parquet,
        mock_format_fil,
        mock_definer_klass,
        mock_display,
    ):
        """
        Purpose
        -------
        Validate a basic end-to-end flow of summing/aggregating rows into age groups.

        Specifically, this test verifies:
          - Input ages are formatted/padded so they can join the hierarchy
          - Hierarchy mapping is applied (alder -> age bucket)
          - Aggregation sums the statistikksvariable into the bucket
          - The function returns the expected metadata (rename_variabel, groupby_variable)
          - The helper functions are called

        Test strategy
        -------------
        Because the real function depends on external files and helper logic,
        we patch dependencies to make the pipeline deterministic:
          1) read_parquet returns a tiny hierarchy DataFrame
          2) format_fil is replaced with a minimal deterministic implementation
          3) definere_klassifikasjonsvariable returns controlled groupby/stat columns
        """

        # ---------------------------------------------------------------------
        # 1) Arrange: input data (deliberately unpadded alder)
        # ---------------------------------------------------------------------
        df_input = pd.DataFrame({
            "periode": ["2025", "2025", "2025"],
            "alder": ["1", "2", "10"],  # intentionally not padded (we expect formatting to pad)
            "kommuneregion": ["0301", "0301", "0301"],
            "personer": [10, 20, 5],
        })

        # ---------------------------------------------------------------------
        # 2) Arrange: hierarchy data returned from read_parquet
        # ---------------------------------------------------------------------
        # Notes:
        # - The function under test likely stringifies + zero-pads 'from' to match 'alder'
        # - Here we deliberately provide from values as ints so we exercise that behavior.
        # - 'to' is the age bucket we expect to aggregate into.
        df_hierarki = pd.DataFrame({
            "periode": ["2025", "2025"],
            "from": [1, 2],             # int -> expected to become "001", "002" for matching
            "to": ["000-004", "000-004"],
        })
        mock_read_parquet.return_value = df_hierarki

        # ---------------------------------------------------------------------
        # 3) Arrange: patch format_fil with a minimal implementation needed for join
        # ---------------------------------------------------------------------
        # We enforce:
        # - periode is string (to match df_hierarki["periode"])
        # - alder is string and padded to 3 digits (so "1" matches "001")
        def fake_format_fil(df):
            df = df.copy()
            df["periode"] = df["periode"].astype(str)
            df["alder"] = df["alder"].astype(str).str.zfill(3)
            return df

        mock_format_fil.side_effect = fake_format_fil

        # ---------------------------------------------------------------------
        # 4) Arrange: patch definere_klassifikasjonsvariable (controls groupby + stats)
        # ---------------------------------------------------------------------
        # IMPORTANT subtlety:
        # The merged df will contain 'to', and later the function renames 'to' -> 'alder'.
        # Therefore, to keep that column during groupby aggregation, 'to' must be included
        # as a classification variable returned by the helper.
        klass_vars = ["periode", "kommuneregion", "to"]
        stat_vars = ["personer"]
        mock_definer_klass.return_value = (klass_vars, stat_vars)

        # ---------------------------------------------------------------------
        # 5) Act: run the function under test
        # ---------------------------------------------------------------------
        rename_variabel, groupby_variable, df_out = summere_til_aldersgrupperinger(
            df_input,
            hierarki_path="dummy/path.parquet",
        )

        # ---------------------------------------------------------------------
        # 6) Assert: returned metadata is as expected
        # ---------------------------------------------------------------------
        self.assertEqual(rename_variabel, ["alder"])

        # groupby_variable should be klass_vars minus the "rename variable" if applicable.
        # In this scenario, the code groups by ["periode", "kommuneregion", "to"] and then
        # renames 'to' to 'alder' later, so groupby_variable should still list 'to'.
        self.assertEqual(
            sorted(groupby_variable),
            sorted(["periode", "kommuneregion", "to"])
        )

        # ---------------------------------------------------------------------
        # 7) Assert: output contains the aggregated age bucket row
        # ---------------------------------------------------------------------
        # If this fails, it often means:
        # - formatting didn't align keys (merge produced no rows)
        # - 'to' was not treated as a group key and got dropped before aggregation
        self.assertTrue(
            (df_out["alder"] == "000-004").any(),
            msg=(
                "Expected aggregated row not present. "
                "This usually means merge produced zero rows or 'to' got lost before groupby."
            )
        )

        # Only ages 1 and 2 map into 000-004 in our hierarchy, so sum should be 10 + 20 = 30.
        aggregated = df_out[df_out["alder"] == "000-004"]
        self.assertEqual(
            aggregated["personer"].iloc[0],
            30,
            msg="10 + 20 should aggregate into 000-004"
        )

        # ---------------------------------------------------------------------
        # 8) Assert: dependency calls happened (sanity that pipeline ran)
        # ---------------------------------------------------------------------
        mock_read_parquet.assert_called_once()
        mock_format_fil.assert_called_once()
        mock_definer_klass.assert_called_once()

