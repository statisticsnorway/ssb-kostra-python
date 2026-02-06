import unittest
from unittest.mock import patch

import pandas as pd
from functions.funksjoner.summere_kjonn import summere_over_kjonn


class TestSummereOverKjonn(unittest.TestCase):
    """Tests for `summere_over_kjonn(df)`.

    What the function appears to do (inferred from tests)
    -----------------------------------------------------
    - If the column 'kjonn' is NOT present:
        return the input unchanged (no aggregation).

    - If 'kjonn' IS present:
        1) Call definere_klassifikasjonsvariable(df) to determine:
           - klassifikasjonsvariable: columns that define the grouping keys
           - statistikkvariable: numeric/stat columns that should be summed
        2) Sum (aggregate) all statistikkvariable over 'kjonn' by removing 'kjonn'
           from the grouping keys, and grouping by the remaining classification vars.
        3) Return a new aggregated DataFrame where the output does not contain 'kjonn'.


    Note on patching definere_klassifikasjonsvariable
    -------------------------------------------------
    The helper function defines grouping and stats columns. In unit tests we patch it
    so:
      - no user interaction happens (if any)
      - the behavior is deterministic and easy to reason about
    """

    def test_returns_input_unchanged_when_kjonn_missing(self):
        """Purpose.

        -------
        Verify that if the DataFrame lacks the 'kjonn' column, the function
        does not attempt aggregation and returns the original content unchanged.

        Steps
        -----
        1) Create a DataFrame without 'kjonn'.
        2) Call summere_over_kjonn(df).
        3) Assert output equals input (same values).
           (Your function returns the input directly in this case, so equality is
            the key property; identity is not strictly required.)
        """
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "kommuneregion": ["0301", "0301"],
                "personer": [10, 20],
            }
        )

        out = summere_over_kjonn(df)

        # Same object is fine here if your implementation returns inputfil directly.
        self.assertTrue(
            out.equals(df),
            msg="If 'kjonn' is missing, function should return the input unchanged.",
        )

    @patch("functions.funksjoner.hjelpefunksjoner.definere_klassifikasjonsvariable")
    def test_sums_over_kjonn_when_present(self, mock_definer_klass):
        """Purpose.

        -------
        Verify that when 'kjonn' exists, the function sums all statistikkvariable
        over kjonn (i.e., removes kjonn as a grouping key) and returns grouped rows.

        What we control via patching
        ----------------------------
        We patch definere_klassifikasjonsvariable to return:
          - classification vars: ["periode", "kommuneregion", "kjonn"]
          - stats vars: ["personer"]

        That means the function should:
          - group by ["periode", "kommuneregion"] (classification vars except kjonn)
          - sum "personer" across kjonn values

        Steps
        -----
        1) Create df containing repeated period+region but different kjonn values.
        2) Patch helper to force grouping rules.
        3) Call summere_over_kjonn(df).
        4) Assert:
           - exactly 1 aggregated row
           - summed value is correct
           - 'kjonn' column is removed from output
        """
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025", "2025"],
                "kommuneregion": ["0301", "0301", "0301"],
                "kjonn": ["1", "2", "1"],
                "personer": [10, 20, 5],
            }
        )

        # Control what the helper returns:
        # - 'kjonn' is part of classification vars (so it will be removed for summing)
        # - stats var is "personer"
        mock_definer_klass.return_value = (
            ["periode", "kommuneregion", "kjonn"],
            ["personer"],
        )

        out = summere_over_kjonn(df)

        # After summing over kjonn, the grouping keys should be:
        #   periode + kommuneregion
        # so only one row remains, and personer should be summed across all rows:
        #   10 + 20 + 5 = 35
        self.assertEqual(
            len(out), 1, msg="Expected one aggregated row after summing over kjonn."
        )
        self.assertEqual(out["periode"].iloc[0], "2025")
        self.assertEqual(out["kommuneregion"].iloc[0], "0301")
        self.assertEqual(
            out["personer"].iloc[0],
            35,
            msg="Expected personer to be summed across kjonn.",
        )

        # kjonn should not be in output columns (since it's summed over / removed as key)
        self.assertNotIn(
            "kjonn",
            out.columns,
            msg="'kjonn' should be removed from the grouping keys after summing over it.",
        )

        mock_definer_klass.assert_called_once()

    @patch("functions.funksjoner.hjelpefunksjoner.definere_klassifikasjonsvariable")
    def test_sums_multiple_stat_columns(self, mock_definer_klass):
        """Purpose.

        -------
        Verify that if the helper returns multiple statistikkvariable, the function
        sums ALL of them when aggregating over kjonn.

        Steps
        -----
        1) Create df with two stat columns: personer and inntekt.
        2) Patch helper to return:
           - classification vars: ["periode", "kommuneregion", "kjonn"]
           - stats vars: ["personer", "inntekt"]
        3) Call summere_over_kjonn(df).
        4) Assert:
           - one aggregated row remains
           - both personer and inntekt are summed correctly
           - 'kjonn' is not present in output
        """
        df = pd.DataFrame(
            {
                "periode": ["2025", "2025"],
                "kommuneregion": ["0301", "0301"],
                "kjonn": ["1", "2"],
                "personer": [10, 20],
                "inntekt": [100, 200],
            }
        )

        mock_definer_klass.return_value = (
            ["periode", "kommuneregion", "kjonn"],
            ["personer", "inntekt"],
        )

        out = summere_over_kjonn(df)

        self.assertEqual(len(out), 1)
        self.assertEqual(out["personer"].iloc[0], 30)
        self.assertEqual(out["inntekt"].iloc[0], 300)
        self.assertNotIn("kjonn", out.columns)

        mock_definer_klass.assert_called_once()
