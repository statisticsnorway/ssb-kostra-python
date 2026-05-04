# import pytest
import unittest
from unittest.mock import patch

import pandas as pd

from ssb_kostra_python import hente_lage_folkemengdefil_31_12

# from ssb_kostra_python.validering import _klass_check
# from ssb_kostra_python.validering import _missing_cols
# from ssb_kostra_python.validering import _missing_values
# from ssb_kostra_python.validering import _number_of_periods_in_df
# from ssb_kostra_python.validering import _valid_periode_region

MODULE = "ssb_kostra_python.hente_lage_folkemengdefil_31_12"

## Bydeler


class TestHentFolkemengdeBydeler3112(unittest.TestCase):

    @patch(f"{MODULE}.regionshierarki.hierarki")
    @patch(f"{MODULE}.summere_kjonn.summere_over_kjonn")
    @patch(f"{MODULE}.summere_til_aldersgrupperinger.summere_til_aldersgrupperinger")
    @patch(f"{MODULE}.pd.read_parquet")
    @patch(f"{MODULE}.latest_version_path")
    def test_hent_folkemengde_bydeler_31_12_success(
        self,
        mock_latest_version_path,
        mock_read_parquet,
        mock_summere_til_aldersgrupperinger,
        mock_summere_over_kjonn,
        mock_regionshierarki,
    ):
        # Arrange
        statistikkaar = 2024
        mock_latest_version_path.return_value = "/fake/path/folkemengde.parquet"

        input_df = pd.DataFrame(
            {
                "region": ["0301"],
                "kjonn": ["1"],
                "alder": ["100"],
                "value": [10],
            }
        )

        df_sum_med_kjonn = pd.DataFrame(
            {
                "region": ["0301"],
                "kjonn": ["1"],
                "alder": ["100"],
                "value": [10],
            }
        )

        df_sum_kjonn = pd.DataFrame(
            {
                "region": ["0301"],
                "alder": ["100"],
                "value": [10],
            }
        )

        df_region = pd.DataFrame(
            {
                "region": ["030101", "030102", "030103"],
                "alder": ["100", "105", "120"],
                "value": [10, 20, 30],
            }
        )

        mock_read_parquet.return_value = input_df

        mock_summere_til_aldersgrupperinger.return_value = (
            "alder",
            ["kjonn", "alder"],
            df_sum_med_kjonn,
        )

        mock_summere_over_kjonn.return_value = df_sum_kjonn
        mock_regionshierarki.return_value = df_region

        # Act
        result = hente_lage_folkemengdefil_31_12.hent_folkemengde_bydeler_31_12(
            statistikkaar
        )

        # Assert
        expected = pd.DataFrame(
            {
                "region": ["030101"],
                "alder": ["100"],
                "value": [10],
            }
        )

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

        mock_latest_version_path.assert_called_once_with(
            "/buckets/delt-kostra-befolkning-delt/bydeler/2024/"
            "folkmengde_bydeler_p2024-12-31"
        )

        mock_read_parquet.assert_called_once_with("/fake/path/folkemengde.parquet")

        mock_summere_til_aldersgrupperinger.assert_called_once()
        mock_summere_over_kjonn.assert_called_once_with(df_sum_med_kjonn)
        mock_regionshierarki.assert_called_once_with(df_sum_kjonn)


if __name__ == "__main__":
    unittest.main()


## Kommuner


class TestHentFolkemengdeKommune3112(unittest.TestCase):

    @patch(f"{MODULE}.regionshierarki.hierarki")
    @patch(f"{MODULE}.summere_kjonn.summere_over_kjonn")
    @patch(f"{MODULE}.summere_til_aldersgrupperinger.summere_til_aldersgrupperinger")
    @patch(f"{MODULE}.duckdb.query")
    @patch(f"{MODULE}.latest_version_path")
    def test_hent_folkemengde_kommune_31_12_success(
        self,
        mock_latest_version_path,
        mock_duckdb_query,
        mock_summere_til_aldersgrupperinger,
        mock_summere_over_kjonn,
        mock_regionshierarki,
    ):
        # Arrange
        mock_latest_version_path.side_effect = [
            "/fake/path/kommuner.parquet",
            "/fake/path/svalbard.parquet",
        ]

        df_kommuner = pd.DataFrame(
            {
                "kjonn": ["1", "1"],
                "kommuneregion": ["0301", "0301"],
                "alder": ["100", "100"],
            }
        )

        df_svalbard = pd.DataFrame(
            {
                "kjonn": ["2"],
                "alder": ["100"],
            }
        )

        mock_duckdb_query.side_effect = [
            unittest.mock.Mock(to_df=unittest.mock.Mock(return_value=df_kommuner)),
            unittest.mock.Mock(to_df=unittest.mock.Mock(return_value=df_svalbard)),
        ]

        df_agg_alder = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "kjonn": ["1"],
                "alder": ["100"],
                "personer": [2],
            }
        )

        df_agg_kjonn = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "alder": ["100"],
                "personer": [2],
            }
        )

        df_kostra = pd.DataFrame(
            {
                "periode": ["2024", "2024", "2024"],
                "kommuneregion": ["0301", "0301", "0301"],
                "alder": ["100", "105", "120"],
                "personer": [2, 3, 4],
            }
        )

        mock_summere_til_aldersgrupperinger.return_value = (
            "alder",
            ["kjonn", "alder"],
            df_agg_alder,
        )
        mock_summere_over_kjonn.return_value = df_agg_kjonn
        mock_regionshierarki.return_value = df_kostra

        # Act
        df_base, df_final = (
            hente_lage_folkemengdefil_31_12.hent_folkemengde_kommune_31_12(2024)
        )

        # Assert base data
        expected_base = pd.DataFrame(
            {
                "periode": ["2024", "2024"],
                "kommuneregion": ["0301", "2111"],
                "kjonn": ["1", "2"],
                "alder": ["100", "100"],
                "personer": [2, 1],
            }
        )

        pd.testing.assert_frame_equal(
            df_base.sort_values(
                ["periode", "kommuneregion", "kjonn", "alder"]
            ).reset_index(drop=True),
            expected_base.sort_values(
                ["periode", "kommuneregion", "kjonn", "alder"]
            ).reset_index(drop=True),
        )

        # Assert final data: alder 105 and 120 removed
        expected_final = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "alder": ["100"],
                "personer": [2],
            }
        )

        pd.testing.assert_frame_equal(
            df_final.reset_index(drop=True),
            expected_final.reset_index(drop=True),
        )

        self.assertEqual(mock_latest_version_path.call_count, 2)
        self.assertEqual(mock_duckdb_query.call_count, 2)

        mock_summere_til_aldersgrupperinger.assert_called_once()
        mock_summere_over_kjonn.assert_called_once_with(df_agg_alder)
        mock_regionshierarki.assert_called_once_with(df_agg_kjonn)

    @patch(f"{MODULE}.regionshierarki.hierarki")
    @patch(f"{MODULE}.summere_kjonn.summere_over_kjonn")
    @patch(f"{MODULE}.summere_til_aldersgrupperinger.summere_til_aldersgrupperinger")
    @patch(f"{MODULE}.duckdb.query")
    @patch(f"{MODULE}.latest_version_path")
    def test_hent_folkemengde_kommune_31_12_only_kommuner_available(
        self,
        mock_latest_version_path,
        mock_duckdb_query,
        mock_summere_til_aldersgrupperinger,
        mock_summere_over_kjonn,
        mock_regionshierarki,
    ):
        # Kommune path succeeds, Svalbard path fails
        mock_latest_version_path.side_effect = [
            "/fake/path/kommuner.parquet",
            FileNotFoundError("Svalbard missing"),
        ]

        df_kommuner = pd.DataFrame(
            {
                "kjonn": ["1"],
                "kommuneregion": ["0301"],
                "alder": ["100"],
            }
        )

        mock_duckdb_query.return_value.to_df.return_value = df_kommuner

        df_agg_alder = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "kjonn": ["1"],
                "alder": ["100"],
                "personer": [1],
            }
        )

        df_agg_kjonn = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "alder": ["100"],
                "personer": [1],
            }
        )

        df_kostra = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "alder": ["100"],
                "personer": [1],
            }
        )

        mock_summere_til_aldersgrupperinger.return_value = (
            "alder",
            ["kjonn", "alder"],
            df_agg_alder,
        )
        mock_summere_over_kjonn.return_value = df_agg_kjonn
        mock_regionshierarki.return_value = df_kostra

        # Act
        with self.assertWarns(Warning):
            df_base, df_final = (
                hente_lage_folkemengdefil_31_12.hent_folkemengde_kommune_31_12(2024)
            )

        expected_base = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "kjonn": ["1"],
                "alder": ["100"],
                "personer": [1],
            }
        )

        pd.testing.assert_frame_equal(
            df_base.reset_index(drop=True),
            expected_base.reset_index(drop=True),
        )

        pd.testing.assert_frame_equal(
            df_final.reset_index(drop=True),
            df_kostra.reset_index(drop=True),
        )

    @patch(f"{MODULE}.latest_version_path")
    def test_hent_folkemengde_kommune_31_12_no_sources_available(
        self,
        mock_latest_version_path,
    ):
        mock_latest_version_path.side_effect = FileNotFoundError("missing file")

        with self.assertRaises(FileNotFoundError):
            (hente_lage_folkemengdefil_31_12.hent_folkemengde_kommune_31_12(2024))


class TestHentFolkemengde3112Fk(unittest.TestCase):

    @patch(f"{MODULE}.regionshierarki.hierarki")
    @patch(f"{MODULE}.summere_kjonn.summere_over_kjonn")
    @patch(f"{MODULE}.summere_til_aldersgrupperinger.summere_til_aldersgrupperinger")
    @patch(f"{MODULE}.hent_folkemengde_kommune_31_12")
    def test_hent_folkemengde_31_12_fk_success(
        self,
        mock_hent_kommune,
        mock_summere_til_aldersgrupperinger,
        mock_summere_over_kjonn,
        mock_hierarki,
    ):
        # Arrange
        df_kommune = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "kjonn": ["1"],
                "alder": ["100"],
                "personer": [10],
            }
        )

        df_sum_med_kjonn = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "kjonn": ["1"],
                "alder": ["100"],
                "personer": [10],
            }
        )

        df_sum_kjonn = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "alder": ["100"],
                "personer": [10],
            }
        )

        df_fk = pd.DataFrame(
            {
                "periode": ["2024"],
                "fylkesregion": ["03"],
                "alder": ["100"],
                "personer": [10],
            }
        )

        df_eafk = pd.DataFrame(
            {
                "periode": ["2024", "2024", "2024"],
                "fylkesregion": ["EAFK", "EAFK", "EAFK"],
                "alder": ["100", "105", "F025-029"],
                "personer": [10, 20, 30],
            }
        )

        mock_hent_kommune.return_value = (df_kommune, None)

        mock_summere_til_aldersgrupperinger.return_value = (
            "alder",
            ["kjonn", "alder"],
            df_sum_med_kjonn,
        )

        mock_summere_over_kjonn.return_value = df_sum_kjonn

        mock_hierarki.side_effect = [
            df_fk,
            df_eafk,
        ]

        # Act
        result = hente_lage_folkemengdefil_31_12.hent_folkemengde_31_12_fk(2024)

        # Assert
        expected = pd.DataFrame(
            {
                "periode": ["2024"],
                "fylkesregion": ["EAFK"],
                "alder": ["100"],
                "personer": [10],
            }
        )

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

        mock_hent_kommune.assert_called_once_with(2024)

        mock_summere_til_aldersgrupperinger.assert_called_once()
        mock_summere_over_kjonn.assert_called_once_with(df_sum_med_kjonn)

        self.assertEqual(mock_hierarki.call_count, 2)

        from unittest.mock import call
        
        # Extract calls
        calls = mock_hierarki.call_args_list
        
        # First call assertions
        args, kwargs = calls[0]
        pd.testing.assert_frame_equal(args[0], df_sum_kjonn)
        self.assertEqual(kwargs, {"aggregeringstype": "kommune_til_fylkeskommune"})
        
        # Second call assertions
        args, kwargs = calls[1]
        pd.testing.assert_frame_equal(args[0], df_fk)
        self.assertEqual(kwargs, {})

    @patch(f"{MODULE}.hent_folkemengde_kommune_31_12")
    def test_hent_folkemengde_31_12_fk_raises_when_kommune_fails(
        self,
        mock_hent_kommune,
    ):
        mock_hent_kommune.side_effect = FileNotFoundError("missing kommune data")

        with self.assertRaises(RuntimeError):
            (hente_lage_folkemengdefil_31_12.hent_folkemengde_31_12_fk(2024))
