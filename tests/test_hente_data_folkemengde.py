# test_hente_data_folkemengde.py

import unittest
from unittest.mock import patch

import pandas as pd

from ssb_kostra_python.hente_data_folkemengde import hente_data_folkemengde


class TestHenteDataFolkemengde(unittest.TestCase):

    @patch(
        "ssb_kostra_python.hente_data_folkemengde.hjelpefunksjoner._hent_folkemengde_bydeler_31_12"
    )
    def test_bydel_henter_reelle_data(self, mock_hent_bydel):
        mock_hent_bydel.return_value = pd.DataFrame(
            {
                "region": ["030101", "030102"],
                "periode": ["2024", "2024"],
                "folkemengde": [1000, 2000],
            }
        )

        result = hente_data_folkemengde(2024, "bydel", testdata=False)

        mock_hent_bydel.assert_called_once_with(2024)
        self.assertTrue((result["periode"] == "2024").all())
        self.assertEqual(len(result), 2)

    @patch(
        "ssb_kostra_python.hente_data_folkemengde.hjelpefunksjoner._hent_folkemengde_bydeler_31_12"
    )
    def test_bydel_testdata_endrer_periode_til_aar(self, mock_hent_bydel):
        mock_hent_bydel.return_value = pd.DataFrame(
            {
                "region": ["030101", "030102"],
                "periode": ["2025", "2025"],
                "folkemengde": [1000, 2000],
            }
        )

        result = hente_data_folkemengde(2026, "bydel", testdata=True)

        mock_hent_bydel.assert_called_once_with(2025)
        self.assertTrue((result["periode"] == "2026").all())

    @patch(
        "ssb_kostra_python.hente_data_folkemengde.hjelpefunksjoner._hent_folkemengde_kommune_31_12"
    )
    def test_kommune_unpacker_tuple(self, mock_hent_kommune):
        kommune_df = pd.DataFrame(
            {
                "region": ["0301"],
                "periode": ["2024"],
                "folkemengde": [700000],
            }
        )

        mock_hent_kommune.return_value = ("metadata", kommune_df)

        result = hente_data_folkemengde(2024, "kommune")

        mock_hent_kommune.assert_called_once_with(2024)
        pd.testing.assert_frame_equal(result, kommune_df)

    @patch(
        "ssb_kostra_python.hente_data_folkemengde.hjelpefunksjoner._hent_folkemengde_fylkeskommune_31_12"
    )
    def test_fylkeskommune_hentes_riktig(self, mock_hent_fylke):
        mock_hent_fylke.return_value = pd.DataFrame(
            {
                "region": ["03"],
                "periode": ["2024"],
                "folkemengde": [1000000],
            }
        )

        result = hente_data_folkemengde(2024, "fylkeskommune")

        mock_hent_fylke.assert_called_once_with(2024)
        self.assertEqual(result.loc[0, "periode"], "2024")

    def test_ugyldig_regionsnivaa_gir_valueerror(self):
        with self.assertRaises(ValueError):
            hente_data_folkemengde(2024, "land")


if __name__ == "__main__":
    unittest.main()
