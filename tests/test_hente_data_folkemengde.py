# tests/test_hente_data_folkemengde.py

import unittest
from unittest.mock import patch

import pandas as pd

from ssb_kostra_python.hente_data_folkemengde import hente_data_folkemengde


class TestHenteDataFolkemengde(unittest.TestCase):
    def test_bydel_henter_reelle_data(self) -> None:
        hentet_df = pd.DataFrame(
            {
                "periode": ["2024", "2024"],
                "bydelsregion": ["030101", "EAB"],
                "alder": ["000", "000"],
                "personer": [1000, 1000],
            }
        )

        forventet_resultat = pd.DataFrame(
            {
                "periode": ["2024", "2024"],
                "bydelsregion": ["030101", "EAB"],
                "alder": ["000", "000"],
                "personer": [1000, 1000],
            }
        )

        with (
            patch(
                "ssb_kostra_python.hente_data_folkemengde."
                "hjelpefunksjoner._hent_folkemengde_bydeler_31_12",
                return_value=hentet_df,
            ) as mock_hent_bydel,
            patch(
                "ssb_kostra_python.hente_data_folkemengde.regionshierarki.hierarki",
                return_value=forventet_resultat,
            ) as mock_hierarki,
        ):
            result = hente_data_folkemengde(
                2024,
                "bydel",
                testdata=False,
            )

        mock_hent_bydel.assert_called_once_with(2024)

        forventet_input_til_hierarki = pd.DataFrame(
            {
                "periode": ["2024"],
                "bydelsregion": ["030101"],
                "alder": ["000"],
                "personer": [1000],
            }
        )

        faktisk_input_til_hierarki = mock_hierarki.call_args.args[0]

        pd.testing.assert_frame_equal(
            faktisk_input_til_hierarki.reset_index(drop=True),
            forventet_input_til_hierarki,
        )
        pd.testing.assert_frame_equal(result, forventet_resultat)

    def test_bydel_testdata_bruker_kildeaar_og_anvender_mapping(self) -> None:
        hentet_df = pd.DataFrame(
            {
                "periode": ["2025"],
                "bydelsregion": ["030101"],
                "alder": ["000"],
                "personer": [1000],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": ["030101"],
                "newCode": ["030102"],
            }
        )

        reformert_df = pd.DataFrame(
            {
                "periode": ["2026"],
                "bydelsregion": ["030102"],
                "alder": ["000"],
                "personer": [1000],
            }
        )

        forventet_resultat = reformert_df.copy()

        with (
            patch(
                "ssb_kostra_python.hente_data_folkemengde."
                "hjelpefunksjoner._hent_folkemengde_bydeler_31_12",
                return_value=hentet_df,
            ) as mock_hent_bydel,
            patch(
                "ssb_kostra_python.hente_data_folkemengde."
                "hjelpefunksjoner._mapping_mellom_aar",
                return_value=mapping,
            ) as mock_mapping,
            patch(
                "ssb_kostra_python.hente_data_folkemengde."
                "hjelpefunksjoner._anvende_kommunereform",
                return_value=reformert_df,
            ) as mock_anvend_reform,
            patch(
                "ssb_kostra_python.hente_data_folkemengde.regionshierarki.hierarki",
                return_value=forventet_resultat,
            ) as mock_hierarki,
            patch("ssb_kostra_python.hente_data_folkemengde.display") as mock_display,
        ):
            result = hente_data_folkemengde(
                2026,
                "bydel",
                testdata=True,
            )

        mock_hent_bydel.assert_called_once_with(2025)
        mock_mapping.assert_called_once_with(2026, "bydel")

        mock_anvend_reform.assert_called_once()

        anvend_kall = mock_anvend_reform.call_args.kwargs

        pd.testing.assert_frame_equal(
            anvend_kall["inputfil"],
            hentet_df,
        )
        pd.testing.assert_frame_equal(
            anvend_kall["mapping"],
            mapping,
        )

        self.assertEqual(
            anvend_kall["statistikkvariable"],
            ["personer"],
        )
        self.assertEqual(
            anvend_kall["statistikkaar"],
            2026,
        )
        self.assertEqual(
            anvend_kall["regionsnivaa"],
            "bydel",
        )

        mock_hierarki.assert_called_once()

        pd.testing.assert_frame_equal(
            mock_hierarki.call_args.args[0].reset_index(drop=True),
            reformert_df,
        )
        pd.testing.assert_frame_equal(result, forventet_resultat)

        mock_display.assert_called_once_with(mapping)

    def test_kommune_henter_andre_element_fra_tuple(self) -> None:
        grunnlagsdata = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "alder": ["000"],
                "personer": [700000],
            }
        )

        kostra_aggregert_df = pd.DataFrame(
            {
                "periode": ["2024", "2024"],
                "kommuneregion": ["0301", "EKG01"],
                "alder": ["000", "000"],
                "personer": [700000, 700000],
            }
        )

        forventet_resultat = pd.DataFrame(
            {
                "periode": ["2024", "2024"],
                "kommuneregion": ["0301", "EKG01"],
                "alder": ["000", "000"],
                "personer": [700000, 700000],
            }
        )

        with (
            patch(
                "ssb_kostra_python.hente_data_folkemengde."
                "hjelpefunksjoner._hent_folkemengde_kommune_31_12",
                return_value=(
                    grunnlagsdata,
                    kostra_aggregert_df,
                ),
            ) as mock_hent_kommune,
            patch(
                "ssb_kostra_python.hente_data_folkemengde.regionshierarki.hierarki",
                return_value=forventet_resultat,
            ) as mock_hierarki,
        ):
            result = hente_data_folkemengde(
                2024,
                "kommune",
                testdata=False,
            )

        mock_hent_kommune.assert_called_once_with(2024)

        forventet_input_til_hierarki = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "alder": ["000"],
                "personer": [700000],
            }
        )

        faktisk_input_til_hierarki = mock_hierarki.call_args.args[0]

        pd.testing.assert_frame_equal(
            faktisk_input_til_hierarki.reset_index(drop=True),
            forventet_input_til_hierarki,
        )
        pd.testing.assert_frame_equal(result, forventet_resultat)

    def test_kommune_gir_runtimeerror_nar_aggregert_data_er_none(self) -> None:
        grunnlagsdata = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "alder": ["000"],
                "personer": [700000],
            }
        )

        with patch(
            "ssb_kostra_python.hente_data_folkemengde."
            "hjelpefunksjoner._hent_folkemengde_kommune_31_12",
            return_value=(grunnlagsdata, None),
        ) as mock_hent_kommune:
            with self.assertRaisesRegex(
                RuntimeError,
                "Klarte ikke å lage KOSTRA-aggregert folkemengdefil",
            ):
                hente_data_folkemengde(
                    2024,
                    "kommune",
                    testdata=False,
                )

        mock_hent_kommune.assert_called_once_with(2024)

    def test_fylkeskommune_hentes_og_kostra_grupper_fjernes(self) -> None:
        hentet_df = pd.DataFrame(
            {
                "periode": ["2024", "2024", "2024"],
                "fylkesregion": ["03", "EAFK", "EAFKUO"],
                "alder": ["000", "000", "000"],
                "personer": [1000000, 1000000, 800000],
            }
        )

        forventet_resultat = pd.DataFrame(
            {
                "periode": ["2024", "2024"],
                "fylkesregion": ["03", "EAFK"],
                "alder": ["000", "000"],
                "personer": [1000000, 1000000],
            }
        )

        with (
            patch(
                "ssb_kostra_python.hente_data_folkemengde."
                "hjelpefunksjoner._hent_folkemengde_fylkeskommune_31_12",
                return_value=hentet_df,
            ) as mock_hent_fylke,
            patch(
                "ssb_kostra_python.hente_data_folkemengde.regionshierarki.hierarki",
                return_value=forventet_resultat,
            ) as mock_hierarki,
        ):
            result = hente_data_folkemengde(
                2024,
                "fylkeskommune",
                testdata=False,
            )

        mock_hent_fylke.assert_called_once_with(2024)

        forventet_input_til_hierarki = pd.DataFrame(
            {
                "periode": ["2024"],
                "fylkesregion": ["03"],
                "alder": ["000"],
                "personer": [1000000],
            }
        )

        faktisk_input_til_hierarki = mock_hierarki.call_args.args[0]

        pd.testing.assert_frame_equal(
            faktisk_input_til_hierarki.reset_index(drop=True),
            forventet_input_til_hierarki,
        )
        pd.testing.assert_frame_equal(result, forventet_resultat)

    def test_ugyldig_regionsnivaa_gir_valueerror(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Du må angi regionsnivå",
        ):
            hente_data_folkemengde(
                2024,
                "land",
                testdata=False,
            )

    def test_regionsnivaa_gjores_om_til_sma_bokstaver(self) -> None:
        hentet_df = pd.DataFrame(
            {
                "periode": ["2024"],
                "bydelsregion": ["030101"],
                "alder": ["000"],
                "personer": [1000],
            }
        )

        with (
            patch(
                "ssb_kostra_python.hente_data_folkemengde."
                "hjelpefunksjoner._hent_folkemengde_bydeler_31_12",
                return_value=hentet_df,
            ) as mock_hent_bydel,
            patch(
                "ssb_kostra_python.hente_data_folkemengde.regionshierarki.hierarki",
                return_value=hentet_df,
            ),
        ):
            result = hente_data_folkemengde(
                2024,
                "BYDEL",
                testdata=False,
            )

        mock_hent_bydel.assert_called_once_with(2024)
        pd.testing.assert_frame_equal(result, hentet_df)


if __name__ == "__main__":
    unittest.main()
