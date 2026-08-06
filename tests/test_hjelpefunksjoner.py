from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ssb_kostra_python import hjelpefunksjoner
from ssb_kostra_python.hjelpefunksjoner import _konvertere_komma_til_punktdesimal
from ssb_kostra_python.hjelpefunksjoner import definere_klassifikasjonsvariable
from ssb_kostra_python.hjelpefunksjoner import format_fil


class TestFormatFil:
    def test_formats_periode_and_alder_fixed_width(self) -> None:
        df = pd.DataFrame(
            {
                "periode": [1, "23", "2025"],
                "alder": [7, "45", "123"],
                "kommuneregion": ["301", "0301", "9999"],
            }
        )

        out = format_fil(df.copy())

        assert out["periode"].tolist() == ["0001", "0023", "2025"]
        assert out["alder"].tolist() == ["007", "045", "123"]
        assert pd.api.types.is_string_dtype(out["periode"])
        assert pd.api.types.is_string_dtype(out["alder"])

    def test_kommuneregion_pads_only_digits_and_only_when_too_short(self) -> None:
        df = pd.DataFrame(
            {
                "periode": ["1", "1", "1", "1", "1"],
                "kommuneregion": ["301", "0301", "12A", "12345", None],
            }
        )

        out = format_fil(df.copy())

        assert out["kommuneregion"].tolist() == [
            "0301",
            "0301",
            "12A",
            "12345",
            pd.NA,
        ]
        assert pd.api.types.is_string_dtype(out["kommuneregion"])

    def test_fylkesregion_pads_only_digits_and_only_when_too_short(self) -> None:
        df = pd.DataFrame({"fylkesregion": ["3", "03", "0301", "AB", ""]})

        out = format_fil(df.copy())

        assert out["fylkesregion"].tolist() == ["0003", "0003", "0301", "AB", ""]

    def test_bydelsregion_pads_only_digits_and_only_when_too_short(self) -> None:
        df = pd.DataFrame({"bydelsregion": ["301", "030101", "12A", "1234567"]})

        out = format_fil(df.copy())

        assert out["bydelsregion"].tolist() == ["000301", "030101", "12A", "1234567"]


class TestDefinereKlassifikasjonsvariable:
    def test_no_additional_variables(self, mocker: Any) -> None:
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
        assert pd.api.types.is_string_dtype(df["periode"])
        assert pd.api.types.is_string_dtype(df["kommuneregion"])

    def test_additional_variables_parsing_dedup_and_order(self, mocker: Any) -> None:
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
        assert pd.api.types.is_string_dtype(df["kjonn"])
        assert pd.api.types.is_string_dtype(df["alder"])

    def test_fixed_vars_only_included_if_present(self, mocker: Any) -> None:
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
    def test_converts_comma_decimal_to_float(self) -> None:
        df = pd.DataFrame({"a": ["1,5", "2,0", "3,25"]})

        out = _konvertere_komma_til_punktdesimal(df)

        assert np.allclose(out["a"].to_numpy(), [1.5, 2.0, 3.25])
        assert pd.api.types.is_float_dtype(out["a"])

    def test_leaves_columns_without_commas_unchanged(self) -> None:
        df = pd.DataFrame(
            {
                "a": ["1,5", "2,0"],
                "b": ["x", "y"],
                "c": [10, 20],
            }
        )

        out = _konvertere_komma_til_punktdesimal(df)

        assert out["b"].tolist() == ["x", "y"]
        assert out["c"].tolist() == [10, 20]

    def test_converts_column_if_any_value_contains_comma(self) -> None:
        df = pd.DataFrame({"a": ["1,5", "2"]})

        out = _konvertere_komma_til_punktdesimal(df)

        assert np.allclose(out["a"].to_numpy(), [1.5, 2.0])
        assert pd.api.types.is_float_dtype(out["a"])

    def test_does_not_modify_input_dataframe(self) -> None:
        df = pd.DataFrame({"a": ["1,5", "2,0"]})
        df_before = df.copy(deep=True)

        _ = _konvertere_komma_til_punktdesimal(df)

        pd.testing.assert_frame_equal(df, df_before)


class TestHentFolkemengdeBydeler3112:
    def test_hent_folkemengde_bydeler_success(self, mocker: Any) -> None:
        mock_latest_version_path = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.latest_version_path",
            return_value="/fake/path/data.parquet",
        )
        mock_read_parquet = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.pd.read_parquet"
        )
        mock_summere_aldersgrupper = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner."
            "summere_til_aldersgrupperinger.summere_til_aldersgrupperinger"
        )
        mock_summere_kjonn = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.summere_kjonn.summere_over_kjonn"
        )
        mock_regionshierarki = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.regionshierarki.hierarki"
        )

        raw_df = pd.DataFrame(
            {
                "periode": ["2024"],
                "bydelsregion": ["030101"],
                "alder": ["0"],
                "kjonn": ["1"],
                "personer": [10],
            }
        )

        df_sum_med_kjonn = pd.DataFrame(
            {
                "periode": ["2024"],
                "bydelsregion": ["030101"],
                "alder": ["0-5"],
                "kjonn": ["1"],
                "personer": [10],
            }
        )

        df_sum_kjonn = pd.DataFrame(
            {
                "periode": ["2024"],
                "bydelsregion": ["030101"],
                "alder": ["0-5"],
                "personer": [10],
            }
        )

        df_hierarki = pd.DataFrame(
            {
                "periode": ["2024", "2024"],
                "bydelsregion": ["030101", "030101"],
                "alder": ["0-5", "105"],
                "personer": [10, 1],
                "ekstra_kolonne": ["x", "y"],
            }
        )

        mock_read_parquet.return_value = raw_df
        mock_summere_aldersgrupper.return_value = df_sum_med_kjonn
        mock_summere_kjonn.return_value = df_sum_kjonn
        mock_regionshierarki.return_value = df_hierarki

        result = hjelpefunksjoner._hent_folkemengde_bydeler_31_12(2024)

        mock_latest_version_path.assert_called_once_with(
            "/buckets/delt-kostra-befolkning-delt/bydeler/2024/"
            "folkmengde_bydeler_p2024-12-31"
        )
        mock_read_parquet.assert_called_once_with("/fake/path/data.parquet")
        mock_summere_aldersgrupper.assert_called_once()
        mock_summere_kjonn.assert_called_once_with(df_sum_med_kjonn)
        mock_regionshierarki.assert_called_once_with(df_sum_kjonn)

        expected = pd.DataFrame(
            {
                "periode": ["2024"],
                "bydelsregion": ["030101"],
                "alder": ["0-5"],
                "personer": [10],
            }
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_feil_i_latest_version_path_gir_runtimeerror(self, mocker: Any) -> None:
        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.latest_version_path",
            side_effect=Exception("bucket ikke funnet"),
        )

        with pytest.raises(
            RuntimeError, match="latest_version_path for bydeldata feilet"
        ):
            hjelpefunksjoner._hent_folkemengde_bydeler_31_12(2024)

    def test_feil_i_read_parquet_gir_runtimeerror(self, mocker: Any) -> None:
        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.latest_version_path",
            return_value="/fake/path/data.parquet",
        )
        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.pd.read_parquet",
            side_effect=Exception("kan ikke lese parquet"),
        )

        with pytest.raises(
            RuntimeError,
            match="innlesing av folkemengdefil for bydeler feilet",
        ):
            hjelpefunksjoner._hent_folkemengde_bydeler_31_12(2024)

    def test_filtrerer_bort_aldre_105_til_120(self, mocker: Any) -> None:
        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.latest_version_path",
            return_value="/fake/path/data.parquet",
        )
        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.pd.read_parquet",
            return_value=pd.DataFrame(),
        )
        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner."
            "summere_til_aldersgrupperinger.summere_til_aldersgrupperinger",
            return_value=pd.DataFrame(),
        )
        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.summere_kjonn.summere_over_kjonn",
            return_value=pd.DataFrame(),
        )
        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.regionshierarki.hierarki",
            return_value=pd.DataFrame(
                {
                    "periode": ["2024", "2024", "2024"],
                    "bydelsregion": ["030101", "030101", "030101"],
                    "alder": ["104", "105", "120"],
                    "personer": [100, 10, 1],
                }
            ),
        )

        result = hjelpefunksjoner._hent_folkemengde_bydeler_31_12("2024")

        assert result["alder"].tolist() == ["104"]


class TestHentFolkemengdeKommune3112:
    def test_combines_kommune_and_svalbard_and_runs_pipeline(self, mocker: Any) -> None:
        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.latest_version_path",
            side_effect=["/kommune.parquet", "/svalbard.parquet"],
        )

        df_kommune = pd.DataFrame(
            {
                "kjonn": ["1", "1"],
                "kommuneregion": ["0301", "0301"],
                "alder": ["10", "10"],
            }
        )

        df_svalbard = pd.DataFrame(
            {
                "kjonn": ["2"],
                "alder": ["20"],
            }
        )

        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.duckdb.query",
            side_effect=[
                MagicMock(to_df=lambda: df_kommune),
                MagicMock(to_df=lambda: df_svalbard),
            ],
        )

        df_after_age = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "alder": ["000-015"],
                "kjonn": ["1"],
                "personer": [2],
            }
        )

        df_after_kjonn = pd.DataFrame(
            {
                "periode": ["2024"],
                "kommuneregion": ["0301"],
                "alder": ["000-015"],
                "personer": [2],
            }
        )

        df_after_hierarki = pd.DataFrame(
            {
                "periode": ["2024", "2024"],
                "kommuneregion": ["0301", "0301"],
                "alder": ["000-015", "105"],
                "personer": [2, 99],
            }
        )

        mock_sum_age = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner."
            "summere_til_aldersgrupperinger.summere_til_aldersgrupperinger",
            return_value=df_after_age,
        )
        mock_sum_kjonn = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.summere_kjonn.summere_over_kjonn",
            return_value=df_after_kjonn,
        )
        mock_hierarki = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.regionshierarki.hierarki",
            return_value=df_after_hierarki,
        )

        base_df, final_df = hjelpefunksjoner._hent_folkemengde_kommune_31_12(2024)

        expected_base = pd.DataFrame(
            {
                "periode": ["2024", "2024"],
                "kommuneregion": ["0301", "2111"],
                "kjonn": ["1", "2"],
                "alder": ["10", "20"],
                "personer": [2, 1],
            }
        )

        pd.testing.assert_frame_equal(
            base_df.sort_values(["kommuneregion", "alder"]).reset_index(drop=True),
            expected_base.sort_values(["kommuneregion", "alder"]).reset_index(
                drop=True
            ),
        )

        assert final_df is not None
        assert final_df["alder"].tolist() == ["000-015"]

        mock_sum_age.assert_called_once()
        mock_sum_kjonn.assert_called_once_with(df_after_age)
        mock_hierarki.assert_called_once_with(df_after_kjonn)

    def test_returns_base_and_none_when_age_grouping_fails(self, mocker: Any) -> None:
        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.latest_version_path",
            side_effect=["/kommune.parquet", Exception("svalbard missing")],
        )

        df_kommune = pd.DataFrame(
            {
                "kjonn": ["1"],
                "kommuneregion": ["0301"],
                "alder": ["10"],
            }
        )

        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.duckdb.query",
            return_value=MagicMock(to_df=lambda: df_kommune),
        )

        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner."
            "summere_til_aldersgrupperinger.summere_til_aldersgrupperinger",
            side_effect=Exception("age grouping failed"),
        )

        base_df, final_df = hjelpefunksjoner._hent_folkemengde_kommune_31_12(2024)

        assert final_df is None
        assert len(base_df) == 1
        assert base_df.loc[0, "personer"] == 1

    def test_raises_filenotfounderror_when_both_sources_missing(
        self, mocker: Any
    ) -> None:
        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.latest_version_path",
            side_effect=[Exception("kommune missing"), Exception("svalbard missing")],
        )

        with pytest.raises(FileNotFoundError):
            hjelpefunksjoner._hent_folkemengde_kommune_31_12(2024)


# +
# Til v2

from ssb_kostra_python.hjelpefunksjoner import _mapping_mellom_aar

FORVENTEDE_KOLONNER = [
    "kildeaar",
    "oldCode",
    "oldName",
    "oldShortName",
    "statistikkaar",
    "newCode",
    "newName",
    "newShortName",
    "changeOccurred",
]


def lag_endringsmapping() -> pd.DataFrame:
    """Lager en gyldig KLASS-endringsmapping for bruk i testene."""
    return pd.DataFrame(
        {
            "oldCode": ["030101", "030102"],
            "oldName": ["Gammel bydel 1", "Gammel bydel 2"],
            "oldShortName": ["G1", "G2"],
            "newCode": ["030201", "030202"],
            "newName": ["Ny bydel 1", "Ny bydel 2"],
            "newShortName": ["N1", "N2"],
            "changeOccurred": [True, True],
        }
    )


class TestMappingMellomAar:
    def test_kommune_henter_riktig_klassifikasjon_og_datoer(
        self,
        mocker,
    ):
        mock_klass_instance = mocker.Mock()
        mock_klass_instance.get_changes.return_value = lag_endringsmapping()

        mock_klassification = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.KlassClassification",
            return_value=mock_klass_instance,
        )

        resultat = _mapping_mellom_aar(
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        mock_klassification.assert_called_once_with(
            131,
            language="nb",
            include_future=True,
        )
        mock_klass_instance.get_changes.assert_called_once_with(
            "2024-12-31",
            "2025-12-31",
        )

        assert list(resultat.columns) == FORVENTEDE_KOLONNER
        assert resultat["kildeaar"].tolist() == ["2024", "2024"]
        assert resultat["statistikkaar"].tolist() == ["2025", "2025"]

    def test_statistikkaar_kan_vare_streng(
        self,
        mocker,
    ):
        mock_klass_instance = mocker.Mock()
        mock_klass_instance.get_changes.return_value = lag_endringsmapping()

        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.KlassClassification",
            return_value=mock_klass_instance,
        )

        resultat = _mapping_mellom_aar(
            statistikkaar="2025",
            regionsnivaa="kommune",
        )

        mock_klass_instance.get_changes.assert_called_once_with(
            "2024-12-31",
            "2025-12-31",
        )

        assert resultat["kildeaar"].eq("2024").all()
        assert resultat["statistikkaar"].eq("2025").all()

    @pytest.mark.parametrize(
        ("regionsnivaa", "forventet_klass_kode"),
        [
            ("kommune", 131),
            ("bydel", 103),
            ("fylkeskommune", 127),
        ],
    )
    def test_bruker_riktig_klass_kode_for_regionsnivaa(
        self,
        mocker,
        regionsnivaa,
        forventet_klass_kode,
    ):
        mock_klass_instance = mocker.Mock()
        mock_klass_instance.get_changes.return_value = lag_endringsmapping()

        mock_klassification = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.KlassClassification",
            return_value=mock_klass_instance,
        )

        _mapping_mellom_aar(
            statistikkaar=2025,
            regionsnivaa=regionsnivaa,
        )

        mock_klassification.assert_called_once_with(
            forventet_klass_kode,
            language="nb",
            include_future=True,
        )

    def test_regionsnivaa_er_ikke_case_sensitivt(
        self,
        mocker,
    ):
        mock_klass_instance = mocker.Mock()
        mock_klass_instance.get_changes.return_value = lag_endringsmapping()

        mock_klassification = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.KlassClassification",
            return_value=mock_klass_instance,
        )

        _mapping_mellom_aar(
            statistikkaar=2025,
            regionsnivaa="KOMMUNE",
        )

        mock_klassification.assert_called_once_with(
            131,
            language="nb",
            include_future=True,
        )

    def test_bydel_beholder_bare_oslo_bydeler(
        self,
        mocker,
    ):
        df_endringer = pd.DataFrame(
            {
                "oldCode": ["030101", "110101", "030102", "030103"],
                "oldName": [
                    "Oslo 1",
                    "Annen kommune 1",
                    "Oslo 2",
                    "Oslo 3",
                ],
                "oldShortName": ["O1", "A1", "O2", "O3"],
                "newCode": ["030201", "110201", "120201", "030203"],
                "newName": [
                    "Ny Oslo 1",
                    "Ny annen kommune",
                    "Ikke Oslo",
                    "Ny Oslo 3",
                ],
                "newShortName": ["NO1", "NA1", "IO", "NO3"],
                "changeOccurred": [True, True, True, True],
            }
        )

        mock_klass_instance = mocker.Mock()
        mock_klass_instance.get_changes.return_value = df_endringer

        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.KlassClassification",
            return_value=mock_klass_instance,
        )

        resultat = _mapping_mellom_aar(
            statistikkaar=2025,
            regionsnivaa="bydel",
        )

        assert resultat["oldCode"].tolist() == ["030101", "030103"]
        assert resultat["newCode"].tolist() == ["030201", "030203"]
        assert resultat["kildeaar"].eq("2024").all()
        assert resultat["statistikkaar"].eq("2025").all()

    def test_bydel_returnerer_tom_dataframe_nar_ingen_oslo_endringer_finnes(
        self,
        mocker,
    ):
        df_endringer = pd.DataFrame(
            {
                "oldCode": ["110101"],
                "oldName": ["Annen bydel"],
                "oldShortName": ["A1"],
                "newCode": ["110201"],
                "newName": ["Ny annen bydel"],
                "newShortName": ["NA1"],
                "changeOccurred": [True],
            }
        )

        mock_klass_instance = mocker.Mock()
        mock_klass_instance.get_changes.return_value = df_endringer

        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.KlassClassification",
            return_value=mock_klass_instance,
        )

        resultat = _mapping_mellom_aar(
            statistikkaar=2025,
            regionsnivaa="bydel",
        )

        assert resultat.empty
        assert list(resultat.columns) == FORVENTEDE_KOLONNER

    def test_returnerer_tom_dataframe_nar_get_changes_returnerer_none(
        self,
        mocker,
    ):
        mock_klass_instance = mocker.Mock()
        mock_klass_instance.get_changes.return_value = None

        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.KlassClassification",
            return_value=mock_klass_instance,
        )

        resultat = _mapping_mellom_aar(
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        assert resultat.empty
        assert list(resultat.columns) == FORVENTEDE_KOLONNER

    def test_returnerer_tom_dataframe_nar_get_changes_returnerer_tom_dataframe(
        self,
        mocker,
    ):
        mock_klass_instance = mocker.Mock()
        mock_klass_instance.get_changes.return_value = pd.DataFrame()

        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.KlassClassification",
            return_value=mock_klass_instance,
        )

        resultat = _mapping_mellom_aar(
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        assert resultat.empty
        assert list(resultat.columns) == FORVENTEDE_KOLONNER

    def test_returnerer_tom_dataframe_nar_klass_kaster_exception(
        self,
        mocker,
    ):
        mock_klassification = mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.KlassClassification",
            side_effect=RuntimeError("KLASS er utilgjengelig"),
        )

        resultat = _mapping_mellom_aar(
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        mock_klassification.assert_called_once_with(
            131,
            language="nb",
            include_future=True,
        )

        assert resultat.empty
        assert list(resultat.columns) == FORVENTEDE_KOLONNER

    def test_returnerer_tom_dataframe_nar_forventet_kolonne_mangler(
        self,
        mocker,
    ):
        df_endringer = lag_endringsmapping().drop(columns="changeOccurred")

        mock_klass_instance = mocker.Mock()
        mock_klass_instance.get_changes.return_value = df_endringer

        mocker.patch(
            "ssb_kostra_python.hjelpefunksjoner.KlassClassification",
            return_value=mock_klass_instance,
        )

        resultat = _mapping_mellom_aar(
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        assert resultat.empty
        assert list(resultat.columns) == FORVENTEDE_KOLONNER

    def test_ugyldig_regionsnivaa_gir_valueerror(self):
        with pytest.raises(
            ValueError,
            match="regionsnivaa må være",
        ):
            _mapping_mellom_aar(
                statistikkaar=2025,
                regionsnivaa="landsdel",
            )


# +
import pytest

from ssb_kostra_python.hjelpefunksjoner import _regionkolonne


class TestRegionkolonne:
    @pytest.mark.parametrize(
        ("regionsnivaa", "forventet_kolonne"),
        [
            ("kommune", "kommuneregion"),
            ("bydel", "bydelsregion"),
            ("fylkeskommune", "fylkesregion"),
        ],
    )
    def test_returnerer_riktig_regionkolonne(
        self,
        regionsnivaa: str,
        forventet_kolonne: str,
    ) -> None:
        resultat = _regionkolonne(regionsnivaa)

        assert resultat == forventet_kolonne

    @pytest.mark.parametrize(
        "ugyldig_regionsnivaa",
        [
            "landsdel",
            "fylke",
            "",
            "KOMMUNE",
        ],
    )
    def test_ugyldig_regionsnivaa_gir_valueerror(
        self,
        ugyldig_regionsnivaa: str,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="regionsnivaa må være",
        ):
            _regionkolonne(ugyldig_regionsnivaa)


# +
import pytest

from ssb_kostra_python.hjelpefunksjoner import _normaliser_regionkode


class TestNormaliserRegionkode:
    @pytest.mark.parametrize(
        ("verdi", "regionsnivaa", "forventet"),
        [
            (301, "kommune", "0301"),
            ("301", "kommune", "0301"),
            ("0301", "kommune", "0301"),
            (31, "fylkeskommune", "0031"),
            (30101, "bydel", "030101"),
            ("030101", "bydel", "030101"),
            (" EKG01 ", "kommune", "EKG01"),
            (" EAK ", "fylkeskommune", "EAK"),
        ],
    )
    def test_normaliserer_regionkode(
        self,
        verdi: object,
        regionsnivaa: str,
        forventet: str,
    ) -> None:
        resultat = _normaliser_regionkode(verdi, regionsnivaa)

        assert resultat == forventet

    @pytest.mark.parametrize(
        ("regionsnivaa", "forventet"),
        [
            ("KOMMUNE", "0301"),
            ("Bydel", "000301"),
            ("FYLKESKOMMUNE", "0301"),
        ],
    )
    def test_regionsnivaa_er_ikke_case_sensitivt(
        self,
        regionsnivaa: str,
        forventet: str,
    ) -> None:
        resultat = _normaliser_regionkode(301, regionsnivaa)

        assert resultat == forventet

    def test_ugyldig_regionsnivaa_gir_valueerror(self) -> None:
        with pytest.raises(
            ValueError,
            match="regionsnivaa må være",
        ):
            _normaliser_regionkode(301, "landsdel")


# +
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ssb_kostra_python.hjelpefunksjoner import _anvende_kommunereform


class TestAnvendeKommunereform:
    def test_mapping_none_oppdaterer_bare_periode(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": ["0301", "1103"],
                "periode": ["2024", "2024"],
                "alder": ["0-5", "0-5"],
                "personer": [100, 200],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=None,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        forventet = pd.DataFrame(
            {
                "kommuneregion": ["0301", "1103"],
                "periode": ["2025", "2025"],
                "alder": ["0-5", "0-5"],
                "personer": [100, 200],
            }
        )

        assert_frame_equal(resultat, forventet)

    def test_tom_mapping_oppdaterer_bare_periode(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": ["0301", "1103"],
                "periode": ["2024", "2024"],
                "personer": [100, 200],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=pd.DataFrame(columns=["oldCode", "newCode"]),
            statistikkvariable=["personer"],
            statistikkaar="2025",
            regionsnivaa="kommune",
        )

        assert resultat["kommuneregion"].tolist() == ["0301", "1103"]
        assert resultat["periode"].tolist() == ["2025", "2025"]
        assert resultat["personer"].tolist() == [100, 200]

    def test_en_til_en_mapping_endrer_regionkode(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": ["0301", "1103"],
                "periode": ["2024", "2024"],
                "alder": ["0-5", "0-5"],
                "personer": [100, 200],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": ["0301"],
                "newCode": ["0302"],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=mapping,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        assert resultat["kommuneregion"].tolist() == ["0302", "1103"]
        assert resultat["periode"].eq("2025").all()
        assert resultat["personer"].tolist() == [100, 200]

    def test_regioner_uten_mapping_kopieres_uendret(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": ["0301", "1103", "5001"],
                "periode": ["2024", "2024", "2024"],
                "personer": [100, 200, 300],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": ["0301"],
                "newCode": ["0302"],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=mapping,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        assert resultat["kommuneregion"].tolist() == [
            "0302",
            "1103",
            "5001",
        ]

    def test_sammenslaing_aggregerer_statistikkvariable(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": ["1501", "1502"],
                "periode": ["2024", "2024"],
                "alder": ["0-5", "0-5"],
                "kjonn": ["Begge kjønn", "Begge kjønn"],
                "personer": [100, 250],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": ["1501", "1502"],
                "newCode": ["1503", "1503"],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=mapping,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        forventet = pd.DataFrame(
            {
                "kommuneregion": ["1503"],
                "periode": ["2025"],
                "alder": ["0-5"],
                "kjonn": ["Begge kjønn"],
                "personer": [350],
            }
        )

        assert_frame_equal(
            resultat.reset_index(drop=True),
            forventet,
        )

    def test_sammenslaing_aggregerer_flere_statistikkvariable(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": ["1501", "1502"],
                "periode": ["2024", "2024"],
                "alder": ["0-5", "0-5"],
                "personer": [100, 250],
                "arbeidsledige": [10, 20],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": ["1501", "1502"],
                "newCode": ["1503", "1503"],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=mapping,
            statistikkvariable=["personer", "arbeidsledige"],
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        assert len(resultat) == 1
        assert resultat.loc[0, "kommuneregion"] == "1503"
        assert resultat.loc[0, "personer"] == 350
        assert resultat.loc[0, "arbeidsledige"] == 30

    def test_sammenslaing_aggregerer_bare_like_klassifikasjonsnokler(
        self,
    ) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": ["1501", "1502"],
                "periode": ["2024", "2024"],
                "alder": ["0-5", "6-15"],
                "personer": [100, 250],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": ["1501", "1502"],
                "newCode": ["1503", "1503"],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=mapping,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        assert len(resultat) == 2
        assert resultat["kommuneregion"].eq("1503").all()
        assert set(resultat["alder"]) == {"0-5", "6-15"}
        assert resultat["personer"].sum() == 350

    def test_splitting_oppretter_en_rad_per_ny_region(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": ["1501"],
                "periode": ["2024"],
                "alder": ["0-5"],
                "personer": [100],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": ["1501", "1501"],
                "newCode": ["1502", "1503"],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=mapping,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        resultat = resultat.sort_values("kommuneregion").reset_index(drop=True)

        forventet = pd.DataFrame(
            {
                "kommuneregion": ["1502", "1503"],
                "periode": ["2025", "2025"],
                "alder": ["0-5", "0-5"],
                "personer": [100, 100],
            }
        )

        assert_frame_equal(resultat, forventet)

    def test_normaliserer_kommunekoder_i_input_og_mapping(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": [301, 1103],
                "periode": ["2024", "2024"],
                "personer": [100, 200],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": [301],
                "newCode": [302],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=mapping,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        assert resultat["kommuneregion"].tolist() == ["0302", "1103"]

    def test_normaliserer_bydelskoder_til_seks_sifre(self) -> None:
        inputfil = pd.DataFrame(
            {
                "bydelsregion": [30101, 30102],
                "periode": ["2024", "2024"],
                "personer": [100, 200],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": [30101],
                "newCode": [30201],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=mapping,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="bydel",
        )

        assert resultat["bydelsregion"].tolist() == [
            "030201",
            "030102",
        ]

    def test_bruker_fylkesregion_for_fylkeskommune(self) -> None:
        inputfil = pd.DataFrame(
            {
                "fylkesregion": [3, 11],
                "periode": ["2024", "2024"],
                "personer": [100, 200],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": [3],
                "newCode": [31],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=mapping,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="fylkeskommune",
        )

        assert resultat["fylkesregion"].tolist() == [
            "0031",
            "0011",
        ]

    def test_regionsnivaa_er_ikke_case_sensitivt(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": [301],
                "periode": ["2024"],
                "personer": [100],
            }
        )

        resultat = _anvende_kommunereform(
            inputfil=inputfil,
            mapping=None,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="KOMMUNE",
        )

        assert resultat.loc[0, "kommuneregion"] == "0301"

    def test_ugyldig_regionsnivaa_gir_valueerror(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": ["0301"],
                "periode": ["2024"],
                "personer": [100],
            }
        )

        with pytest.raises(
            ValueError,
            match="regionsnivaa må være",
        ):
            _anvende_kommunereform(
                inputfil=inputfil,
                mapping=None,
                statistikkvariable=["personer"],
                statistikkaar=2025,
                regionsnivaa="landsdel",
            )

    def test_manglende_regionkolonne_gir_keyerror(self) -> None:
        inputfil = pd.DataFrame(
            {
                "region": ["0301"],
                "periode": ["2024"],
                "personer": [100],
            }
        )

        with pytest.raises(KeyError, match="kommuneregion"):
            _anvende_kommunereform(
                inputfil=inputfil,
                mapping=None,
                statistikkvariable=["personer"],
                statistikkaar=2025,
                regionsnivaa="kommune",
            )

    @pytest.mark.parametrize(
        "manglende_kolonne",
        [
            "oldCode",
            "newCode",
        ],
    )
    def test_manglende_mappingkolonne_gir_keyerror(
        self,
        manglende_kolonne: str,
    ) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": ["0301"],
                "periode": ["2024"],
                "personer": [100],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": ["0301"],
                "newCode": ["0302"],
            }
        ).drop(columns=manglende_kolonne)

        with pytest.raises(KeyError, match=manglende_kolonne):
            _anvende_kommunereform(
                inputfil=inputfil,
                mapping=mapping,
                statistikkvariable=["personer"],
                statistikkaar=2025,
                regionsnivaa="kommune",
            )

    def test_inputfil_endres_ikke(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": [301],
                "periode": ["2024"],
                "personer": [100],
            }
        )
        inputfil_for = inputfil.copy(deep=True)

        _anvende_kommunereform(
            inputfil=inputfil,
            mapping=None,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        assert_frame_equal(inputfil, inputfil_for)

    def test_mapping_endres_ikke(self) -> None:
        inputfil = pd.DataFrame(
            {
                "kommuneregion": [301],
                "periode": ["2024"],
                "personer": [100],
            }
        )

        mapping = pd.DataFrame(
            {
                "oldCode": [301],
                "newCode": [302],
            }
        )
        mapping_for = mapping.copy(deep=True)

        _anvende_kommunereform(
            inputfil=inputfil,
            mapping=mapping,
            statistikkvariable=["personer"],
            statistikkaar=2025,
            regionsnivaa="kommune",
        )

        assert_frame_equal(mapping, mapping_for)
