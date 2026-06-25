from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ssb_kostra_python import hjelpefunksjoner
from ssb_kostra_python.hjelpefunksjoner import definere_klassifikasjonsvariable
from ssb_kostra_python.hjelpefunksjoner import format_fil
from ssb_kostra_python.hjelpefunksjoner import konvertere_komma_til_punktdesimal


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

        out = konvertere_komma_til_punktdesimal(df)

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

        out = konvertere_komma_til_punktdesimal(df)

        assert out["b"].tolist() == ["x", "y"]
        assert out["c"].tolist() == [10, 20]

    def test_converts_column_if_any_value_contains_comma(self) -> None:
        df = pd.DataFrame({"a": ["1,5", "2"]})

        out = konvertere_komma_til_punktdesimal(df)

        assert np.allclose(out["a"].to_numpy(), [1.5, 2.0])
        assert pd.api.types.is_float_dtype(out["a"])

    def test_does_not_modify_input_dataframe(self) -> None:
        df = pd.DataFrame({"a": ["1,5", "2,0"]})
        df_before = df.copy(deep=True)

        _ = konvertere_komma_til_punktdesimal(df)

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
