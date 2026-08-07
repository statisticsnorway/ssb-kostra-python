import contextlib
import io
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ssb_kostra_python.avrunding import _round_half_up
from ssb_kostra_python.avrunding import konverter_dtypes
from ssb_kostra_python.avrunding import print_instruks_konverter_dtypes


class TestRoundHalfUp:
    """Tester hjelpefunksjonen for kommersiell avrunding."""

    def test_round_half_up_to_integer(self) -> None:
        """Halvverdier skal avrundes bort fra null."""
        values = pd.Series(
            [0.5, 1.5, 2.4, 2.5, -0.5, -1.5, -2.5, -2.4],
            name="verdier",
        )

        result = _round_half_up(values, decimals=0)

        expected = pd.Series(
            [1.0, 2.0, 2.0, 3.0, -1.0, -2.0, -3.0, -2.0],
            name="verdier",
            dtype="float64",
        )

        pd.testing.assert_series_equal(result, expected)

    def test_round_half_up_problematic_float_values(self) -> None:
        """Verdier som tidligere ga feil på grunn av binære flyttall."""
        values = pd.Series(
            [0.125, 0.575, 1.005, 1.275, 2.445],
            name="var2",
        )

        result = _round_half_up(values, decimals=2)

        expected = pd.Series(
            [0.13, 0.58, 1.01, 1.28, 2.45],
            name="var2",
            dtype="float64",
        )

        pd.testing.assert_series_equal(result, expected)

    def test_round_half_up_negative_decimal_values(self) -> None:
        """Negative halvverdier skal også avrundes bort fra null."""
        values = pd.Series(
            [-0.125, -0.575, -1.005, -1.275, -2.445],
            name="verdier",
        )

        result = _round_half_up(values, decimals=2)

        expected = pd.Series(
            [-0.13, -0.58, -1.01, -1.28, -2.45],
            name="verdier",
            dtype="float64",
        )

        pd.testing.assert_series_equal(result, expected)

    def test_round_half_up_with_one_decimal(self) -> None:
        """Funksjonen skal kunne runde til én desimal."""
        values = pd.Series(
            [1.25, 1.35, -1.25, -1.35],
            name="verdier",
        )

        result = _round_half_up(values, decimals=1)

        expected = pd.Series(
            [1.3, 1.4, -1.3, -1.4],
            name="verdier",
            dtype="float64",
        )

        pd.testing.assert_series_equal(result, expected)

    def test_round_half_up_preserves_missing_values(self) -> None:
        """Manglende og ugyldige verdier skal bli NaN."""
        values = pd.Series(
            [1.25, None, np.nan, "ikke et tall"],
            name="verdier",
        )

        result = _round_half_up(values, decimals=1)

        expected = pd.Series(
            [1.3, np.nan, np.nan, np.nan],
            name="verdier",
            dtype="float64",
        )

        pd.testing.assert_series_equal(result, expected)

    def test_round_half_up_accepts_numeric_strings(self) -> None:
        """Tekstverdier som representerer tall skal kunne avrundes."""
        values = pd.Series(
            ["1.005", "2.445", "-1.275"],
            name="verdier",
        )

        result = _round_half_up(values, decimals=2)

        expected = pd.Series(
            [1.01, 2.45, -1.28],
            name="verdier",
            dtype="float64",
        )

        pd.testing.assert_series_equal(result, expected)

    def test_round_half_up_preserves_index_and_name(self) -> None:
        """Resultatet skal beholde indeks og serienavn."""
        values = pd.Series(
            [1.25, 2.35],
            index=["a", "b"],
            name="belop",
        )

        result = _round_half_up(values, decimals=1)

        expected = pd.Series(
            [1.3, 2.4],
            index=["a", "b"],
            name="belop",
            dtype="float64",
        )

        pd.testing.assert_series_equal(result, expected)

    def test_round_half_up_rejects_negative_decimals(self) -> None:
        """Negativt antall desimaler skal gi ValueError."""
        values = pd.Series([1.25])

        with pytest.raises(
            ValueError,
            match="'decimals' kan ikke være negativ",
        ):
            _round_half_up(values, decimals=-1)


class TestKonverterDtypes:
    """Tester konvertering av kolonner basert på dtype-mapping."""

    def test_converts_all_groups_correctly(self, mocker: Any) -> None:
        """Alle støttede konverteringsgrupper skal behandles korrekt."""
        mock_logger = mocker.patch(
            "ssb_kostra_python.avrunding.logger",
            autospec=True,
        )
        mock_display = mocker.patch(
            "ssb_kostra_python.avrunding.display",
            autospec=True,
        )

        df = pd.DataFrame(
            {
                "kategori": ["a", "b", "a", None],
                "heltall": [0.5, 1.5, np.nan, -2.5],
                "desimal_1": [1.25, 1.35, -1.25, np.nan],
                "desimal_2": [0.125, 0.575, 1.005, 2.445],
                "tekst": [1, None, "A", 4],
                "boolsk": [1, 0, None, 1],
                "uendret": ["x", "y", "z", "w"],
            }
        )

        original = df.copy(deep=True)

        mapping = {
            "klassifikasjonsvariabel": ["kategori"],
            "heltall": ["heltall"],
            "desimaltall_1_des": ["desimal_1"],
            "desimaltall_2_des": ["desimal_2"],
            "stringvar": ["tekst"],
            "bool_var": ["boolsk"],
        }

        result, dtypes = konverter_dtypes(df, mapping)

        # Funksjonen skal returnere en kopi.
        assert result is not df

        # Den opprinnelige dataframen skal ikke endres.
        pd.testing.assert_frame_equal(df, original)

        assert isinstance(result["kategori"].dtype, pd.CategoricalDtype)

        expected_heltall = pd.Series(
            [1, 2, pd.NA, -3],
            name="heltall",
            dtype="Int64",
        )
        pd.testing.assert_series_equal(
            result["heltall"],
            expected_heltall,
        )

        expected_desimal_1 = pd.Series(
            [1.3, 1.4, -1.3, np.nan],
            name="desimal_1",
            dtype="float64",
        )
        pd.testing.assert_series_equal(
            result["desimal_1"],
            expected_desimal_1,
        )

        expected_desimal_2 = pd.Series(
            [0.13, 0.58, 1.01, 2.45],
            name="desimal_2",
            dtype="float64",
        )
        pd.testing.assert_series_equal(
            result["desimal_2"],
            expected_desimal_2,
        )

        expected_tekst = pd.Series(
            ["1", pd.NA, "A", "4"],
            name="tekst",
            dtype="string",
        )
        pd.testing.assert_series_equal(
            result["tekst"],
            expected_tekst,
        )

        expected_boolsk = pd.Series(
            [True, False, pd.NA, True],
            name="boolsk",
            dtype="boolean",
        )
        pd.testing.assert_series_equal(
            result["boolsk"],
            expected_boolsk,
        )

        expected_uendret = pd.Series(
            ["x", "y", "z", "w"],
            name="uendret",
        )
        pd.testing.assert_series_equal(
            result["uendret"],
            expected_uendret,
        )

        pd.testing.assert_series_equal(
            dtypes,
            result.dtypes,
        )

        assert mock_display.call_count == 2
        assert mock_logger.info.called

    def test_empty_mapping_leaves_dataframe_unchanged(
        self,
        mocker: Any,
    ) -> None:
        """En tom mapping skal ikke endre innhold eller datatyper."""
        mocker.patch(
            "ssb_kostra_python.avrunding.logger",
            autospec=True,
        )
        mocker.patch(
            "ssb_kostra_python.avrunding.display",
            autospec=True,
        )

        df = pd.DataFrame(
            {
                "a": [1, 2],
                "b": ["x", "y"],
            }
        )

        result, dtypes = konverter_dtypes(df, {})

        pd.testing.assert_frame_equal(result, df)
        pd.testing.assert_series_equal(dtypes, result.dtypes)

        assert result is not df

    def test_warns_for_missing_column_and_unknown_group(
        self,
        mocker: Any,
    ) -> None:
        """Manglende kolonner og ukjente grupper skal gi advarsler."""
        mocker.patch(
            "ssb_kostra_python.avrunding.logger",
            autospec=True,
        )
        mock_display = mocker.patch(
            "ssb_kostra_python.avrunding.display",
            autospec=True,
        )

        df = pd.DataFrame({"x": [1, 2]})

        mapping = {
            "heltall": ["missing_col"],
            "weird_group": ["x"],
        }

        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            result, dtypes = konverter_dtypes(df, mapping)

        printed = buffer.getvalue()

        assert "Advarsel: Kolonnen 'missing_col' finnes ikke i dataframen." in printed
        assert (
            "Advarsel: Ukjent gruppe 'weird_group' "
            "for kolonnen 'x'. Ingen konvertering utført." in printed
        )

        pd.testing.assert_frame_equal(result, df)
        pd.testing.assert_series_equal(dtypes, result.dtypes)

        assert mock_display.call_count == 2

    def test_same_column_can_be_processed_in_mapping_order(
        self,
        mocker: Any,
    ) -> None:
        """Mappingen behandles i innsettingsrekkefølge."""
        mocker.patch(
            "ssb_kostra_python.avrunding.logger",
            autospec=True,
        )
        mocker.patch(
            "ssb_kostra_python.avrunding.display",
            autospec=True,
        )

        df = pd.DataFrame({"verdi": [1.25, 2.55]})

        mapping = {
            "desimaltall_1_des": ["verdi"],
            "stringvar": ["verdi"],
        }

        result, _ = konverter_dtypes(df, mapping)

        expected = pd.Series(
            ["1.3", "2.6"],
            name="verdi",
            dtype="string",
        )

        pd.testing.assert_series_equal(
            result["verdi"],
            expected,
        )


class TestPrintInstruksKonverterDtypes:
    """Tester funksjonen som skriver ut mappinginstruksen."""

    def test_prints_and_returns_instruction(self) -> None:
        """Instruksen skal både skrives ut og returneres."""
        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            result = print_instruks_konverter_dtypes()

        printed = buffer.getvalue()

        assert result in printed
        assert "dtype_mapping" in result
        assert '"klassifikasjonsvariabel"' in result
        assert '"heltall"' in result
        assert '"desimaltall_2_des"' in result
        assert '"bool_var"' in result
