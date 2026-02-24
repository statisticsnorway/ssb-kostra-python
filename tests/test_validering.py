import logging
from typing import Any

import pandas as pd
import pytest

from ssb_kostra_python.validering import _klass_check
from ssb_kostra_python.validering import _missing_cols
from ssb_kostra_python.validering import _missing_values
from ssb_kostra_python.validering import _number_of_periods_in_df
from ssb_kostra_python.validering import _valid_periode_region


@pytest.fixture
def df_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "periode": ["2024", "2024", "2024"],
            "kommuneregion": ["0301", "1506", "4601"],
            "fylkesregion": [
                "0300",
                "1500",
                "4600",
            ],  # intentionally mixed for some tests
            "bydelsregion": ["030101", "030103", "030107"],  # last one invalid by range
            "funksjon": ["100", "200", "999"],
        }
    )


# Tests for _missing_cols and _missing_values
# tests/test_validation_helpers.py


def test_missing_cols_logs_error_when_missing(
    caplog: Any, df_base: pd.DataFrame
) -> None:
    caplog.clear()
    _missing_cols(df_base, ["periode", "kommuneregion", "MISSING_COL"])

    assert "Missing required column(s)" in caplog.text
    assert "MISSING_COL" in caplog.text


def test_missing_cols_logs_ok_when_all_present(
    caplog: Any, df_base: pd.DataFrame
) -> None:
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _missing_cols(df_base, ["periode", "kommuneregion"])
    assert "No missing columns" in caplog.text


def test_missing_values_detects_native_na_and_tokens(caplog: Any, mocker: Any) -> None:
    df = pd.DataFrame(
        {
            "periode": ["2024", None, "  ", "nan", "000<NA>", "2024"],
            "kommuneregion": ["0301", "0301", "0301", "0301", "0301", "0301"],
        }
    )

    # prevent notebook UI side effects
    mock_show_toggle = mocker.patch("ssb_kostra_python.validering.show_toggle")

    caplog.clear()
    _missing_values(df, ["periode", "kommuneregion"], preview_rows=5)

    # should flag periode
    assert "Missing values detected in 'periode'" in caplog.text
    # should have called show_toggle at least once for periode
    assert mock_show_toggle.call_count >= 1


def test_missing_values_zero_codes_are_valid_when_configured(
    caplog: Any, mocker: Any
) -> None:
    df = pd.DataFrame(
        {
            "noekkelkode": ["000", "0", "00", "0000"],  # all should be valid if allowed
        }
    )

    mock_show_toggle = mocker.patch("ssb_kostra_python.validering.show_toggle")

    caplog.clear()
    _missing_values(df, ["noekkelkode"], zeros_valid_for={"noekkelkode"})

    # no error
    assert "Missing values detected in 'noekkelkode'" not in caplog.text
    assert mock_show_toggle.call_count == 0


def test_missing_values_zero_codes_are_missing_by_default(
    caplog: Any, mocker: Any
) -> None:
    df = pd.DataFrame({"noekkelkode": ["000", "0", "00", "0000"]})

    mock_show_toggle = mocker.patch("ssb_kostra_python.validering.show_toggle")

    caplog.clear()
    _missing_values(df, ["noekkelkode"])  # zeros_valid_for not set

    assert "Missing values detected in 'noekkelkode'" in caplog.text
    assert mock_show_toggle.call_count >= 1


# Tests for _valid_periode_region
def test_valid_periode_region_flags_bad_periode_format(
    caplog: Any, mocker: Any
) -> None:
    df = pd.DataFrame({"periode": ["2024", "202P", "  ", "000<NA>"]})

    mock_show_toggle = mocker.patch("ssb_kostra_python.validering.show_toggle")

    caplog.clear()
    _valid_periode_region(df, ["periode"])

    # "202P" should be format-invalid
    assert "not four digits" in caplog.text
    assert mock_show_toggle.call_count >= 1


def test_valid_periode_region_kommuneregion_requires_4_digits_if_numeric(
    caplog: Any, mocker: Any
) -> None:
    df = pd.DataFrame({"kommuneregion": ["0301", "301", "03A1", "  "]})
    mock_show_toggle = mocker.patch("ssb_kostra_python.validering.show_toggle")

    caplog.clear()
    _valid_periode_region(df, ["kommuneregion"])

    # "301" numeric but length != 4 => flagged
    assert "not four digits" in caplog.text
    assert mock_show_toggle.call_count >= 1

    # "03A1" contains non-digits => allowed by your rule (only digits-only are constrained)
    # so the test mainly checks the numeric-length rule works.


def test_valid_periode_region_bydelsregion_range_and_length(
    caplog: Any, mocker: Any
) -> None:
    df = pd.DataFrame(
        {"bydelsregion": ["030101", "039999", "030000", "03010", "ABCDEF"]}
    )
    mock_show_toggle = mocker.patch("ssb_kostra_python.validering.show_toggle")

    caplog.clear()
    _valid_periode_region(df, ["bydelsregion"])

    # invalids: "030000" (below 030101), "03010" (len 5)
    assert "must be 6-digit numeric in 030101-039999" in caplog.text
    assert mock_show_toggle.call_count >= 1


# Tests for _number_of_periods_in_df
def test_number_of_periods_returns_only_valid_years(caplog: Any, mocker: Any) -> None:
    df = pd.DataFrame(
        {"periode": ["2024", "2025", "202P", None, "  ", "000<NA>", "2024"]}
    )

    mock_show_toggle = mocker.patch("ssb_kostra_python.validering.show_toggle")

    caplog.clear()
    valid = _number_of_periods_in_df(df)

    assert sorted(valid) == ["2024", "2025"]
    assert "Format-invalid 'periode' tokens" in caplog.text
    assert mock_show_toggle.call_count >= 1


# Tests for _klass_check (mock KLASS; no network)
class FakeKlass:
    def __init__(
        self,
        klass_id: str | int,
        language: str = "en",
        include_future: bool = True,
    ) -> None:
        """Initialize FakeKlass with the given parameters.

        Args:
            klass_id: The KLASS classification ID.
            language: The language code (default "en").
            include_future: Whether to include future codes (default True).
        """
        self.klass_id = klass_id

    def get_codes(self, from_date: Any = None, to_date: Any = None) -> pd.DataFrame:
        # Return something that looks like the library output:
        # your code supports either `.data` or a dataframe directly
        return pd.DataFrame({"code": ["0301", "1101", "9999", "100", "200"]})


def test_klass_check_skips_when_multiple_periods(caplog: Any, mocker: Any) -> None:
    df = pd.DataFrame(
        {
            "periode": ["2024", "2025"],  # multiple valid years => should skip
            "kommuneregion": ["0301", "0301"],
        }
    )

    # if it tries to call klass, we want to know (but it should skip before that)
    mocker.patch(
        "ssb_kostra_python.validering.KlassClassification",
        side_effect=RuntimeError("Should not be called"),
    )

    caplog.clear()
    _klass_check(df, ["periode", "kommuneregion"], interactive=False)

    assert "contains 2 valid periods" in caplog.text
    assert "KLASS check runs only when exactly one period is present" in caplog.text


def test_klass_check_flags_invalid_codes(caplog: Any, mocker: Any) -> None:
    df = pd.DataFrame(
        {
            "periode": ["2024", "2024", "2024"],
            "kommuneregion": ["0301", "0301", "XXXX"],  # XXXX not in fake KLASS codes
        }
    )

    mocker.patch("ssb_kostra_python.validering.KlassClassification", FakeKlass)

    mock_show_toggle = mocker.patch("ssb_kostra_python.validering.show_toggle")

    caplog.clear()
    _klass_check(df, ["periode", "kommuneregion"], interactive=False)

    assert "contains codes not present in KLASS for 2024" in caplog.text
    assert mock_show_toggle.call_count >= 1


def test_klass_check_passes_when_all_codes_valid(caplog: Any, mocker: Any) -> None:
    df = pd.DataFrame(
        {
            "periode": ["2024", "2024"],
            "kommuneregion": ["0301", "1101"],
        }
    )

    mocker.patch("ssb_kostra_python.validering.KlassClassification", FakeKlass)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        _klass_check(df, ["periode", "kommuneregion"], interactive=False)

    assert "All 'kommuneregion' codes are present in KLASS for 2024" in caplog.text


# Testing interactive prompt path
def test_klass_check_prompts_for_unknown_cols(mocker: Any, caplog: Any) -> None:
    df = pd.DataFrame(
        {
            "periode": ["2024", "2024"],
            "funksjon": ["100", "999"],  # unknown col => prompts for KLASS id
        }
    )

    mocker.patch("ssb_kostra_python.validering.KlassClassification", FakeKlass)
    mocker.patch("builtins.input", return_value="277")  # user supplies a KLASS id

    caplog.clear()
    _klass_check(df, ["periode", "funksjon"], interactive=True)

    # With FakeKlass codes including 100/200 only, 999 should be flagged
    assert "contains codes not present" in caplog.text
