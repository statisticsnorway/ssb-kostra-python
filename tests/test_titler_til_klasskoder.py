from typing import Any

import pandas as pd
import pytest

from ssb_kostra_python.titler_til_klasskoder import _attach_one_mapping
from ssb_kostra_python.titler_til_klasskoder import _fetch_mapping_for_year
from ssb_kostra_python.titler_til_klasskoder import _pick_level_columns
from ssb_kostra_python.titler_til_klasskoder import kodelister_navn

# --- Test doubles (fakes) for KLASS ---


class FakeCodes:
    """Test double for the object returned by KlassClassification.get_codes(...)."""

    def __init__(self, pivot_df: pd.DataFrame):
        """Initialize FakeCodes with a DataFrame to be returned by pivot_level."""
        self._pivot_df = pivot_df

    def pivot_level(self) -> pd.DataFrame:
        """Return the pivot table as if it came from real KLASS."""
        return self._pivot_df


from typing import ClassVar

import pandas as pd


class FakeKlassClassification:
    """Test double for klass.KlassClassification."""

    pivot_df: ClassVar[pd.DataFrame | None] = None  # set per test

    def __init__(
        self, klass_id: int, language: str = "nb", include_future: bool = True
    ) -> None:
        """Initialize FakeKlassClassification with the given parameters."""
        self.klass_id = klass_id
        self.language = language
        self.include_future = include_future

    def get_codes(self, **kwargs: Any) -> FakeCodes:
        """Return a FakeCodes object containing the test-provided pivot table."""
        if self.__class__.pivot_df is None:
            raise ValueError(
                "pivot_df must be set to a DataFrame before calling get_codes."
            )
        return FakeCodes(self.__class__.pivot_df)


PATCH_TARGET = "ssb_kostra_python.titler_til_klasskoder.KlassClassification"


# -------------------------
# _pick_level_columns tests
# -------------------------


def test_pick_level_columns_selects_smallest_level_when_none() -> None:
    """Verify level selection logic."""
    pivot = pd.DataFrame(
        {
            "code_3": ["A"],
            "name_3": ["NameA"],
            "code_1": ["B"],
            "name_1": ["NameB"],
        }
    )
    level, mcode, mname = _pick_level_columns(pivot, level=None)
    assert level == 1
    assert mcode == "code_1"
    assert mname == "name_1"


def test_pick_level_columns_raises_if_missing_expected_columns() -> None:
    """Verify error on missing columns."""
    pivot = pd.DataFrame({"something_else": [1]})
    with pytest.raises(RuntimeError, match="missing expected"):
        _pick_level_columns(pivot, level=None)


def test_pick_level_columns_raises_if_requested_level_not_present() -> None:
    """Verify error on missing requested level."""
    pivot = pd.DataFrame({"code_1": ["1"], "name_1": ["One"]})
    with pytest.raises(RuntimeError, match="Expected columns 'code_2' and 'name_2'"):
        _pick_level_columns(pivot, level=2)


# --------------------------------
# _fetch_mapping_for_year tests
# --------------------------------


def test_fetch_mapping_for_year_returns_two_col_mapping_and_level(mocker: Any) -> None:
    """Verify mapping fetch logic."""
    mocker.patch(PATCH_TARGET, FakeKlassClassification)
    FakeKlassClassification.pivot_df = pd.DataFrame(
        {
            "code_1": [" 0301 ", "0302"],
            "name_1": ["Oslo", "Bergen"],
        }
    )

    mapping, level = _fetch_mapping_for_year(
        klass_id=231,
        year=2024,
        language="nb",
        include_future=True,
        select_level=None,
    )

    assert level == 1
    assert list(mapping.columns) == ["_map_code", "_map_name"]
    assert mapping["_map_code"].tolist() == ["0301", "0302"]
    assert mapping["_map_name"].tolist() == ["Oslo", "Bergen"]


# --------------------------------
# _attach_one_mapping tests
# --------------------------------


def test_attach_one_mapping_inserts_name_column_after_code_and_builds_diagnostics(
    mocker: Any,
) -> None:
    """Verify name attachment and diagnostics."""
    mocker.patch(PATCH_TARGET, FakeKlassClassification)
    FakeKlassClassification.pivot_df = pd.DataFrame(
        {
            "code_1": ["0301", "0302"],
            "name_1": ["Oslo", "Bergen"],
        }
    )

    df_in = pd.DataFrame(
        {
            "periode": [2024, 2024, 2024],
            "kommuneregion": ["0301", "9999", "0302"],
            "value": [10, 20, 30],
        }
    )

    out, diag = _attach_one_mapping(
        df_in,
        year=2024,
        code_col="kommuneregion",
        klass_id=231,
        name_col_out=None,
        language="nb",
        include_future=True,
        select_level=1,
    )

    cols = list(out.columns)
    idx_code = cols.index("kommuneregion")
    assert cols[idx_code + 1] == "kommuneregion_navn"
    assert out["kommuneregion_navn"].tolist()[0] == "Oslo"
    assert pd.isna(out["kommuneregion_navn"].tolist()[1])
    assert out["kommuneregion_navn"].tolist()[2] == "Bergen"

    assert diag["invalid_count"] == 1
    assert diag["invalid_sample"] == ["9999"]
    assert diag["all_invalid"] == ["9999"]
    assert diag["level"] == 1


def test_attach_one_mapping_raises_if_code_col_missing(mocker: Any) -> None:
    """Verify error on missing code column."""
    mocker.patch(PATCH_TARGET, FakeKlassClassification)
    FakeKlassClassification.pivot_df = pd.DataFrame(
        {"code_1": ["1"], "name_1": ["One"]}
    )
    df_in = pd.DataFrame({"periode": [2024], "value": [1]})
    with pytest.raises(ValueError, match="Column 'kommuneregion' not found"):
        _attach_one_mapping(df_in, year=2024, code_col="kommuneregion", klass_id=231)


# -----------------------
# kodelister_navn tests
# -----------------------


def test_kodelister_navn_requires_periode() -> None:
    """Verify error if 'periode' is missing."""
    df = pd.DataFrame({"kommuneregion": ["0301"]})
    with pytest.raises(ValueError, match="must contain a 'periode'"):
        kodelister_navn(
            df,
            mappings=[{"code_col": "kommuneregion", "klass_id": 231}],
            verbose=False,
        )


def test_kodelister_navn_requires_exactly_one_unique_year() -> None:
    """Verify error if multiple years are present."""
    df = pd.DataFrame({"periode": [2023, 2024], "kommuneregion": ["0301", "0301"]})
    with pytest.raises(ValueError, match="exactly one unique value"):
        kodelister_navn(
            df,
            mappings=[{"code_col": "kommuneregion", "klass_id": 231}],
            verbose=False,
        )


def test_kodelister_navn_applies_multiple_mappings_in_order(mocker: Any) -> None:
    """Verify sequential mapping application."""
    mocker.patch(PATCH_TARGET, FakeKlassClassification)
    FakeKlassClassification.pivot_df = pd.DataFrame(
        {
            "code_1": ["A", "B"],
            "name_1": ["Alpha", "Beta"],
        }
    )

    df = pd.DataFrame(
        {
            "periode": [2024, 2024],
            "col1": ["A", "B"],
            "col2": ["B", "A"],
            "value": [1, 2],
        }
    )

    mappings = [
        {"code_col": "col1", "klass_id": 111, "select_level": 1},
        {
            "code_col": "col2",
            "klass_id": 222,
            "name_col_out": "col2_name",
            "select_level": 1,
        },
    ]

    out, diag = kodelister_navn(df, mappings=mappings, verbose=False)
    cols = list(out.columns)

    i1 = cols.index("col1")
    assert cols[i1 + 1] == "col1_navn"
    assert out["col1_navn"].tolist() == ["Alpha", "Beta"]

    i2 = cols.index("col2")
    assert cols[i2 + 1] == "col2_name"
    assert out["col2_name"].tolist() == ["Beta", "Alpha"]

    assert "col1" in diag
    assert "col2" in diag
    assert diag["col1"]["klass_id"] == 111
    assert diag["col2"]["klass_id"] == 222
