import inspect

import pandas as pd
import pytest

from ssb_kostra_python.enkel_editering import dataframe_cell_editor_mvp


@pytest.fixture
def df():
    """Create a small representative DataFrame for testing."""
    return pd.DataFrame(
        {
            "periode": ["2024K4", "2024K4"],
            "kommuneregion": ["0301", "0301"],
            "utgifter": pd.Series([10, 20], dtype="Int64"),
            "flag": pd.Series([True, False], dtype="boolean"),
            "tekst": pd.Series(["ja", "nei"], dtype="string"),
        }
    )


def test_initial_get_results_returns_copy_and_empty_log(mocker, df):
    """Verify the "initial state" behavior of the editor."""
    # Arrange: Make the helper deterministic (no input prompts)
    mock_define = mocker.patch(
        "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable",
        autospec=True,
    )
    mock_define.return_value = (
        ["periode", "kommuneregion"],
        ["utgifter", "flag", "tekst"],
    )
    mocker.patch("ssb_kostra_python.enkel_editering.display", autospec=True)
    mocker.patch("ssb_kostra_python.enkel_editering.clear_output", autospec=True)

    # Act: Build editor
    get_results = dataframe_cell_editor_mvp(df, preview_rows=5, log_rows=None)

    # Act: Export current state (no edits yet).
    df_edited, log_df = get_results()

    # Assert: Original data should be preserved
    pd.testing.assert_frame_equal(df_edited, df)

    # Assert: Internal ROW_ID should not leak out
    assert "__row_id__" not in df_edited.columns

    # Assert: Log should be empty initially
    assert log_df.empty


def test_get_results_reflects_mutations_to_captured_state(mocker, df):
    """Validate that `get_results()` reflects the *current internal state* of the editor."""
    # Arrange: deterministic helper output
    mock_define = mocker.patch(
        "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable",
        autospec=True,
    )
    mock_define.return_value = (
        ["periode", "kommuneregion"],
        ["utgifter", "flag", "tekst"],
    )
    mocker.patch("ssb_kostra_python.enkel_editering.display", autospec=True)
    mocker.patch("ssb_kostra_python.enkel_editering.clear_output", autospec=True)

    # Act: build the editor
    get_results = dataframe_cell_editor_mvp(df, preview_rows=5, log_rows=None)

    # Inspect closure to access internal state (nonlocals)
    closure = inspect.getclosurevars(get_results)
    df_working = closure.nonlocals["df_working"]
    change_log = closure.nonlocals["change_log"]
    row_id_col = closure.nonlocals["ROW_ID"]

    # Simulate one committed edit
    rid0 = int(df_working.loc[0, row_id_col])
    old_val = df_working.loc[0, "utgifter"]
    new_val = 9999
    df_working.loc[0, "utgifter"] = new_val

    # Append a log entry
    change_log.append(
        {
            "timestamp": "2026-01-06T12:00:00",
            "user": "tester",
            "row_id": rid0,
            "column": "utgifter",
            "old_value": old_val,
            "new_value": new_val,
            "reason": "unit test change",
            "id_periode": df_working.loc[0, "periode"],
            "id_kommuneregion": df_working.loc[0, "kommuneregion"],
        }
    )

    # Act: export results
    df_edited, log_df = get_results()

    # Assert: edited value is present
    assert int(df_edited.loc[0, "utgifter"]) == 9999

    # Assert: change log contains our entry
    assert len(log_df) == 1
    assert log_df.loc[0, "column"] == "utgifter"
    assert log_df.loc[0, "new_value"] == 9999
    assert log_df.loc[0, "reason"] == "unit test change"

    # Assert: internal ROW_ID removed
    assert "__row_id__" not in df_edited.columns


def test_row_id_is_added_internally_but_not_returned(mocker, df):
    """Verify that ROW_ID is used internally but not exposed in the exported DataFrame."""
    # Arrange
    mock_define = mocker.patch(
        "ssb_kostra_python.hjelpefunksjoner.definere_klassifikasjonsvariable",
        autospec=True,
    )
    mock_define.return_value = (
        ["periode", "kommuneregion"],
        ["utgifter", "flag", "tekst"],
    )
    mocker.patch("ssb_kostra_python.enkel_editering.display", autospec=True)
    mocker.patch("ssb_kostra_python.enkel_editering.clear_output", autospec=True)

    # Act: build editor
    get_results = dataframe_cell_editor_mvp(df, preview_rows=5, log_rows=None)

    closure = inspect.getclosurevars(get_results)
    df_working = closure.nonlocals["df_working"]
    row_id_col = closure.nonlocals["ROW_ID"]

    # Assert: ROW_ID exists internally
    assert row_id_col in df_working.columns

    # Assert: ROW_ID does not appear in exported results
    df_edited, _ = get_results()
    assert row_id_col not in df_edited.columns
