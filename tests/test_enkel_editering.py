import unittest
from unittest.mock import patch
import inspect
import pandas as pd
import numpy as np
from pandas.api.types import (
    is_integer_dtype, is_float_dtype, is_bool_dtype, is_extension_array_dtype
)
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime
import getpass

from functions.funksjoner.enkel_editering import dataframe_cell_editor_mvp


class TestDataframeCellEditorMVP(unittest.TestCase):
    """
    Test suite for `dataframe_cell_editor_mvp`.

    What `dataframe_cell_editor_mvp` appears to do (conceptually)
    ------------------------------------------------------------
    - It takes an input DataFrame.
    - It creates an internal "working copy" (often called df_working) that can be edited.
    - It adds an internal row identifier column (ROW_ID) to reliably track edits.
    - It builds an edit UI using ipywidgets and displays it (display/clear_output side effects).
    - It returns a callable (here named `get_results`) which, when called, exports:
        (df_edited, log_df)
      where:
        - df_edited is the edited data (with internal helper columns removed)
        - log_df is a DataFrame of committed edits (change log)

    What we can and cannot test in pure unit tests
    ----------------------------------------------
    - We cannot easily simulate "clicking" widgets (Commit buttons, callbacks, etc.)
      unless the function returns widget handles or exposes callbacks.
    - We *can* test:
        1) That initial export returns a copy of the input and an empty log.
        2) That internal state is captured in the closure and `get_results()` reflects it.
        3) That internal ROW_ID exists internally but is not leaked in exported output.

    Why we patch display/clear_output and helper function
    -----------------------------------------------------
    - display() and clear_output() are notebook UI side effects; unit tests should not
      actually display widgets or clear output.
    - definere_klassifikasjonsvariable likely prompts the user or inspects df; we patch it
      to be deterministic (no interactive inputs, stable outputs).
    """

    def setUp(self):
        """
        setUp runs before each test method.

        We create a small representative DataFrame with:
          - identifier columns: periode, kommuneregion
          - numeric column: utgifter (nullable Int64)
          - boolean column: flag (nullable boolean)
          - string column: tekst (pandas string dtype)

        Using extension dtypes (Int64, boolean, string) is important because
        notebook editors often need to handle them carefully.
        """
        self.df = pd.DataFrame({
            "periode": ["2024K4", "2024K4"],
            "kommuneregion": ["0301", "0301"],
            "utgifter": pd.Series([10, 20], dtype="Int64"),
            "flag": pd.Series([True, False], dtype="boolean"),
            "tekst": pd.Series(["ja", "nei"], dtype="string"),
        })

    @patch("functions.funksjoner.enkel_editering.display", autospec=True)
    @patch("functions.funksjoner.enkel_editering.clear_output", autospec=True)
    @patch("functions.funksjoner.hjelpefunksjoner.definere_klassifikasjonsvariable", autospec=True)
    def test_initial_get_results_returns_copy_and_empty_log(
        self,
        mock_define,
        mock_clear_output,
        mock_display
    ):
        """
        Purpose
        -------
        Verify the "initial state" behavior of the editor:
          - Calling dataframe_cell_editor_mvp returns a callable get_results()
          - Calling get_results() immediately (without any edits committed) returns:
              * a DataFrame equal to the original input
              * an empty change log DataFrame
          - Internal ROW_ID should not appear in the exported DataFrame

        Test strategy
        -------------
        1) Patch definere_klassifikasjonsvariable so the function does not prompt the user
           and has stable, deterministic classification columns.
        2) Call dataframe_cell_editor_mvp(...) to build the editor.
        3) Call get_results() and inspect exported df + log.
        """

        # 1) Arrange: Make the helper deterministic (no input prompts)
        # The editor likely needs to know which columns identify a row vs. editable columns.
        mock_define.return_value = (["periode", "kommuneregion"], ["utgifter", "flag", "tekst"])

        # 2) Act: Build editor; it returns a getter function (closure) for results.
        get_results = dataframe_cell_editor_mvp(self.df, preview_rows=5, log_rows=None)

        # 3) Act: Export current state (no edits yet).
        df_edited, log_df = get_results()

        # 4) Assert: Original data should be preserved (since no edits committed)
        pd.testing.assert_frame_equal(df_edited, self.df)

        # 5) Assert: Internal ROW_ID should not leak out in exported results
        self.assertNotIn("__row_id__", df_edited.columns)

        # 6) Assert: Log should be empty initially
        self.assertTrue(log_df.empty)

    @patch("functions.funksjoner.enkel_editering.display", autospec=True)
    @patch("functions.funksjoner.enkel_editering.clear_output", autospec=True)
    @patch("functions.funksjoner.hjelpefunksjoner.definere_klassifikasjonsvariable", autospec=True)
    def test_get_results_reflects_mutations_to_captured_state(
        self,
        mock_define,
        mock_clear_output,
        mock_display
    ):
        """
        Purpose
        -------
        Validate that `get_results()` reflects the *current internal state* of the editor.

        Why this test exists
        --------------------
        In a real notebook, edits happen via widgets and callbacks (e.g. a Commit button).
        In a unit test, we can't "click Commit" unless the function exposes callback handles.

        However, we CAN:
          - access the closure variables captured by get_results (df_working, change_log)
          - mutate them as if a commit callback had run
          - call get_results() and verify it exports those mutations

        Steps
        -----
        1) Patch definere_klassifikasjonsvariable to avoid interaction.
        2) Create get_results via dataframe_cell_editor_mvp(...).
        3) Use inspect.getclosurevars(get_results) to find internal nonlocals:
             - df_working: internal working DataFrame (with ROW_ID)
             - change_log: list of dicts representing committed edits
             - ROW_ID: name of the internal row id column
        4) Simulate one committed edit:
             - change df_working value
             - append a log entry into change_log
        5) Call get_results() and verify:
             - exported df includes the new value
             - exported log includes our entry
             - internal ROW_ID is removed from exported df
        """

        # 1) Arrange: deterministic helper output
        mock_define.return_value = (["periode", "kommuneregion"], ["utgifter", "flag", "tekst"])

        # 2) Act: build the editor and get the results exporter function
        get_results = dataframe_cell_editor_mvp(self.df, preview_rows=5, log_rows=None)

        # 3) Inspect closure to access internal state (nonlocals)
        closure = inspect.getclosurevars(get_results)
        df_working = closure.nonlocals["df_working"]
        change_log = closure.nonlocals["change_log"]
        row_id_col = closure.nonlocals["ROW_ID"]

        # 4) Simulate one committed edit (similar to what an internal commit callback would do)
        # Identify the internal row id for row 0:
        rid0 = int(df_working.loc[0, row_id_col])

        # Capture old value and set new value
        old_val = df_working.loc[0, "utgifter"]
        new_val = 9999
        df_working.loc[0, "utgifter"] = new_val

        # Append a log entry describing the change
        change_log.append({
            "timestamp": "2026-01-06T12:00:00",
            "user": "tester",
            "row_id": rid0,
            "column": "utgifter",
            "old_value": old_val,
            "new_value": new_val,
            "reason": "unit test change",
            "id_periode": df_working.loc[0, "periode"],
            "id_kommuneregion": df_working.loc[0, "kommuneregion"],
        })

        # 5) Act: export results from the closure
        df_edited, log_df = get_results()

        # 6) Assert: edited value is present in exported DataFrame
        self.assertEqual(int(df_edited.loc[0, "utgifter"]), 9999)

        # 7) Assert: change log exported and contains our entry
        self.assertEqual(len(log_df), 1)
        self.assertEqual(log_df.loc[0, "column"], "utgifter")
        self.assertEqual(log_df.loc[0, "new_value"], 9999)
        self.assertEqual(log_df.loc[0, "reason"], "unit test change")

        # 8) Assert: internal ROW_ID removed in exported df
        self.assertNotIn("__row_id__", df_edited.columns)

    @patch("functions.funksjoner.enkel_editering.display", autospec=True)
    @patch("functions.funksjoner.enkel_editering.clear_output", autospec=True)
    @patch("functions.funksjoner.hjelpefunksjoner.definere_klassifikasjonsvariable", autospec=True)
    def test_row_id_is_added_internally_but_not_returned(
        self,
        mock_define,
        mock_clear_output,
        mock_display
    ):
        """
        Purpose
        -------
        Verify that ROW_ID is used internally to track rows, but is not exposed
        in the returned/exported DataFrame from get_results().

        Steps
        -----
        1) Build editor, capture get_results().
        2) Inspect closure to verify df_working contains ROW_ID column.
        3) Call get_results() and verify ROW_ID column is not present in exported df.
        """

        # 1) Arrange
        mock_define.return_value = (["periode", "kommuneregion"], ["utgifter", "flag", "tekst"])

        # 2) Act: build editor and access internal state
        get_results = dataframe_cell_editor_mvp(self.df, preview_rows=5, log_rows=None)

        closure = inspect.getclosurevars(get_results)
        df_working = closure.nonlocals["df_working"]
        row_id_col = closure.nonlocals["ROW_ID"]

        # 3) Assert: ROW_ID exists internally
        self.assertIn(row_id_col, df_working.columns)

        # 4) Assert: ROW_ID does not appear in exported results
        df_edited, _ = get_results()
        self.assertNotIn(row_id_col, df_edited.columns)


if __name__ == "__main__":
    unittest.main()
