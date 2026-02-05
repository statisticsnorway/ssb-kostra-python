# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: kostra-fellesfunksjoner
#     language: python
#     name: kostra-fellesfunksjoner
# ---

# %%
import getpass
from datetime import datetime

import ipywidgets as widgets
import numpy as np
import pandas as pd
from functions.funksjoner import hjelpefunksjoner
from IPython.display import clear_output
from IPython.display import display
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_extension_array_dtype
from pandas.api.types import is_float_dtype
from pandas.api.types import is_integer_dtype


# %%
def dataframe_cell_editor_mvp(
    df, *, preview_rows: int = 30, log_rows: int | None = None
):
    """Interactive, user-friendly dataframe cell editor for Jupyter notebooks.

    This function launches an ipywidgets-based UI that allows non-technical users
    to safely edit individual cell values in a pandas DataFrame, with strong
    guardrails to prevent accidental bulk edits and with full change logging.

    The workflow is strictly:
        1) Filter rows using exact-match filters on classification variables
        2) Review the filtered slice (editing is only allowed if ≤ 250 rows)
        3) Commit controlled cell edits with a required reason
        4) Automatically preview the updated dataframe and change log

    The original input dataframe is never modified. All edits are applied to an
    internal working copy.

    ------------------------------------------------------------------------
    Key features
    ------------------------------------------------------------------------
    - Designed for large dataframes (hundreds to millions of rows)
    - Exact-match, AND-only filtering on classification variables
    - Editing allowed only on small slices (≤ 250 rows)
    - Supports editing numeric, boolean, and string statistical variables
    - Safe type coercion based on column dtype (e.g. Int64, Float64, string)
    - Optional setting of missing values (NaN / pd.NA where supported)
    - Required reason for every committed edit
    - Full, row-level change log created at edit time
    - Automatic UI refresh after each commit:
        * filtered slice preview
        * edited dataframe preview
        * change log

    ------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------
    df : pandas.DataFrame
        The input dataframe to be edited. This dataframe is not modified.

    preview_rows : int, default 30
        Number of rows to display in the "Edited dataframe (preview)" panel.
        This is a preview only; the full edited dataframe is still available
        via the returned function.

    log_rows : int or None, default None
        If None, display the full change log in the UI.
        If an integer, display only the last `log_rows` entries in the UI.

    ------------------------------------------------------------------------
    User interface behavior
    ------------------------------------------------------------------------
    - Filtering is performed using text inputs with exact string matching.
    - Filters are combined using logical AND.
    - If the filter matches more than 250 rows, editing is blocked until the
      filter is narrowed.
    - Users may apply edits to:
        * all matched rows, or
        * a selected subset of rows within the filtered slice
    - Each edit operation modifies exactly one column per commit.
    - After each commit:
        * a confirmation message is shown
        * the preview table updates automatically
        * the edited dataframe preview refreshes
        * the change log refreshes

    ------------------------------------------------------------------------

    Returns:
    ------------------------------------------------------------------------
    get_results : callable
        A zero-argument function that returns the current edited dataframe
        and the full change log.

        Calling:
            df_edited, change_log_df = get_results()

        returns:
            df_edited : pandas.DataFrame
                The fully edited dataframe (original unchanged).

            change_log_df : pandas.DataFrame
                A dataframe containing one row per committed cell edit, including:
                    - timestamp
                    - user
                    - row identifier
                    - column edited
                    - old value
                    - new value
                    - reason
                    - classification variable snapshot

    ------------------------------------------------------------------------

    Notes:
    ------------------------------------------------------------------------
    - This function is intended for interactive use in Jupyter environments.
    - It relies on ipywidgets and a live kernel.
    - It builds upon `definere_klassifikasjonsvariable()` to identify
      classification vs statistical variables.
    - Persistence (saving edited data or logs to disk) is intentionally
      left out and can be added later if needed.


    ------------------------------------------------------------------------
    How to use
    ------------------------------------------------------------------------
    1. Open the interface with:
        - get_results = enkel_editering.dataframe_cell_editor_mvp(df_to_be_edited)
            - "df_to_be_edited" is the dataframe you want to edit
    2. Follow the instructions in the interface.
    3. Generate an edited dataframe (df_edited) and the log (change_log_df) for the changes you made.
        - df_edited, change_log_df = get_results()
        - display(df_edited)       <--- optional
        - display(change_log_df)   <--- optional
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    df_working = df.copy()

    ROW_ID = "__row_id__"
    if ROW_ID not in df_working.columns:
        df_working[ROW_ID] = np.arange(len(df_working))

    klassifikasjonsvariable, statistikkvariable = (
        hjelpefunksjoner.definere_klassifikasjonsvariable(df_working)
    )

    change_log: list[dict] = []
    MAX_EDIT_ROWS = 250

    # ------------------------------------------------------------------
    # Widgets: Filter UI
    # ------------------------------------------------------------------
    def make_filter_row(col):
        return widgets.Text(
            description=col,
            placeholder="exact value",
            layout=widgets.Layout(width="400px"),
        )

    filter_boxes = [make_filter_row(col) for col in klassifikasjonsvariable]

    apply_filter_btn = widgets.Button(
        description="Apply filter", button_style="primary"
    )
    filter_status = widgets.HTML()
    preview_out = widgets.Output()

    # ------------------------------------------------------------------
    # Widgets: Edit UI
    # ------------------------------------------------------------------
    edit_column_dd = widgets.Dropdown(
        options=statistikkvariable,
        description="Edit column:",
        layout=widgets.Layout(width="400px"),
    )

    new_value_box = widgets.Text(
        description="New value:", layout=widgets.Layout(width="400px")
    )

    set_nan_chk = widgets.Checkbox(value=False, description="Set to missing (NaN)")
    apply_all_chk = widgets.Checkbox(
        value=False, description="Apply to ALL matched rows"
    )

    row_select_box = widgets.SelectMultiple(
        options=[],
        description="Row IDs:",
        layout=widgets.Layout(width="400px", height="150px"),
    )

    reason_box = widgets.Textarea(
        description="Reason:",
        placeholder="Required",
        layout=widgets.Layout(width="600px", height="80px"),
    )

    commit_btn = widgets.Button(description="Commit edit", button_style="danger")
    edit_status = widgets.HTML()

    # ------------------------------------------------------------------
    # NEW: Live outputs for edited df + log
    # ------------------------------------------------------------------
    edited_df_out = widgets.Output()
    log_out = widgets.Output()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def coerce_value_for_column(col_name: str, raw_text: str, set_missing: bool):
        """Convert widget text input into a value compatible with df_working[col_name].dtype."""
        dtype = df_working[col_name].dtype
        text = (raw_text or "").strip()

        # Missing value handling
        if set_missing:
            if is_extension_array_dtype(dtype):
                return pd.NA
            if is_float_dtype(dtype):
                return np.nan
            if is_integer_dtype(dtype) or is_bool_dtype(dtype):
                raise ValueError(
                    f"Column '{col_name}' cannot be set to missing because it is non-nullable dtype '{dtype}'."
                )
            return np.nan

        if text == "":
            raise ValueError(
                "New value is empty. Type a value or tick 'Set to missing'."
            )

        if is_integer_dtype(dtype):
            try:
                f = float(text)
            except Exception:
                raise ValueError(
                    f"'{text}' is not a valid integer for column '{col_name}'."
                )
            if not f.is_integer():
                raise ValueError(
                    f"'{text}' is not a valid integer for column '{col_name}'."
                )
            return int(f)

        if is_float_dtype(dtype):
            try:
                return float(text)
            except Exception:
                raise ValueError(
                    f"'{text}' is not a valid number for column '{col_name}'."
                )

        if is_bool_dtype(dtype):
            t = text.lower()
            if t in ("true", "1", "yes", "y"):
                return True
            if t in ("false", "0", "no", "n"):
                return False
            raise ValueError(
                f"'{text}' is not a valid boolean (true/false/1/0/yes/no)."
            )

        return text

    def get_results():
        """Return the full edited df and the full change log df."""
        df_final = df_working.drop(columns=[ROW_ID])
        log_df = pd.DataFrame(change_log)
        return df_final, log_df

    def render_results():
        """Refresh the two live output panels."""
        df_final, log_df = get_results()

        with edited_df_out:
            clear_output()
            display(df_final.head(preview_rows))

        with log_out:
            clear_output()
            if log_df.empty:
                display(
                    pd.DataFrame(
                        columns=[
                            "timestamp",
                            "user",
                            "row_id",
                            "column",
                            "old_value",
                            "new_value",
                            "reason",
                        ]
                    )
                )
            else:
                display(log_df if log_rows is None else log_df.tail(log_rows))

    # Initial render (shows empty log + df head)
    render_results()

    # ------------------------------------------------------------------
    # Filtering logic
    # ------------------------------------------------------------------
    current_slice = None

    def apply_filter(_, clear_commit_msg: bool = True):
        nonlocal current_slice
        if clear_commit_msg:
            edit_status.value = ""

        mask = pd.Series(True, index=df_working.index)
        filter_desc = []

        for box in filter_boxes:
            if box.value.strip() != "":
                col = box.description
                val = box.value.strip()
                mask &= df_working[col].astype(str) == val
                filter_desc.append(f"{col} == {val}")

        current_slice = df_working.loc[mask]

        with preview_out:
            clear_output()

            n = len(current_slice)
            if n == 0:
                filter_status.value = "<b style='color:red'>No rows matched.</b>"
                row_select_box.options = []
                return

            if n > MAX_EDIT_ROWS:
                filter_status.value = (
                    f"<b style='color:red'>Matched {n} rows. "
                    f"Please narrow to ≤ {MAX_EDIT_ROWS}.</b>"
                )
                row_select_box.options = []
                return

            filter_status.value = (
                f"<b style='color:green'>Matched {n} rows.</b><br>"
                f"Filters: {' AND '.join(filter_desc) if filter_desc else '(none)'}"
            )

            display(current_slice.head(50))
            row_select_box.options = list(current_slice[ROW_ID])

    # Bind filter button (only once)
    apply_filter_btn.on_click(lambda btn: apply_filter(btn, clear_commit_msg=True))

    # ------------------------------------------------------------------
    # Commit logic
    # ------------------------------------------------------------------
    def commit_edit(_):
        nonlocal df_working

        if current_slice is None or len(current_slice) == 0:
            edit_status.value = "<b style='color:red'>Nothing to edit.</b>"
            return

        if reason_box.value.strip() == "":
            edit_status.value = "<b style='color:red'>Reason is required.</b>"
            return

        col = edit_column_dd.value

        if apply_all_chk.value:
            target_rows = current_slice[ROW_ID].tolist()
        else:
            target_rows = list(row_select_box.value)

        if len(target_rows) == 0:
            edit_status.value = "<b style='color:red'>No rows selected.</b>"
            return

        try:
            new_val = coerce_value_for_column(
                col, new_value_box.value, set_nan_chk.value
            )
        except Exception as e:
            edit_status.value = f"<b style='color:red'>{e}</b>"
            return

        any_change = False

        for rid in target_rows:
            idx = df_working.index[df_working[ROW_ID] == rid][0]
            old_val = df_working.at[idx, col]

            if (pd.isna(old_val) and pd.isna(new_val)) or (old_val == new_val):
                continue

            df_working.at[idx, col] = new_val
            any_change = True

            log_entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "user": getpass.getuser(),
                "row_id": int(rid),
                "column": col,
                "old_value": old_val,
                "new_value": new_val,
                "reason": reason_box.value,
            }

            for k in klassifikasjonsvariable:
                log_entry[f"id_{k}"] = df_working.at[idx, k]

            change_log.append(log_entry)

        if any_change:
            edit_status.value = "<b style='color:green'>Changes committed.</b>"
        else:
            edit_status.value = "<b style='color:orange'>No changes applied.</b>"

        # Refresh filter preview (without clearing commit msg)
        apply_filter(None, clear_commit_msg=False)

        # NEW: Refresh edited df + log panels
        render_results()

    commit_btn.on_click(commit_edit)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    filter_panel = widgets.VBox(filter_boxes + [apply_filter_btn, filter_status])

    edit_panel = widgets.VBox(
        [
            edit_column_dd,
            new_value_box,
            set_nan_chk,
            apply_all_chk,
            row_select_box,
            reason_box,
            commit_btn,
            edit_status,
        ]
    )

    # New right-hand side: preview/edit + live outputs
    right_panel = widgets.VBox(
        [
            widgets.HTML("<h3>Preview & Edit</h3>"),
            preview_out,
            edit_panel,
            widgets.HTML("<h3>Edited dataframe (preview)</h3>"),
            edited_df_out,
            widgets.HTML("<h3>Change log</h3>"),
            log_out,
        ]
    )

    ui = widgets.HBox(
        [widgets.VBox([widgets.HTML("<h3>Filters</h3>"), filter_panel]), right_panel]
    )

    display(ui)

    return get_results
