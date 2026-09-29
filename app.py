"""DDM Dashboard — hit rates & RT distributions with interactive filters.

Run:  uv run bokeh serve --show app.py
"""

import polars as pl
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.palettes import Category10
from bokeh.models import Button, Div, MultiSelect, Select, Spacer

from behaviz.backends.renderer_manager import set_renderer
from piepy.viz.plots import psychometric, reaction_time_dist

# ── renderer ──────────────────────────────────────────────────────────
set_renderer("bokeh")

# ── data (lazy scan) ──────────────────────────────────────────────────
DATA_PATH = "data/260923_training_data.parquet"
LF = pl.scan_parquet(DATA_PATH)
# remove all the earlies from the start
LF = LF.filter(pl.col("outcome") != "early")

# Collect unique values for widgets once (fast: only touches metadata + single cols)
_OPTS_CACHE: dict[str, list[str]] = {}


def _unique_opts(col: str) -> list[str]:
    if col not in _OPTS_CACHE:
        vals = LF.select(pl.col(col)).unique().collect().to_series().drop_nulls().sort().to_list()
        opts = [str(v) for v in vals]
        has_nulls = LF.select(pl.col(col).is_null().any()).collect().item()
        if has_nulls:
            opts.append("None")
        _OPTS_CACHE[col] = opts
    return _OPTS_CACHE[col]


# ── helpers ───────────────────────────────────────────────────────────
BOOL_COLS = {"isCNO", "opto"}
FLOAT_COLS = {"contrast"}


def _filter_expr(col: str, selected: list[str]):
    include_null = "None" in selected
    clean = [v for v in selected if v != "None"]

    if col in FLOAT_COLS:
        typed = [float(v) for v in clean]
    elif col in BOOL_COLS:
        typed = [v == "True" for v in clean]
    else:
        typed = clean

    expr = pl.col(col).is_in(typed)
    if include_null:
        expr = expr | pl.col(col).is_null()
    return expr


# ── widgets ───────────────────────────────────────────────────────────
contrast_w = MultiSelect(title="Contrast", value=[], options=_unique_opts("contrast"), size=8, width=120)
isCNO_w = MultiSelect(title="isCNO", value=[], options=_unique_opts("isCNO"), size=3, width=80)
opto_w = MultiSelect(title="Opto", value=[], options=_unique_opts("opto"), size=3, width=80)
stim_type_w = MultiSelect(title="Stim type", value=[], options=_unique_opts("stim_type"), size=8, width=150)
area_w = MultiSelect(
    title="Area (opto only)",
    value=[],
    options=_unique_opts("area"),
    size=8,
    width=140,
    disabled=True,
)
stim_combo_w = MultiSelect(title="Stim combo", value=[], options=_unique_opts("stim_combination"), size=8, width=180)
animalid_w = MultiSelect(title="Animal", value=[], options=_unique_opts("animalid"), size=8, width=120)

compare_w = Select(title="Compare by", value="none", options=["none"], width=120)

compare_vals_w = MultiSelect(
    title="Compare values (pick 2)",
    value=[],
    options=[],
    size=8,
    width=150,
    disabled=True,
)

scope_w = Select(
    title="Scope",
    value="pooled",
    options=["pooled", "animalid", "session_uid"],
    width=120,
)

plot_btn = Button(label="PLOT", button_type="primary", width=100, height=40)

FILTER_WIDGETS: dict[str, MultiSelect] = {
    "contrast": contrast_w,
    "isCNO": isCNO_w,
    "opto": opto_w,
    "stim_type": stim_type_w,
    "stim_combination": stim_combo_w,
    "area": area_w,
    "animalid": animalid_w,
}

status_div = Div(text="", styles={"font-size": "13px", "color": "#888"})
summary_div = Div(text="", styles={"font-size": "13px", "color": "#555"})
plot_container = column(Div(text="<i>Select filters and press PLOT</i>"), sizing_mode="stretch_both")


# ── lazy filter builder ───────────────────────────────────────────────
def _build_lazy_query() -> pl.LazyFrame:
    """Build a lazy query with all active filters pushed down."""
    lf = LF
    for name, w in FILTER_WIDGETS.items():
        if name == "area" and area_w.disabled:
            continue
        if not w.value:
            continue
        lf = lf.filter(_filter_expr(name, w.value))
    return lf


# ── widget-only updates (no collect, instant) ─────────────────────────
def _compare_candidates() -> list[str]:
    cands = ["none"]
    for name, w in FILTER_WIDGETS.items():
        if name == "area" and area_w.disabled:
            continue
        if not w.value:
            cands.append(name)
    return cands


def _build_summary() -> str:
    parts = []

    if animalid_w.value:
        parts.append(f"<b>{', '.join(animalid_w.value)}</b>")
    else:
        parts.append("<b>ALL</b> animals")

    if stim_type_w.value:
        parts.append(f"for <b>{', '.join(stim_type_w.value)}</b> stimuli")

    if stim_combo_w.value:
        parts.append(f"combo <b>{', '.join(stim_combo_w.value)}</b>")

    if contrast_w.value:
        parts.append(f"at contrast <b>{', '.join(contrast_w.value)}</b>")

    if opto_w.value == ["True"]:
        parts.append("for <b>Opto</b> trials")
    elif opto_w.value == ["False"]:
        parts.append("for <b>Non-opto</b> trials")

    if not area_w.disabled and area_w.value:
        parts.append(f"targeting <b>{', '.join(area_w.value)}</b>")

    if isCNO_w.value == ["True"]:
        parts.append("with <b>CNO</b> injection")
    elif isCNO_w.value == ["False"]:
        parts.append("<b>without CNO</b>")

    cmp = compare_w.value
    if cmp != "none":
        if compare_vals_w.value:
            parts.append(f"— comparing <b>{cmp}</b>: {', '.join(compare_vals_w.value)}")
        else:
            parts.append(f"— comparing by <b>{cmp}</b>")

    return "Showing " + " ".join(parts)


def _on_widget_change(attr, old, new):
    """Instant: update widget states + summary. No data collected."""
    # Gate area
    area_w.disabled = opto_w.value != ["True"]
    if area_w.disabled:
        area_w.value = []

    # Refresh compare-by
    cands = _compare_candidates()
    compare_w.options = cands
    if compare_w.value not in cands:
        compare_w.value = "none"

    # Refresh compare-values
    cmp_col = compare_w.value if compare_w.value != "none" else None
    if cmp_col:
        # ponytail: this one small collect is fast — single column unique
        lf = _build_lazy_query()
        opts = [str(v) for v in lf.select(pl.col(cmp_col)).unique().collect().to_series().drop_nulls().sort().to_list()]
        compare_vals_w.options = opts
        compare_vals_w.disabled = False
        compare_vals_w.value = [v for v in compare_vals_w.value if v in opts]
    else:
        compare_vals_w.options = []
        compare_vals_w.value = []
        compare_vals_w.disabled = True

    summary_div.text = _build_summary()


# ── PLOT button: collect + render ─────────────────────────────────────
def _on_plot():
    plot_btn.label = "Plotting..."
    plot_btn.disabled = True
    try:
        cmp_col = compare_w.value if compare_w.value != "none" else None

        # Collect filtered data
        lf = _build_lazy_query()
        if cmp_col and compare_vals_w.value:
            lf = lf.filter(_filter_expr(cmp_col, compare_vals_w.value))

        filtered = lf.collect()

        if filtered.height == 0:
            plot_container.children = [Div(text="<b>No data matches current filters.</b>")]
            status_div.text = "0 trials"
            return

        n_animals = filtered["animalid"].n_unique()
        n_sessions = filtered["session_uid"].n_unique()
        status_div.text = f"{filtered.height:,} trials · {n_animals} animals · {n_sessions} sessions"

        avg_over = None if scope_w.value == "pooled" else scope_w.value
        _cmp_list = filtered[cmp_col].drop_nulls().unique(maintain_order=True).to_list()
        palette = {c: Category10[10][i % 10] for i, c in enumerate(_cmp_list)}

        # Psychometric
        psych_cmp = cmp_col if cmp_col and cmp_col != "contrast" else None
        res_psych = psychometric(
            filtered,
            x="signed_contrast",
            compare=psych_cmp,
            average_over=avg_over,
            fit_curve=False,
            palette=palette,
            preset="ddm-dashboard/psychometric",
        )
        fig_psych, _ = res_psych.figure
        fig_psych.sizing_mode = "stretch_both"
        fig_psych.styles = {"flex": "2"}
        for legend in fig_psych.legend:
            legend.click_policy = "mute"

        # RT dist
        rt_df = filtered.filter((pl.col("outcome") == "hit") & pl.col("reaction_time").is_not_null())
        if rt_df.height == 0:
            fig_rt = Div(text="<b>No hit trials with valid RT.</b>")
        else:
            res_rt = reaction_time_dist(
                rt_df,
                value="reaction_time",
                comparing=cmp_col,
                average_over=avg_over,
                palette=palette,
                preset="ddm-dashboard/reaction_distribution",
            )
            fig_rt, _ = res_rt.figure
            fig_rt.sizing_mode = "stretch_both"
            fig_rt.styles = {"flex": "3"}
            for legend in fig_rt.legend:
                legend.click_policy = "mute"

        plot_container.children = [row(fig_psych, fig_rt, sizing_mode="stretch_both")]
    except Exception as e:
        plot_container.children = [Div(text=f"<b>Plot error:</b><pre>{e}</pre>")]
    finally:
        plot_btn.label = "PLOT"
        plot_btn.disabled = False


# ── wire callbacks ────────────────────────────────────────────────────
for w in [
    contrast_w,
    isCNO_w,
    opto_w,
    stim_type_w,
    stim_combo_w,
    area_w,
    animalid_w,
    compare_w,
    compare_vals_w,
    scope_w,
]:
    w.on_change("value", _on_widget_change)

plot_btn.on_click(_on_plot)

# ── layout ────────────────────────────────────────────────────────────
title_div = Div(
    text="<b style='font-size:18px'>DDM Dashboard</b>"
    "&nbsp;&nbsp;<span style='color:#888;font-size:12px'>empty selection = all</span>",
)

filter_row = row(
    contrast_w,
    isCNO_w,
    opto_w,
    stim_type_w,
    stim_combo_w,
    area_w,
    animalid_w,
    Spacer(width=30),
    compare_w,
    compare_vals_w,
    Spacer(width=30),
    scope_w,
    sizing_mode="stretch_width",
)

summary_row = row(summary_div, Spacer(width=20), plot_btn, Spacer(width=10), status_div)

root = column(
    title_div,
    filter_row,
    summary_row,
    plot_container,
    sizing_mode="stretch_both",
)

doc = curdoc()
doc.add_root(root)
doc.title = "DDM Dashboard"

# Initial widget state (no plot until button pressed)
_on_widget_change(None, None, None)
