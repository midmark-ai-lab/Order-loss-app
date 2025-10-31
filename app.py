# # streamlit_app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# st.set_page_config(page_title="Order Loss Analysis", layout="wide")
# st.title("ORDER LOSS ANALYSIS")

# # -------------------------
# # Required defaults (flex)
# # -------------------------
# COL_MAP = {
#     "date": ["Exp Closing Date", "Date", "Closing Date"],
#     "amount": ["Amount", "Lost Value", "Loss Amount"],
#     # NEW: optional helpers for tooltips
#     "qty": ["Qty", "Quantity", "Units"],
#     "opp": ["Opportunity: Opportunity No", "Opp Id", "Order Id", "OrderID", "Orders"]
# }

# def pick_col(df, candidates):
#     for c in candidates:
#         if c in df.columns:
#             return c
#     return None

# def guess_categorical_columns(df, exclude: set[str]):
#     cats = []
#     for c in df.columns:
#         if c in exclude:
#             continue
#         s = df[c]
#         nunique = s.nunique(dropna=True)
#         if s.dtype == "object" or str(s.dtype).startswith("category") or nunique <= max(50, int(0.05 * len(df))):
#             cats.append(c)
#     return sorted(cats)

# # -------------------------
# # Upload
# # -------------------------
# uploaded = st.file_uploader("Upload your Order Loss file (CSV or Excel)", type=["csv", "xlsx"])
# if not uploaded:
#     st.info("Upload a file to begin. Required fields: a Date column and an Amount column.")
#     st.stop()

# # Read
# if uploaded.name.lower().endswith(".csv"):
#     df = pd.read_csv(uploaded)
# else:
#     df = pd.read_excel(uploaded)

# # -------------------------
# # Map required columns
# # -------------------------
# date_col = pick_col(df, COL_MAP["date"]) or st.selectbox("Select Date column", df.columns, index=None, placeholder="Pick a date column")
# amt_col  = pick_col(df, COL_MAP["amount"]) or st.selectbox("Select Amount (lost value) column", df.columns, index=None, placeholder="Pick an amount column")
# qty_col  = pick_col(df, COL_MAP["qty"])
# opp_col  = pick_col(df, COL_MAP["opp"])

# if not date_col or not amt_col:
#     st.error("Please select a Date and an Amount column.")
#     st.stop()

# # Clean
# df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
# df = df.dropna(subset=[date_col])
# df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
# if qty_col:
#     df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)

# # -------------------------
# # Dynamic LEVEL role selection
# # -------------------------
# exclude = {date_col, amt_col, qty_col, opp_col}
# exclude = {c for c in exclude if c is not None}
# cat_candidates = guess_categorical_columns(df, exclude)

# with st.expander("Dimension roles (pick any columns for each level)", expanded=True):
#     c1, c2, c3 = st.columns(3)
#     lvl1_col = c1.selectbox("Level 1 (top slicer)", options=cat_candidates, index=0 if cat_candidates else None, key="lvl1")
#     lvl2_col = c2.selectbox("Level 2 (bar breakdown)", options=["— None —"] + cat_candidates, index=1 if len(cat_candidates) > 1 else 0, key="lvl2")
#     lvl3_col = c3.selectbox("Level 3 (trend legend)", options=["— None —"] + cat_candidates, index=2 if len(cat_candidates) > 2 else 0, key="lvl3")

# lvl2_col = None if lvl2_col in (None, "— None —") else lvl2_col
# lvl3_col = None if lvl3_col in (None, "— None —") else lvl3_col

# dup_cols = [x for x in [lvl1_col, lvl2_col, lvl3_col] if x is not None]
# if len(set(dup_cols)) < len(dup_cols):
#     st.warning("You selected the same column in multiple levels. It works, but different columns give better insight.")

# # -------------------------
# # Period + YTD controls (for TREND)
# # -------------------------
# pc1, pc2 = st.columns([1, 0.5])
# period = pc1.segmented_control("PERIOD", options=["Monthly","Quarterly","Yearly"], default="Monthly")
# ytd = pc2.toggle("YTD", value=False, help="When on, clips each year to the latest date available for a fair YoY comparison.")

# if period == "Monthly":
#     rule, trend_title, periods_per_year = "MS", "ORDER LOSS TREND (MONTHLY)", 12
# elif period == "Quarterly":
#     rule, trend_title, periods_per_year = "QS", "ORDER LOSS TREND (QUARTERLY)", 4
# else:
#     rule, trend_title, periods_per_year = "YS", "ORDER LOSS TREND (YEARLY)", 1

# # -------------------------
# # Cascading filters
# # -------------------------
# if lvl1_col:
#     lvl1_vals_all = sorted(df[lvl1_col].dropna().unique().tolist())
#     sel_lvl1 = st.multiselect(f"{lvl1_col.upper()} (Level 1)", lvl1_vals_all, default=lvl1_vals_all)
#     df1 = df[df[lvl1_col].isin(sel_lvl1)]
# else:
#     df1 = df.copy()

# if lvl2_col:
#     lvl2_vals_all = sorted(df1[lvl2_col].dropna().unique().tolist())
#     sel_lvl2 = st.multiselect(f"{lvl2_col.upper()} (Level 2)", lvl2_vals_all, default=lvl2_vals_all)
#     df2 = df1[df1[lvl2_col].isin(sel_lvl2)]
# else:
#     df2 = df1.copy()

# if lvl3_col:
#     lvl3_vals_all = sorted(df2[lvl3_col].dropna().unique().tolist())
#     sel_lvl3 = st.multiselect(f"{lvl3_col.upper()} (Level 3)", lvl3_vals_all, default=lvl3_vals_all)
#     d = df2[df2[lvl3_col].isin(sel_lvl3)]
# else:
#     d = df2.copy()

# # -------------------------
# # BAR: Year selector applies only to BAR
# # -------------------------
# years_avail = sorted(d[date_col].dt.year.unique().tolist())
# year_choice = st.selectbox("YEAR for bar (optional)", options=["All"] + years_avail, index=len(years_avail), help="Affects bar only, not the trend.")
# bar_df = d if year_choice == "All" else d[d[date_col].dt.year == year_choice]

# topn = st.number_input("TOP N OPTIONS (for bar chart)", min_value=3, max_value=20, value=5, step=1)

# # -------------------------
# # CHARTS
# # -------------------------
# left, right = st.columns(2)

# # ---- BAR: group by Level-2 else Level-1; add Qty and Unique Opp in hover ----
# bar_dim = lvl2_col or lvl1_col
# if bar_dim:
#     gb = bar_df.groupby(bar_dim, dropna=False)
#     agg = gb[amt_col].sum().rename("LostValue")
#     if qty_col:
#         qty_agg = gb[qty_col].sum().rename("QtySum")
#     else:
#         qty_agg = pd.Series(index=agg.index, data=np.nan, name="QtySum")
#     if opp_col and opp_col in bar_df.columns:
#         opp_agg = gb[opp_col].nunique().rename("OppUnique")
#     else:
#         opp_agg = pd.Series(index=agg.index, data=np.nan, name="OppUnique")

#     top_table = (
#         pd.concat([agg, qty_agg, opp_agg], axis=1)
#         .sort_values("LostValue", ascending=False)
#         .head(int(topn))
#         .reset_index()
#     )

#     if not top_table.empty:
#         fig_bar = px.bar(
#             top_table,
#             x=bar_dim, y="LostValue",
#             labels={bar_dim: bar_dim, "LostValue": "Lost Revenue ($)"},
#             title=f"TOP {int(topn)} ORDER LOSS BY {bar_dim.upper()}",
#             text_auto=".2s"
#         )
#         # Custom hover: value + qty + unique opp
#         fig_bar.update_traces(
#             hovertemplate=(
#                 f"<b>{bar_dim}</b>: %{ {'x'} }<br>"
#                 "Lost Value: %{y:,.0f}<br>"
#                 "Qty: %{customdata[0]}<br>"
#                 "Unique Opp: %{customdata[1]}<extra></extra>"
#             ),
#             customdata=top_table[["QtySum", "OppUnique"]].to_numpy()
#         )
#         fig_bar.update_layout(margin=dict(l=10,r=10,b=10,t=50))
#         left.plotly_chart(fig_bar, use_container_width=True)
#     else:
#         left.info("No data for the selected filters/year.")
# else:
#     left.info("Select at least one Level column to draw the bar chart.")

# # ---- TREND: apply YTD clipping and show YoY% in hover ----
# legend_dim = lvl3_col or lvl2_col or lvl1_col

# trend_base = d.copy()
# if ytd and period in ("Monthly", "Quarterly"):
#     max_dt = d[date_col].max()
#     cutoff_month, cutoff_day = max_dt.month, max_dt.day
#     trend_base = trend_base[
#         (trend_base[date_col].dt.month < cutoff_month)
#         | ((trend_base[date_col].dt.month == cutoff_month) & (trend_base[date_col].dt.day <= cutoff_day))
#     ]

# # Aggregate by time (and legend if present)
# groupers = [pd.Grouper(key=date_col, freq=rule)]
# if legend_dim:
#     groupers.append(legend_dim)

# trend = trend_base.groupby(groupers)[amt_col].sum().reset_index().sort_values(date_col)

# # Compute YoY% vs previous year (same period index) within each legend group
# if not trend.empty:
#     if legend_dim:
#         trend["prev"] = trend.groupby(legend_dim)[amt_col].shift(periods_per_year)
#     else:
#         trend["prev"] = trend[amt_col].shift(periods_per_year)
#     trend["YoY_pct"] = np.where(trend["prev"] > 0, (trend[amt_col] - trend["prev"]) / trend["prev"] * 100.0, np.nan)

#     if legend_dim:
#         fig_line = px.line(
#             trend, x=date_col, y=amt_col, color=legend_dim,
#             labels={date_col: "Period", amt_col: "Lost Revenue ($)", legend_dim: legend_dim},
#             title=trend_title
#         )
#     else:
#         fig_line = px.line(
#             trend, x=date_col, y=amt_col,
#             labels={date_col: "Period", amt_col: "Lost Revenue ($)"},
#             title=trend_title
#         )

#     # Attach YoY% to hover
#     # customdata needs to be aligned to traces; easiest is to map per-trace
#     if legend_dim:
#         for tr_name in trend[legend_dim].dropna().unique().tolist():
#             mask = trend[legend_dim] == tr_name
#             cd = trend.loc[mask, ["YoY_pct"]].to_numpy()
#             fig_line.for_each_trace(
#                 lambda t: t.update(
#                     customdata=cd,
#                     hovertemplate="%{x|%b %Y}<br>Value: %{y:,.0f}<br>YoY: %{customdata[0]:.1f}%<extra></extra>"
#                 ) if t.name == str(tr_name) else ()
#             )
#         # Handle NaNs legend categories if any
#         if trend[legend_dim].isna().any():
#             mask = trend[legend_dim].isna()
#             cd = trend.loc[mask, ["YoY_pct"]].to_numpy()
#             fig_line.for_each_trace(
#                 lambda t: t.update(
#                     customdata=cd,
#                     hovertemplate="%{x|%b %Y}<br>Value: %{y:,.0f}<br>YoY: %{customdata[0]:.1f}%<extra></extra>"
#                 ) if t.name == "nan" else ()
#             )
#     else:
#         cd = trend[["YoY_pct"]].to_numpy()
#         fig_line.update_traces(
#             customdata=cd,
#             hovertemplate="%{x|%b %Y}<br>Value: %{y:,.0f}<br>YoY: %{customdata[0]:.1f}%<extra></extra>"
#         )

#     fig_line.update_layout(margin=dict(l=10,r=10,b=10,t=50))
#     right.plotly_chart(fig_line, use_container_width=True)
# else:
#     right.info("No data for the selected filters or period.")

# # -------------------------
# # KPIs + export
# # -------------------------
# total_lost = float(d[amt_col].sum())
# total_rows = len(d)
# k1, k2 = st.columns(2)
# k1.metric("TOTAL LOST REVENUE", f"${total_lost:,.0f}")
# k2.metric("ROWS (proxy for orders if no ID)", f"{total_rows:,}")

# st.download_button(
#     "Download filtered data (CSV)",
#     data=d.to_csv(index=False).encode("utf-8"),
#     file_name="order_loss_filtered.csv",
#     mime="text/csv"
# )

# # -------------------------
# # Optional PPT hook
# # -------------------------
# with st.expander("Generate PPT (optional)"):
#     st.caption("Wire your PPT generator here if you want one-click slides.")
#     if st.button("Generate PPT from filtered data"):
#         try:
#             st.warning("Placeholder: connect to your PPT function.")
#         except Exception as e:
#             st.error(f"PPT generation failed: {e}")


# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Order Loss Analysis", layout="wide")
st.title("ORDER LOSS ANALYSIS")

# -------------------------
# Column mapping (flex)
# -------------------------
COL_MAP = {
    "date":   ["Exp Closing Date", "Date", "Closing Date"],
    "amount": ["Amount", "Lost Value", "Loss Amount"],
    "qty":    ["Qty", "Quantity", "Units"],
    "opp":    ["Opportunity: Opportunity No", "Opp Id", "Order Id", "OrderID", "Orders"]
}

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def guess_categorical_columns(df, exclude: set[str]):
    cats = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        nunique = s.nunique(dropna=True)
        if s.dtype == "object" or str(s.dtype).startswith("category") or nunique <= max(50, int(0.05 * len(df))):
            cats.append(c)
    return sorted(cats)

def apply_ytd_clip(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Clip each calendar year to the same month/day as the latest date in the filtered set."""
    if frame.empty:
        return frame
    max_dt = pd.to_datetime(frame[date_col].max())
    cut_m, cut_d = int(max_dt.month), int(max_dt.day)
    y = frame[date_col].dt.year
    per_year_cutoff = pd.to_datetime(dict(year=y, month=cut_m, day=cut_d), errors="coerce")
    return frame[frame[date_col] <= per_year_cutoff]

# -------------------------
# Upload
# -------------------------
uploaded = st.file_uploader("Upload your Order Loss file (CSV or Excel)", type=["csv", "xlsx"])
if not uploaded:
    st.info("Upload a file to begin. Required fields: a Date column and an Amount column.")
    st.stop()

# Read
if uploaded.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

# -------------------------
# Map required columns
# -------------------------
date_col = pick_col(df, COL_MAP["date"]) \
    or st.selectbox("Select Date column", df.columns, index=None, placeholder="Pick a date column")
amt_col  = pick_col(df, COL_MAP["amount"]) \
    or st.selectbox("Select Amount (lost value) column", df.columns, index=None, placeholder="Pick an amount column")
qty_col  = pick_col(df, COL_MAP["qty"])
opp_col  = pick_col(df, COL_MAP["opp"])

if not date_col or not amt_col:
    st.error("Please select a Date and an Amount column.")
    st.stop()

# Clean
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
if qty_col:
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)

# -------------------------
# Dynamic LEVEL role selection
# -------------------------
exclude = {date_col, amt_col, qty_col, opp_col}
exclude = {c for c in exclude if c is not None}
cat_candidates = guess_categorical_columns(df, exclude)

with st.expander("Dimension roles (pick any columns for each level)", expanded=True):
    c1, c2, c3 = st.columns(3)
    lvl1_col = c1.selectbox(
        "Level 1 (top slicer)", options=cat_candidates,
        index=0 if cat_candidates else None, key="lvl1"
    )
    lvl2_col = c2.selectbox(
        "Level 2 (bar breakdown)", options=["— None —"] + cat_candidates,
        index=1 if len(cat_candidates) > 1 else 0, key="lvl2"
    )
    lvl3_col = c3.selectbox(
        "Level 3 (trend legend) — disabled", options=["— None —"] + cat_candidates,
        index=0, key="lvl3"
    )

# Force legend OFF for trend (per your requirement)
lvl2_col = None if lvl2_col in (None, "— None —") else lvl2_col
lvl3_col = None  # hard-disable legend in trend chart

dup_cols = [x for x in [lvl1_col, lvl2_col, lvl3_col] if x is not None]
if len(set(dup_cols)) < len(dup_cols):
    st.warning("You selected the same column in multiple levels. It works, but different columns give better insight.")

# -------------------------
# Period + YTD controls (for TREND)
# -------------------------
pc1, pc2 = st.columns([1, 0.6])
# Fall back if segmented_control is unavailable on your Streamlit version
# period = getattr(st, "segmented_control", st.radio)(
#     "PERIOD",
#     options=["Monthly", "Quarterly", "Yearly"],
#     index=0 if getattr(st, "segmented_control", None) else 0
# )

# PERIOD selector (supports older Streamlit versions)
if hasattr(st, "segmented_control"):
    period = st.segmented_control(
        "PERIOD",
        options=["Monthly", "Quarterly", "Yearly"],
        default="Monthly"
    )
else:
    period = st.radio(
        "PERIOD",
        options=["Monthly", "Quarterly", "Yearly"],
        index=0
    )

ytd = pc2.toggle("YTD", value=False, help="Clips each year to the latest date available for a fair YoY comparison.")

if period == "Monthly":
    rule, trend_title, periods_per_year = "MS", "ORDER LOSS TREND (MONTHLY)", 12
elif period == "Quarterly":
    rule, trend_title, periods_per_year = "QS", "ORDER LOSS TREND (QUARTERLY)", 4
else:
    rule, trend_title, periods_per_year = "YS", "ORDER LOSS TREND (YEARLY)", 1

# -------------------------
# Cascading filters
# -------------------------
if lvl1_col:
    lvl1_vals_all = sorted(df[lvl1_col].dropna().unique().tolist())
    default_lvl1 = lvl1_vals_all if lvl1_vals_all else []
    sel_lvl1 = st.multiselect(f"{lvl1_col.upper()} (Level 1)", lvl1_vals_all, default=default_lvl1)
    df1 = df[df[lvl1_col].isin(sel_lvl1)] if sel_lvl1 else df.iloc[0:0]
else:
    df1 = df.copy()

if lvl2_col:
    lvl2_vals_all = sorted(df1[lvl2_col].dropna().unique().tolist())
    default_lvl2 = lvl2_vals_all if lvl2_vals_all else []
    sel_lvl2 = st.multiselect(f"{lvl2_col.upper()} (Level 2)", lvl2_vals_all, default=default_lvl2)
    df2 = df1[df1[lvl2_col].isin(sel_lvl2)] if sel_lvl2 else df1.iloc[0:0]
else:
    df2 = df1.copy()

# Level 3 selector kept for UI symmetry, but not used in trend legend
if lvl3_col:
    lvl3_vals_all = sorted(df2[lvl3_col].dropna().unique().tolist())
    default_lvl3 = lvl3_vals_all if lvl3_vals_all else []
    sel_lvl3 = st.multiselect(f"{lvl3_col.upper()} (Level 3)", lvl3_vals_all, default=default_lvl3)
    d = df2[df2[lvl3_col].isin(sel_lvl3)] if sel_lvl3 else df2.iloc[0:0]
else:
    d = df2.copy()

# -------------------------
# BAR: Year selector applies only to BAR
# -------------------------
if d.empty:
    years_avail = []
else:
    years_avail = sorted(d[date_col].dt.year.dropna().unique().tolist())

year_opts = ["All"] + years_avail
default_idx = len(year_opts) - 1 if len(year_opts) > 1 else 0
year_choice = st.selectbox("YEAR for bar (optional)", options=year_opts, index=default_idx,
                           help="Affects bar only, not the trend.")
bar_df = d if year_choice == "All" else d[d[date_col].dt.year == year_choice]

topn = int(st.number_input("TOP N OPTIONS (for bar chart)", min_value=3, max_value=20, value=5, step=1))

# -------------------------
# CHARTS
# -------------------------
left, right = st.columns(2)

# ---- BAR: group by Level-2 else Level-1; add Qty and Unique Opp in hover ----
bar_dim = lvl2_col or lvl1_col
if bar_dim:
    gb = bar_df.groupby(bar_dim, dropna=False)
    agg = gb[amt_col].sum().rename("LostValue")
    qty_agg = gb[qty_col].sum().rename("QtySum") if qty_col else pd.Series(index=agg.index, data=np.nan, name="QtySum")
    if opp_col and opp_col in bar_df.columns:
        opp_agg = gb[opp_col].nunique().rename("OppUnique")
    else:
        opp_agg = pd.Series(index=agg.index, data=np.nan, name="OppUnique")

    top_table = (
        pd.concat([agg, qty_agg, opp_agg], axis=1)
        .sort_values("LostValue", ascending=False)
        .head(topn)
        .reset_index()
    )

    if not top_table.empty:
        fig_bar = px.bar(
            top_table,
            x=bar_dim, y="LostValue",
            labels={bar_dim: bar_dim, "LostValue": "Lost Revenue ($)"},
            title=f"TOP {topn} ORDER LOSS BY {bar_dim.upper()}",
            text_auto=".2s"
        )
        # Proper hover with %{x}
        fig_bar.update_traces(
            hovertemplate="<b>%{x}</b><br>"
                          "Lost Value: %{y:,.0f}<br>"
                          "Qty: %{customdata[0]}<br>"
                          "Unique Opp: %{customdata[1]}<extra></extra>",
            customdata=top_table[["QtySum", "OppUnique"]].to_numpy()
        )
        fig_bar.update_layout(margin=dict(l=10, r=10, b=10, t=50))
        left.plotly_chart(fig_bar, use_container_width=True)
    else:
        left.info("No data for the selected filters/year.")
else:
    left.info("Select at least one Level column to draw the bar chart.")

# ---- TREND: single line, YTD-aware, YoY hover ----
trend_base = d.copy()
if ytd and not trend_base.empty and period in ("Monthly", "Quarterly", "Yearly"):
    trend_base = apply_ytd_clip(trend_base, date_col)

# Aggregate by time only (legend is disabled)
groupers = [pd.Grouper(key=date_col, freq=rule)]
trend = (
    trend_base
    .groupby(groupers, dropna=False)[amt_col]
    .sum()
    .reset_index()
    .sort_values(date_col)
)

if not trend.empty:
    trend["prev"] = trend[amt_col].shift(periods_per_year)
    trend["YoY_pct"] = np.where(trend["prev"] > 0,
                                (trend[amt_col] - trend["prev"]) / trend["prev"] * 100.0,
                                np.nan)

    fig_line = px.line(
        trend, x=date_col, y=amt_col,
        labels={date_col: "Period", amt_col: "Lost Revenue ($)"},
        title=trend_title
    )
    fig_line.update_traces(
        customdata=trend[["YoY_pct"]].to_numpy(),
        hovertemplate="%{x|%b %Y}<br>Value: %{y:,.0f}<br>YoY: %{customdata[0]:.1f}%<extra></extra>"
    )
    fig_line.update_layout(margin=dict(l=10, r=10, b=10, t=50), showlegend=False)
    right.plotly_chart(fig_line, use_container_width=True)
else:
    right.info("No data for the selected filters or period.")

# -------------------------
# KPIs + export
# -------------------------
total_lost = float(d[amt_col].sum()) if not d.empty else 0.0
total_rows = int(len(d))
k1, k2 = st.columns(2)
k1.metric("TOTAL LOST REVENUE", f"${total_lost:,.0f}")
k2.metric("ROWS (proxy for orders if no ID)", f"{total_rows:,}")

st.download_button(
    "Download filtered data (CSV)",
    data=d.to_csv(index=False).encode("utf-8"),
    file_name="order_loss_filtered.csv",
    mime="text/csv",
    disabled=d.empty
)

# -------------------------
# Optional PPT hook
# -------------------------
with st.expander("Generate PPT (optional)"):
    st.caption("Wire your PPT generator here if you want one-click slides.")
    if st.button("Generate PPT from filtered data"):
        try:
            # TODO: call your PPT function with 'd' and other selections
            st.warning("Placeholder: connect to your PPT function.")
        except Exception as e:
            st.error(f"PPT generation failed: {e}")
