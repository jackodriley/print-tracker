import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import re
import pydeck as pdk

st.set_page_config(page_title="Observer Retail Sales – Week Change Reporter", layout="wide")
st.title("🗞️ Observer Retail Sales – Week Change Reporter")
st.caption("Upload an XLS/XLSX/CSV where weekly sales are in **columns** (date headers). The app finds the latest two weeks, and shows changes by location.")

# -----------------------------
# Helpers
# -----------------------------

DATE_FMT_DISPLAY = "%Y-%m-%d"

KNOWN_META = [
    "Box No", "STORE NAME", "MULTIPLE CATEGORY", "ADDRESS LINE 1", "ADDRESS LINE 2",
    "POSTCODE", "WHOLESALER"
]

def detect_week_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of columns that look like weekly date columns (parseable)."""
    week_cols = []
    for c in df.columns:
        # Ignore obvious meta columns
        if str(c).strip() in KNOWN_META:
            continue
        # Try parse as datetime; also allow pandas Timestamp objects
        try:
            _ = pd.to_datetime(c)
            week_cols.append(c)
        except Exception:
            # Some files might store datetimes as excel serials that pandas already parsed; also allow numeric that are all-NaN/ints
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                week_cols.append(c)
    return week_cols

def melt_wide(df: pd.DataFrame, meta_cols: List[str], week_cols: List[str]) -> pd.DataFrame:
    """Melt wide weekly columns into tidy rows: meta + week + units."""
    work = df.copy()
    # Ensure week columns are numeric
    for w in week_cols:
        work[w] = pd.to_numeric(work[w], errors="coerce")
    tidy = work.melt(id_vars=meta_cols, value_vars=week_cols,
                     var_name="_week_col_raw", value_name="_units")
    # Parse the week label to a normalized YYYY-MM-DD (date) string for sorting
    tidy["_week_dt"] = pd.to_datetime(tidy["_week_col_raw"], errors="coerce")
    tidy["_week"] = tidy["_week_dt"].dt.strftime(DATE_FMT_DISPLAY)
    return tidy

def choose_two_most_recent(weeks: List[str]) -> Tuple[str, str]:
    """Given a list of YYYY-MM-DD strings, return (prev, curr)."""
    ws = sorted([w for w in weeks if isinstance(w, str)])
    if len(ws) < 2:
        return None, None
    return ws[-2], ws[-1]

# Outward postcode helper
OUTWARD_RE = re.compile(r"^[A-Z]{1,2}\d{1,2}[A-Z]?", re.IGNORECASE)

def to_outward(postcode: str) -> str:
    """Return outward code e.g. 'SK4' from 'SK4 4DF'. Falls back to token before the space."""
    if not isinstance(postcode, str):
        return ""
    pc = postcode.strip().upper()
    # Prefer the clear token before the first space if present
    if " " in pc:
        return pc.split()[0]
    # Otherwise, try a compact match
    m = OUTWARD_RE.match(pc)
    return m.group(0) if m else pc

def build_location_label(row, primary_choice: str, add_postcode: bool):
    base = str(row.get(primary_choice, "")).strip()
    if add_postcode:
        pc = str(row.get("POSTCODE", "")).strip()
        if pc and pc.lower() != "(blank)":
            return f"{base} [{pc}]"
    return base or "(Unnamed)"

def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
    return df_in.to_csv(index=False).encode("utf-8")

# -----------------------------
# Sidebar: upload + options
# -----------------------------
with st.sidebar:
    st.header("1) Upload data")
    uploaded = st.file_uploader("Drop XLS/XLSX/CSV here", type=["xls", "xlsx", "csv"])

    st.divider()
    st.header("2) Column options")

    # Defaults lined up with your sample file
    default_primary = "STORE NAME"
    default_group = "WHOLESALER"

    st.caption("Pick how locations are labelled and optionally grouped.")
    primary_label = st.selectbox("Primary location label", options=KNOWN_META, index=KNOWN_META.index(default_primary))
    add_postcode = st.checkbox("Append POSTCODE to label", value=True)
    group_by = st.selectbox("Group by (optional)", options=["— none —"] + KNOWN_META, index=(KNOWN_META.index(default_group)+1))

    st.divider()
    st.header("3) Filters")
    min_prev_units = st.number_input("Minimum prior week units to include", min_value=0, value=10, step=5)
    min_abs_change = st.number_input("Minimum absolute change (Δ units) to include", min_value=0, value=0, step=5)
    z_threshold = st.number_input("Outlier z-score threshold (|z| ≥)", min_value=0.0, value=2.0, step=0.5)
    top_n = st.number_input("Top N for winners/decliners", min_value=5, value=20, step=5)

if uploaded is None:
    st.info("Upload your sales file to begin. Use the sample format where weekly dates are column headers.")
    st.stop()

# -----------------------------
# Load file
# -----------------------------
if uploaded.name.lower().endswith(".csv"):
    raw = pd.read_csv(uploaded)
else:
    # Allow the user to choose a sheet if multiple exist
    try:
        xls = pd.ExcelFile(uploaded)
        sheet_names = xls.sheet_names
        if len(sheet_names) > 1:
            sheet = st.sidebar.selectbox("Sheet", options=sheet_names, index=0)
        else:
            sheet = sheet_names[0]
        raw = pd.read_excel(xls, sheet_name=sheet)
        st.sidebar.caption(f"Using sheet: **{sheet}**")
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        st.stop()

st.success(f"Loaded {len(raw):,} rows from **{uploaded.name}**")

# Ensure column names are stripped
raw.columns = [str(c).strip() for c in raw.columns]

# -----------------------------
# Detect meta + week columns
# -----------------------------
meta_cols_present = [c for c in KNOWN_META if c in raw.columns]
week_cols = detect_week_columns(raw)

if not week_cols:
    st.error("No weekly date columns detected. Make sure your week columns are actual dates or date-like strings.")
    st.stop()

# Include any extra meta columns beyond KNOWN_META (but exclude week columns)
extra_meta = [c for c in raw.columns if c not in week_cols and c not in meta_cols_present]
meta_cols = meta_cols_present + extra_meta

with st.expander("Detected columns", expanded=False):
    st.write("**Meta columns:**", meta_cols)
    st.write("**Week columns:**", week_cols)

# -----------------------------
# Tidy transform
# -----------------------------
tidy = melt_wide(raw, meta_cols=meta_cols, week_cols=week_cols)

# Drop rows without a valid week or units
tidy = tidy.dropna(subset=["_week"])
tidy["_units"] = tidy["_units"].fillna(0)

# Build location label
tidy["_loc_label"] = tidy.apply(build_location_label, axis=1, args=(primary_label, add_postcode))

# Derive outward postcode (e.g., 'SK4' from 'SK4 4DF') for geographic aggregation
if "POSTCODE" in tidy.columns:
    tidy["_outward"] = tidy["POSTCODE"].astype(str).apply(to_outward)
else:
    tidy["_outward"] = ""

# -----------------------------
# Category filter
# -----------------------------
category_col = None
for candidate in ["CATEGORY", "MULTIPLE CATEGORY"]:
    if candidate in tidy.columns:
        category_col = candidate
        break

selected_category = "ALL"
if category_col:
    categories = sorted([c for c in tidy[category_col].dropna().astype(str).unique() if c.strip() and c.strip().lower() != "(blank)"])
    with st.sidebar:
        st.header("4) Category")
        selected_category = st.selectbox("Filter CATEGORY", options=["ALL"] + categories, index=0)

    if selected_category != "ALL":
        tidy = tidy[tidy[category_col].astype(str) == selected_category]

# -----------------------------
# Multiples mapping (optional CSV)
# -----------------------------
with st.sidebar:
    st.header("5) Multiples mapping (optional)")
    multiples_map_file = st.file_uploader("Upload mapping CSV", type=["csv"], key="multiples_map_csv")

# Choose grouping
group_col = None if group_by == "— none —" else group_by

# Aggregate to location-week (and optional group)
group_fields = ["_loc_label", "_week"]
if group_col and group_col in tidy.columns:
    group_fields.insert(1, group_col)
agg = tidy.groupby(group_fields, dropna=False)["_units"].sum().reset_index()

# -----------------------------
# Week selection
# -----------------------------
all_weeks = sorted(agg["_week"].unique())
default_prev, default_curr = choose_two_most_recent(all_weeks)

st.subheader("Select comparison weeks")
c1, c2 = st.columns(2)
with c1:
    week_curr = st.selectbox("Current week", options=all_weeks, index=all_weeks.index(default_curr) if default_curr in all_weeks else len(all_weeks)-1)
with c2:
    # default to immediately prior week
    prior_index = max(0, all_weeks.index(week_curr)-1)
    week_prev = st.selectbox("Previous week", options=all_weeks, index=prior_index)

if week_prev == week_curr:
    st.error("Please choose two different weeks.")
    st.stop()

# -----------------------------
# Pivot and compute changes
# -----------------------------
pivot_index = ["_loc_label"] + ([group_col] if group_col else [])
pivot = agg.pivot_table(index=pivot_index, columns="_week", values="_units", aggfunc="sum").fillna(0)

# Ensure both selected weeks exist
for w in [week_prev, week_curr]:
    if w not in pivot.columns:
        pivot[w] = 0

pivot = pivot.reset_index()
pivot["_abs_change"] = pivot[week_curr] - pivot[week_prev]
pivot["_pct_change"] = np.where(pivot[week_prev] == 0, np.nan, (pivot["_abs_change"] / pivot[week_prev]) * 100)

# Apply basic filters
view = pivot.copy()
view = view[view[week_prev] >= min_prev_units]
view = view[view["_abs_change"].abs() >= min_abs_change]

# Outlier z-score on % change
pct = view["_pct_change"].dropna()
if len(pct) >= 5 and pct.std(ddof=0) > 0:
    mean_pct = pct.mean()
    std_pct = pct.std(ddof=0)
    view["_z_pct_change"] = (view["_pct_change"] - mean_pct) / std_pct
else:
    view["_z_pct_change"] = np.nan
view["_is_outlier"] = view["_z_pct_change"].abs() >= z_threshold

# --- Geographical aggregation by outward code ---
geo = tidy.copy()
geo = geo[geo["_outward"].notna() & (geo["_outward"].astype(str).str.len() > 0)]
geo_agg = geo.groupby(["_outward", "_week"])['_units'].sum().unstack("_week").fillna(0)

for w in [week_prev, week_curr]:
    if w not in geo_agg.columns:
        geo_agg[w] = 0

geo_agg = geo_agg[[week_prev, week_curr]]
geo_agg["Δ units"] = geo_agg[week_curr] - geo_agg[week_prev]
geo_agg["Δ % (WoW)"] = np.where(geo_agg[week_prev] == 0, np.nan, (geo_agg["Δ units"] / geo_agg[week_prev]) * 100)
geo_agg = geo_agg.reset_index().rename(columns={"_outward": "Outward", week_prev: f"{week_prev} units", week_curr: f"{week_curr} units"})

# Format percentage to whole numbers for display in tables
if "Δ % (WoW)" in geo_agg.columns:
    geo_agg["Δ % (WoW)"] = pd.to_numeric(geo_agg["Δ % (WoW)"], errors="coerce").round(0)
    geo_agg["Δ % (WoW)"] = geo_agg["Δ % (WoW)"].apply(lambda v: "" if pd.isna(v) else f"{int(v)}%")

# -----------------------------
# Overview KPIs
# -----------------------------
st.subheader("Overview")
k1, k2, k3, k4 = st.columns(4)
total_prev = float(pivot[week_prev].sum())
total_curr = float(pivot[week_curr].sum())
total_abs = total_curr - total_prev
total_pct = (total_abs / total_prev * 100) if total_prev else np.nan

k1.metric(f"Total units {week_prev}", f"{int(total_prev):,}")
k2.metric(f"Total units {week_curr}", f"{int(total_curr):,}", delta=f"{int(total_abs):,}")
k3.metric("Total % change", f"{total_pct:.0f}%" if pd.notna(total_pct) else "—")
k4.metric("Locations flagged (outliers)", f"{int(view['_is_outlier'].sum()):,}")

# -----------------------------
# Winners / Decliners tables
# -----------------------------
def present(df_in: pd.DataFrame) -> pd.DataFrame:
    cols = ["_loc_label"]
    if group_col:
        cols.append(group_col)
    cols += [week_prev, week_curr, "_abs_change", "_pct_change", "_is_outlier"]
    out = df_in[cols].copy()
    rename_map = {
        "_loc_label": "Location",
        week_prev: f"{week_prev} units",
        week_curr: f"{week_curr} units",
        "_abs_change": "Δ units",
        "_pct_change": "Δ % (WoW)",
        "_is_outlier": "Outlier?"
    }
    if group_col:
        rename_map[group_col] = group_col.title()
    out = out.rename(columns=rename_map)
    # Format percentage as whole number for display
    if "Δ % (WoW)" in out.columns:
        try:
            out["Δ % (WoW)"] = pd.to_numeric(out["Δ % (WoW)"], errors="coerce").round(0)
            out["Δ % (WoW)"] = out["Δ % (WoW)"].apply(lambda v: "" if pd.isna(v) else f"{int(v)}%")
        except Exception:
            pass
    return out

st.subheader("Location changes")

st.markdown("### 🔼 Top increases")
winners = view[view["_abs_change"] > 0].sort_values(by=["_abs_change", "_pct_change"], ascending=[False, False]).head(int(top_n))
st.dataframe(present(winners), use_container_width=True, hide_index=True)

st.markdown("### 🔽 Top declines")
decliners = view[view["_abs_change"] < 0].sort_values(by=["_abs_change", "_pct_change"], ascending=[True, True]).head(int(top_n))
st.dataframe(present(decliners), use_container_width=True, hide_index=True)

# Category week-on-week rollup (only when ALL is selected and a category column exists)
if 'category_col' in locals() and category_col and 'selected_category' in locals() and selected_category == "ALL":
    st.markdown("### 📊 Categories – WoW change (ALL)")
    cat_agg = tidy.groupby([category_col, "_week"])['_units'].sum().unstack("_week").fillna(0)
    for w in [week_prev, week_curr]:
        if w not in cat_agg.columns:
            cat_agg[w] = 0
    cat_agg = cat_agg[[week_prev, week_curr]]
    cat_agg["Δ units"] = cat_agg[week_curr] - cat_agg[week_prev]
    cat_agg["Δ % (WoW)"] = np.where(cat_agg[week_prev] == 0, np.nan, (cat_agg["Δ units"] / cat_agg[week_prev]) * 100)
    cat_out = cat_agg.reset_index().rename(columns={category_col: "Category", week_prev: f"{week_prev} units", week_curr: f"{week_curr} units"})
    if "Δ % (WoW)" in cat_out.columns:
        cat_out["Δ % (WoW)"] = pd.to_numeric(cat_out["Δ % (WoW)"], errors="coerce").round(0)
        cat_out["Δ % (WoW)"] = cat_out["Δ % (WoW)"].apply(lambda v: "" if pd.isna(v) else f"{int(v)}%")
    st.dataframe(cat_out.sort_values(by="Δ units", ascending=False), use_container_width=True, hide_index=True)

# -----------------------------
# Distribution chart
# -----------------------------
st.subheader("Distribution of % changes (WoW)")
chart_df = view[["_pct_change", "_is_outlier"]].dropna()
if not chart_df.empty:
    base = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("_pct_change", bin=alt.Bin(maxbins=40), title="% change WoW"),
        y="count()",
        tooltip=[alt.Tooltip("count()", title="Locations")]
    )
    st.altair_chart(base, use_container_width=True)
else:
    st.caption("No % changes available to chart (perhaps all zero prior-week units after filters).")

# -----------------------------
# Optional group rollup
# -----------------------------
if group_col:
    st.subheader(f"{group_col.title()} rollup")
    reg = pivot.groupby(group_col).agg(
        prev=(week_prev, "sum"),
        curr=(week_curr, "sum"),
        locations=("_loc_label", "nunique")
    ).reset_index()
    reg["Δ units"] = reg["curr"] - reg["prev"]
    reg["Δ % (WoW)"] = np.where(reg["prev"] == 0, np.nan, reg["Δ units"] / reg["prev"] * 100)
    reg = reg.rename(columns={"prev": f"{week_prev} units", "curr": f"{week_curr} units"})
    # Format percentage to whole number for display
    if "Δ % (WoW)" in reg.columns:
        reg["Δ % (WoW)"] = pd.to_numeric(reg["Δ % (WoW)"], errors="coerce").round(0)
        reg["Δ % (WoW)"] = reg["Δ % (WoW)"].apply(lambda v: "" if pd.isna(v) else f"{int(v)}%")
    st.dataframe(reg, use_container_width=True)

 # =============================
# Tab: Geographical change
# =============================
geo_tab, = st.tabs(["🗺️ Geographical change"])

with geo_tab:
    st.markdown("#### Top outward codes by change")

    # Positive movers (only positive Δ units)
    pos = geo_agg.sort_values(by="Δ units", ascending=False)
    pos = pos[pos["Δ units"] > 0].head(int(top_n))
    st.markdown("**Top increases (outward)**")
    st.dataframe(pos, use_container_width=True, hide_index=True)

    # Negative movers (only negative Δ units)
    neg = geo_agg.sort_values(by="Δ units", ascending=True)
    neg = neg[neg["Δ units"] < 0].head(int(top_n))
    st.markdown("**Top declines (outward)**")
    st.dataframe(neg, use_container_width=True, hide_index=True)

    st.markdown("#### UK map (outward centroids)")
    import os
    centroids_path = os.path.join(os.path.dirname(__file__), "uk_outward_centroids.csv")
    if os.path.exists(centroids_path):
        centroids = pd.read_csv(centroids_path)
        centroids["Outward"] = centroids["Outward"].astype(str).str.upper().str.replace(" ", "", regex=False)
        # Ensure lat/lon are numeric and sane
        centroids["lat"] = pd.to_numeric(centroids["lat"], errors="coerce")
        centroids["lon"] = pd.to_numeric(centroids["lon"], errors="coerce")
        # Keep rough UK bounds to avoid bad points
        centroids = centroids[(centroids["lat"].between(49, 61)) & (centroids["lon"].between(-8.5, 2.5))]
        geo_agg["Outward"] = geo_agg["Outward"].astype(str).str.upper().str.replace(" ", "", regex=False)

        map_df = pd.merge(centroids, geo_agg, on="Outward", how="left").fillna({"Δ units": 0})
        # Drop rows without valid coordinates
        map_df_valid = map_df.dropna(subset=["lat", "lon"]).copy()
        with st.expander("Map debug", expanded=False):
            st.write({
                "centroids_rows": len(centroids),
                "geo_agg_outwards": geo_agg["Outward"].nunique(),
                "joined_rows": len(map_df),
                "valid_coord_rows": len(map_df_valid),
                "nonzero_points": int((map_df_valid["Δ units"] != 0).sum()),
            })
            st.dataframe(map_df_valid[["Outward","lat","lon","Δ units"]].head(20), use_container_width=True)
        # Build color and radius from Δ units
        if map_df_valid.empty:
            st.warning("No valid points to display for the selected weeks (no coordinates or all filtered). Try different weeks.")
        else:
            max_abs = max(1.0, float(map_df_valid["Δ units"].abs().max()))
            def color_fn(x):
                if x > 0:
                    return [0, int(200 * (abs(x) / max_abs)) + 55, 0]
                elif x < 0:
                    return [int(200 * (abs(x) / max_abs)) + 55, 0, 0]
                else:
                    return [160, 160, 160]
            map_df_valid["color"] = map_df_valid["Δ units"].apply(color_fn)
            # Add alpha channel for visibility and bump base radius
            map_df_valid["color_rgba"] = map_df_valid["color"].apply(lambda c: c + [220])
            map_df_valid["radius"] = map_df_valid["Δ units"].abs().apply(lambda v: 15000 + 30000 * (v / max_abs))

            # Dynamic view: center on data; fallback to UK
            lat0 = float(map_df_valid["lat"].mean())
            lon0 = float(map_df_valid["lon"].mean())
            view_state = pdk.ViewState(latitude=lat0 if not pd.isna(lat0) else 54.5,
                                       longitude=lon0 if not pd.isna(lon0) else -3.5,
                                       zoom=5.2, pitch=0)

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df_valid,
                get_position=["lon", "lat"],
                get_fill_color="color_rgba",
                get_radius="radius",
                pickable=True,
                auto_highlight=True,
            )
            tooltip = {"html": "<b>{Outward}</b><br/>Δ units: {Δ units}", "style": {"backgroundColor": "white", "color": "black"}}
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
        with st.expander("Centroid data & notes"):
            st.caption("Using outward-code centroids. For true area shading (polygons), we can switch to a postcode-district GeoJSON later.")
    else:
        st.info("To show the map, add `uk_outward_centroids.csv` next to app.py with columns: Outward,lat,lon.")

# -----------------------------
# Multiples table (full width)
# -----------------------------
# Determine the source column for multiples (Row C): prefer 'MULTIPLE CATEGORY'
multiple_src_col = None
cols = list(tidy.columns)

# Helper to normalise
norm = lambda s: str(s).strip().lower()
exact_names = {"multiple category", "multiples", "multiple", "retailer", "chain"}

# First pass: exact case-insensitive match
for c in cols:
    if norm(c) in exact_names:
        multiple_src_col = c
        break

# Second pass: any column containing 'multiple' (case-insensitive)
if multiple_src_col is None:
    for c in cols:
        if "multiple" in norm(c):
            multiple_src_col = c
            break

# If still not found, allow manual selection from sidebar (meta columns first)
if multiple_src_col is None:
    with st.sidebar:
        st.caption("No obvious Multiples column found. Pick one:")
        # Prefer known meta cols first if present
        preferred_options = [c for c in KNOWN_META if c in tidy.columns]
        other_options = [c for c in tidy.columns if c not in preferred_options]
        pick = st.selectbox("Multiples source column", options=["— none —"] + preferred_options + other_options, key="multiples_src_pick")
        if pick != "— none —":
            multiple_src_col = pick

if multiple_src_col is None:
    st.subheader("Multiples – WoW change")
    st.info("No 'MULTIPLE CATEGORY' style column found or selected, so this table cannot be built.")
else:
    # Build a mapping function from optional CSV
    def build_mapping_func(map_df: pd.DataFrame):
        # Try to use first two columns as (pattern -> canonical)
        if map_df.shape[1] < 2:
            return None
        left, right = map_df.columns[:2]
        # Clean
        m = map_df[[left, right]].dropna().copy()
        m[left] = m[left].astype(str).str.strip().str.upper()
        m[right] = m[right].astype(str).str.strip()
        # Exact dict
        exact = dict(zip(m[left], m[right]))
        # Substring list (patterns with length >=3)
        substr = [(p, g) for p, g in exact.items() if isinstance(p, str) and len(p) >= 3]

        def mapper(val: str):
            s = "" if pd.isna(val) else str(val).strip().upper()
            if s in exact:
                return exact[s]
            # substring pass (e.g., 'TESCO EXPRESS' matches 'TESCO')
            for p, g in substr:
                if p in s:
                    return g
            return s or "Other"
        return mapper

    mapper = None
    import os
    if multiples_map_file is not None:
        # User-uploaded mapping takes priority
        try:
            map_df = pd.read_csv(multiples_map_file)
            mapper = build_mapping_func(map_df)
            st.sidebar.caption("Using uploaded Mapping CSV")
        except Exception as e:
            st.warning(f"Could not read mapping CSV: {e}")
    else:
        # Fallback to local Mapping_Table.csv next to the app file
        local_map_path = os.path.join(os.path.dirname(__file__), "Mapping_Table.csv")
        if os.path.exists(local_map_path):
            try:
                map_df = pd.read_csv(local_map_path)
                mapper = build_mapping_func(map_df)
                st.sidebar.caption("Using local Mapping_Table.csv")
            except Exception as e:
                st.warning(f"Could not read local Mapping_Table.csv: {e}")

    # Apply mapping
    tidy["_multiple_raw"] = tidy[multiple_src_col].astype(str)
    if mapper:
        tidy["_multiple_group"] = tidy["_multiple_raw"].apply(mapper)
    else:
        tidy["_multiple_group"] = tidy["_multiple_raw"]

    # Aggregate by multiples across the two selected weeks
    mult = tidy.groupby(["_multiple_group", "_week"])['_units'].sum().unstack("_week").fillna(0)
    for w in [week_prev, week_curr]:
        if w not in mult.columns:
            mult[w] = 0
    mult = mult[[week_prev, week_curr]]
    mult["Δ units"] = mult[week_curr] - mult[week_prev]
    # Percent change; avoid divide-by-zero
    mult["Δ % (WoW)"] = np.where(mult[week_prev] == 0, np.nan, (mult["Δ units"] / mult[week_prev]) * 100)
    mult_out = mult.reset_index().rename(columns={"_multiple_group": "Multiple", week_prev: f"{week_prev} units", week_curr: f"{week_curr} units"})

    # Sort by largest absolute movement
    mult_out = mult_out.sort_values(by="Δ units", ascending=False)

    # Format percentage as whole number for display
    if "Δ % (WoW)" in mult_out.columns:
        mult_out["Δ % (WoW)"] = pd.to_numeric(mult_out["Δ % (WoW)"], errors="coerce").round(0)
        mult_out["Δ % (WoW)"] = mult_out["Δ % (WoW)"].apply(lambda v: "" if pd.isna(v) else f"{int(v)}%")

    st.subheader("Multiples – WoW change")
    st.dataframe(mult_out, use_container_width=True, hide_index=True)

# -----------------------------
# Downloads
# -----------------------------
st.subheader("Download sub-reports")
reports = {
    f"winners_top_{int(top_n)}.csv": present(winners),
    f"decliners_top_{int(top_n)}.csv": present(decliners),
    "all_locations_changes.csv": present(view.sort_values(by="_abs_change", ascending=False)),
}
dcols = st.columns(len(reports))
for (name, data), col in zip(reports.items(), dcols):
    with col:
        st.download_button(
            label=f"Download {name}",
            data=to_csv_bytes(data),
            file_name=name,
            mime="text/csv"
        )

st.caption("Tips: keep headers consistent each week; the app will auto-detect week columns. Label = STORE NAME [+ POSTCODE]. Group by WHOLESALER is optional.")