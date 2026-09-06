import sys

sys.path.insert(0, r"C:\Users\Ricci\Fisheries")

import numpy as np
import pandas as pd
import streamlit as st

from app.components.folium_map import (add_cell_layer, coarsen_grid, html_for,
                                        integrated_map, ph_overview_map, squares_map)
from app.components.ocean_map import VARIABLE_META, _stable_var_ranges
from app.components.state import get_cube, time_options

cube = get_cube()
times = time_options(cube)
ranges = _stable_var_ranges()

st.title("iframe dynamic test")

var = st.selectbox("Variable", [v for v in VARIABLE_META if v in cube.data_vars],
                   format_func=lambda v: VARIABLE_META[v]["label"])
idx = st.slider("Composite window", 0, len(times) - 1, len(times) - 1)
time = times[idx]
st.caption(f"SELECTED: {var} @ {pd.Timestamp(time):%Y-%m-%d} (idx={idx})")

da = cube[var].sel(time=str(time), method="nearest")
if da.ndim > 2:
    da = da.isel(time=0)
lat2d, lon2d = np.meshgrid(da["lat"].values, da["lon"].values, indexing="ij")
yy, xx, vv = coarsen_grid(
    lat2d.ravel(), lon2d.ravel(), np.asarray(da.values, dtype="float64").ravel(), 2
)
lo, hi = ranges[var]
st.caption(f"MEAN VALUE of current map: {np.nanmean(vv):.3f} {VARIABLE_META[var]['cbar']} | "
           f"cells={len(vv)} | range {lo:.2f}-{hi:.2f}")

style = st.radio("Layer style", ["Smooth", "Squares", "Dots"], horizontal=True)
m1 = ph_overview_map()
if style == "Dots":
    add_cell_layer(m1, yy, xx, vv, VARIABLE_META[var]["label"], VARIABLE_META[var]["cbar"],
                   vmin=lo, vmax=hi)
elif style == "Squares":
    squares_map(m1, yy, xx, vv, VARIABLE_META[var]["label"], VARIABLE_META[var]["cbar"],
                vmin=lo, vmax=hi)
else:
    integrated_map(m1, yy, xx, vv, VARIABLE_META[var]["label"], VARIABLE_META[var]["cbar"],
                   vmin=lo, vmax=hi)
h1 = html_for(m1)

st.subheader("Map via st.iframe")
st.iframe(h1, height=500)