"""Animated ocean variable map (SST, chlorophyll, PAR, Kd490, indices).

Uses plotly Scatter for the gridded fields plus a grey land silhouette from
bathymetry, so no basemap tiles or API tokens are needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from app.components.state import land_points, time_options

# Variable metadata shown in the UI.
VARIABLE_META: dict[str, dict[str, Any]] = {
    "sst": {"label": "Sea Surface Temperature", "colorscale": "RdYlBu_r",
            "cbar": "degC", "ndigits": 1},
    "chl": {"label": "Chlorophyll-a", "colorscale": "YlGnBu",
            "cbar": "mg/m^3", "ndigits": 2},
    "par": {"label": "Photosynthetically Active Radiation", "colorscale": "Viridis",
            "cbar": "E/m^2/day", "ndigits": 0},
    "kd490": {"label": "Diffuse Attenuation Kd(490)", "colorscale": "Cividis",
              "cbar": "1/m", "ndigits": 2},
    "frontal_index": {"label": "SST Frontal Index", "colorscale": "Reds",
                      "cbar": "degC/deg", "ndigits": 2},
    "productivity_index": {"label": "Productivity Index", "colorscale": "YlGn",
                           "cbar": "log10(chl)", "ndigits": 2},
    "upwelling_proxy": {"label": "Upwelling Proxy", "colorscale": "Blues",
                        "cbar": "degC (cooling)", "ndigits": 2},
}


def ocean_figure(
    cube: xr.Dataset,
    var: str,
    time: pd.Timestamp | str,
    coarse: int = 2,
) -> go.Figure:
    """Render one composite window of ``var`` as an interactive scatter map."""
    if var not in cube.data_vars:
        raise ValueError(f"Variable {var!r} not in cube: {list(cube.data_vars)}")

    meta = VARIABLE_META.get(var, {"label": var, "colorscale": "Viridis",
                                   "cbar": "", "ndigits": 2})
    da = cube[var].sel(time=str(time), method="nearest")
    if da.ndim > 2:
        da = da.isel(time=0)

    if coarse > 1:
        da = _coarsen(da, coarse)
    land = land_points(cube)

    lon2d, lat2d = np.meshgrid(da["lon"].values, da["lat"].values, indexing="ij")
    vals = np.asarray(da.values, dtype="float64").T  # -> (lon, lat)
    x, y, z = lon2d.ravel(), lat2d.ravel(), vals.ravel()
    ocean = ~np.isnan(z)
    x, y, z = x[ocean], y[ocean], z[ocean]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=land["lon"], y=land["lat"], mode="markers",
            marker=dict(size=1.6, color="#d9d3c0", opacity=0.85),
            name="land", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(
                size=3.0,
                color=z,
                colorscale=meta["colorscale"],
                colorbar=dict(title=meta["cbar"]),
                showscale=True,
            ),
            name=meta["label"],
            customdata=np.stack([np.round(z, meta["ndigits"])], axis=-1),
            hovertemplate="lon %{x:.2f}<br>lat %{y:.2f}<br>%{customdata[0]}"
            + f" {meta['cbar']}<extra></extra>",
        )
    )
    fig.update_layout(
        height=560,
        title=f"{meta['label']} -- composite {pd.Timestamp(time):%Y-%m-%d}",
        xaxis_title="Longitude", yaxis_title="Latitude",
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="#0e1117", font=dict(color="#ffffff"),
        xaxis=dict(range=[116.0, 127.0], showgrid=False),
        yaxis=dict(range=[4.5, 21.5], showgrid=False),
    )
    return fig


def _var_ranges(cube: xr.Dataset) -> dict[str, tuple[float, float]]:
    """Per-variable colour-scale range (2-98 percentile) over all windows, so
    the map keeps a fixed colour scale and time changes stay visible."""
    ranges: dict[str, tuple[float, float]] = {}
    for var in VARIABLE_META:
        if var not in cube.data_vars:
            continue
        v = np.asarray(cube[var].values, dtype="float64")
        lo, hi = np.nanpercentile(v, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        ranges[var] = (float(lo), float(hi))
    return ranges


@st.cache_data(show_spinner="Computing stable colour scales ...", ttl=3600)
def _stable_var_ranges() -> dict[str, tuple[float, float]]:
    from app.components.state import get_cube

    return _var_ranges(get_cube())


def ocean_tab(cube: xr.Dataset) -> None:
    from app.components.folium_map import (add_cell_layer, coarsen_grid, html_for,
                                           integrated_map, ph_overview_map, squares_map)

    variables = [v for v in VARIABLE_META if v in cube.data_vars]
    times = time_options(cube)

    col_ctrl, col_map = st.columns([1, 3])
    with col_ctrl:
        var = st.selectbox("Variable", variables,
                           format_func=lambda v: VARIABLE_META[v]["label"])
        idx = st.slider("Composite window", 0, len(times) - 1, len(times) - 1)
        time = times[idx]
        st.caption(f"Window: {pd.Timestamp(time):%Y-%m-%d}")
        coarse = st.select_slider("Detail", options=[1, 2, 3, 4], value=2,
                                  help="Larger = faster; smaller = finer cells")
        style = st.radio("Layer style", ["Smooth", "Squares", "Dots"], horizontal=True,
                         help="Smooth = sharp sea gradient with real coastline; "
                              "Squares = flat colored blocks per cell; Dots = one circle per cell")
        st.markdown("")
        st.caption("Basemap layers: Plain (dark) / Street (OSM) / Satellite (Esri) / Topographic")

    with col_map:
        if var not in cube.data_vars:
            st.warning(f"Variable {var!r} not in cube.")
        else:
            da = cube[var].sel(time=str(time), method="nearest")
            if da.ndim > 2:
                da = da.isel(time=0)
            lat2d, lon2d = np.meshgrid(da["lat"].values, da["lon"].values, indexing="ij")
            val2d = np.asarray(da.values, dtype="float64")
            yy, xx, vv = coarsen_grid(
                lat2d.ravel(), lon2d.ravel(), val2d.ravel(), coarse
            )
            meta = VARIABLE_META[var]
            lo, hi = _stable_var_ranges()[var]
            nd = meta.get("ndigits", 2)
            st.caption(
                f"{meta['label']} · {pd.Timestamp(time):%Y-%m-%d} · "
                f"{len(vv):,} cells · {lo:.{nd}f}–{hi:.{nd}f} {meta['cbar']} · "
                f"mean {np.nanmean(vv):.{nd}f} {meta['cbar']}"
            )
            m = ph_overview_map()
            if style == "Dots":
                add_cell_layer(m, yy, xx, vv, meta["label"], meta["cbar"], vmin=lo, vmax=hi)
            elif style == "Squares":
                squares_map(m, yy, xx, vv, meta["label"], meta["cbar"], vmin=lo, vmax=hi)
            else:
                integrated_map(m, yy, xx, vv, meta["label"], meta["cbar"], vmin=lo, vmax=hi)
            st.iframe(html_for(m), height=500)


def _coarsen(da: xr.DataArray, factor: int) -> xr.DataArray:
    """Block-average onto a coarser grid for fast plotting."""
    lat = len(da["lat"]) - (len(da["lat"]) % factor)
    lon = len(da["lon"]) - (len(da["lon"]) % factor)
    with np.errstate(invalid="ignore", divide="ignore"):
        return da.isel(lat=slice(0, lat), lon=slice(0, lon)).coarsen(
            lat=factor, lon=factor, boundary="trim"
        ).mean()