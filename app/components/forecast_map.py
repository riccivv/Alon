"""Species yield forecast map for a selected composite window.

Uses the trained per-species model (log1p tonne predictions back-transformed)
when available; otherwise falls back to the habitat-suitability score. Both
paths return a value per ocean cell which is rendered exactly like the ocean
variable maps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

from app.components.state import get_models, land_points, snapshot_frame, species_options, time_options
from src.data.fisheries import habitat_suitability_from_frame
from src.models.train import predict_frame

FALLBACK_FEATURES = ["sst", "chl"]


def _forecast_series(
    cube: xr.Dataset,
    species_code: str,
    time: pd.Timestamp,
    bundles: dict[str, dict],
) -> pd.Series:
    """Predicted yield (0-1 suitability, or tonnes) per ocean cell."""
    features = FALLBACK_FEATURES
    frame = snapshot_frame(cube, time, features)

    if species_code in bundles:
        features = bundles[species_code]["features"]
        frame = snapshot_frame(cube, time, features)
        pred = predict_frame(bundles[species_code], frame)
        series = pd.Series(pred, index=frame.index)
    else:
        series = habitat_suitability_from_frame(frame, species_code)
        series.index = frame.index

    out = frame[["lat", "lon"]].copy()
    out["value"] = series
    out = out.dropna(subset=["value"])
    return out.set_index(["lat", "lon"])["value"]


def forecast_figure(
    cube: xr.Dataset,
    species_code: str,
    time: pd.Timestamp,
    bundles: dict[str, dict],
    coarse: int = 2,
) -> tuple[go.Figure, dict]:
    """Map of per-cell predicted yield plus a stats summary."""
    land = land_points(cube)
    series = _forecast_series(cube, species_code, time, bundles)

    lon = np.array(sorted(set(series.index.get_level_values("lon"))))
    lat = np.array(sorted(set(series.index.get_level_values("lat"))))
    z = np.full((len(lat), len(lon)), np.nan)
    li = {lo: i for i, lo in enumerate(lon)}
    la = {la_: i for i, la_ in enumerate(lat)}
    rows = series.reset_index()
    z[[la[r["lat"]] for _, r in rows.iterrows()],
      [li[r["lon"]] for _, r in rows.iterrows()]] = rows["value"]

    if coarse > 1:
        with np.errstate(invalid="ignore"):
            lat2 = lat[: len(lat) - (len(lat) % coarse)]
            lon2 = lon[: len(lon) - (len(lon) % coarse)]
            nla, nlo = len(lat2) // coarse, len(lon2) // coarse
            zblock = np.nanmean(
                z[: len(lat2), : len(lon2)].reshape(nla, coarse, nlo, coarse), axis=(1, 3)
            )
            lat, lon, z = lat2[::coarse], lon2[::coarse], zblock

    lon2d, lat2d = np.meshgrid(lon, lat, indexing="ij")
    x, y, v = lon2d.ravel(), lat2d.ravel(), z.ravel()
    ocean = ~np.isnan(v)

    is_tonnes = species_code in bundles and (
        (bundles[species_code]["report"].get("target_transform") or "") == "log1p"
    )
    ndigits = 2 if not is_tonnes else 1
    cbar_title = "tonnes (predicted)" if is_tonnes else "0-1 suitability"
    label = (bundles[species_code]["report"].get("species_name") if species_code in bundles
             else next((s["name"] for s in species_options() if s["code"] == species_code),
                       species_code))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=land["lon"], y=land["lat"], mode="markers",
                             marker=dict(size=1.6, color="#d9d3c0", opacity=0.85),
                             name="land", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x[ocean], y=y[ocean], mode="markers",
        marker=dict(size=3.0, color=v[ocean], colorscale="Turbo",
                    colorbar=dict(title=cbar_title), showscale=True),
        customdata=np.stack([np.round(v[ocean], ndigits)], axis=-1),
        hovertemplate="lon %{x:.2f}<br>lat %{y:.2f}<br>%{customdata[0]}"
        + f" {cbar_title}<extra></extra>",
    ))

    finite = v[~np.isnan(v)]
    summary = {
        "mean": float(np.mean(finite)) if finite.size else 0.0,
        "p90": float(np.percentile(finite, 90)) if finite.size else 0.0,
        "max": float(np.max(finite)) if finite.size else 0.0,
        "top_areas": _top_areas(series, n=5),
        "label": label,
        "units": cbar_title,
    }

    fig.update_layout(
        height=560,
        title=f"{label} -- forecast from {pd.Timestamp(time):%Y-%m-%d} composite",
        xaxis_title="Longitude", yaxis_title="Latitude",
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="#0e1117", font=dict(color="#ffffff"),
        xaxis=dict(range=[116.0, 127.0], showgrid=False),
        yaxis=dict(range=[4.5, 21.5], showgrid=False),
    )
    return fig, summary


def _top_areas(series: pd.Series, n: int = 5) -> list[dict[str, float]]:
    """Highest-value cells (lat/lon/value) as a compact list."""
    top = series.nlargest(n)
    out = []
    for (lat, lon), value in top.items():
        out.append({"lat": float(lat), "lon": float(lon), "value": float(value)})
    return out


def forecast_tab(cube: xr.Dataset) -> None:
    from app.components.folium_map import (add_cell_layer, coarsen_grid, html_for, integrated_map,
                                            ph_overview_map, squares_map)

    species = species_options()
    times = time_options(cube)

    col_ctrl, col_map = st.columns([1, 3])
    with col_ctrl:
        code = st.selectbox("Species", [s["code"] for s in species],
                            format_func=lambda c: next(s["name"] for s in species if s["code"] == c),
                            key="fc_species")
        idx = st.slider("Base composite", 0, len(times) - 1, len(times) - 1, key="fc_time")
        time = times[idx]
        st.caption("Forecast represents the next 8-day window after the base composite")
        coarse = st.select_slider("Detail", options=[1, 2, 3, 4], value=2, key="fc_coarse")
        style = st.radio("Layer style", ["Smooth", "Squares", "Dots"], horizontal=True, key="fc_style",
                         help="Smooth = sharp sea gradient with real coastline; "
                              "Squares = flat colored blocks per cell; Dots = one circle per cell")
        st.markdown("")
        st.caption("Basemap layers: Plain (dark) / Street (OSM) / Satellite (Esri) / Topographic")

    bundles = get_models((code,))
    used_model = code in bundles
    if not used_model:
        st.info("No trained model found -- showing habitat-suitability score. "
                "Run `python scripts/train_model.py` to enable learned forecasts.")

    with col_map:
        series = _forecast_series(cube, code, time, bundles)
        if series.empty:
            st.warning("No valid prediction cells for this composite.")
        else:
            is_tonnes = code in bundles and (
                (bundles[code]["report"].get("target_transform") or "") == "log1p"
            )
            units = "tonnes (predicted)" if is_tonnes else "0-1 suitability"
            label = (bundles[code]["report"].get("species_name") if code in bundles
                     else next((s["name"] for s in species_options() if s["code"] == code), code))

            frame = series.reset_index()
            yy, xx, vv = coarsen_grid(frame["lat"].to_numpy(), frame["lon"].to_numpy(),
                                      frame["value"].to_numpy(), coarse)
            m = ph_overview_map()
            if style == "Dots":
                add_cell_layer(m, yy, xx, vv, label, units)
            elif style == "Squares":
                squares_map(m, yy, xx, vv, label, units)
            else:
                integrated_map(m, yy, xx, vv, label, units)
            st.iframe(html_for(m), height=500)

            finite = vv[np.isfinite(vv)]
            summary = {
                "mean": float(np.mean(finite)) if finite.size else 0.0,
                "p90": float(np.percentile(finite, 90)) if finite.size else 0.0,
                "max": float(np.max(finite)) if finite.size else 0.0,
                "top_areas": _top_areas(series, n=5),
                "label": label,
                "units": units,
            }

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean yield", f"{summary['mean']:.3f}", summary["units"])
            c2.metric("P90 hot spot", f"{summary['p90']:.3f}", summary["units"])
            c3.metric("Peak cell", f"{summary['max']:.3f}", summary["units"])

            if summary["top_areas"]:
                st.caption("Top projected cells")
                st.dataframe(pd.DataFrame(summary["top_areas"]), width="stretch")