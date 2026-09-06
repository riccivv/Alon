"""Interactive Leaflet basemap for the app's cell-level fields.

The gridded satellite/forecast fields are drawn as a semi-transparent colored
mesh over a clean light-gray basemap (key-free Esri/OSM/OpenTopoMap tiles -- no
API tokens, consistent with the project's design). The plain default keeps the
color field readable; the detailed basemaps are still one click away in the
layer control.

The maps are embedded through ``streamlit.iframe`` (which wraps the HTML in a
sandboxed iframe) rather than a third-party component so they always mount.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import folium
from folium import LinearColormap

# Philippine EEZ window (~0.1 deg cells).
PH_CENTER = [12.5, 122.0]
PH_BOUNDS = [[4.5, 116.0], [21.5, 127.0]]

# RdYlBu-style ramp: cool = low, warm = high.
_COLOR_RAMP = [
    "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
    "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026",
]

# Key-free geographic tile layers (CartoDB's built-ins now demand an API key,
# so they are deliberately avoided).
_ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_ESRI_DARK_GRAY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
_OPEN_TOPO = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"

# folium's branca colormap legend needs d3 at runtime in the browser; when
# embedding via raw HTML we inject the same two d3 loads streamlit-folium adds.
_D3_LIBS = (
    '<script src="https://d3js.org/d3.v4.min.js"></script>'
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>'
)

_LAND_JSON = Path(__file__).resolve().parents[2] / "data" / "ph_land.json"


@functools.lru_cache(maxsize=1)
def _ph_land_polys() -> list:
    """Natural Earth land polygons clipped to the PH window (JSON cache)."""
    return json.loads(_LAND_JSON.read_text(encoding="utf-8"))


def ph_overview_map() -> folium.Map:
    """A Leaflet map on the PH EEZ with a dark plain base layer by default.

    The default is a solid dark gray canvas (mirrors the satellite look, with
    no street noise or heavy labels) so the colored field overlay stands out
    clearly; the detailed basemaps remain available through the layer control.
    """
    m = folium.Map(location=PH_CENTER, zoom_start=6, control_scale=True, tiles=None)
    folium.TileLayer(
        _ESRI_DARK_GRAY,
        name="Plain (dark)",
        attr="Esri, USGS, NOAA",
        max_zoom=16,
        show=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street (OSM)", show=False).add_to(m)
    folium.TileLayer(
        _ESRI_WORLD_IMAGERY,
        name="Satellite (Esri)",
        attr="Esri, Maxar, Earthstar Geographics",
        show=False,
    ).add_to(m)
    folium.TileLayer(
        _OPEN_TOPO,
        name="Topographic (OpenTopoMap)",
        attr="OpenStreetMap contributors; SRTM; OpenTopoMap (CC-BY-SA)",
        show=False,
    ).add_to(m)
    m.fit_bounds(PH_BOUNDS)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def coarsen_grid(
    lat_pts, lon_pts, values, factor: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block-average irregular cell values onto a coarser regular grid.

    Returns aligned ``(lat, lon, value)`` arrays for ocean cells only.
    ``factor`` mirrors the app's "Detail" slider (1 = no coarsening).
    """
    lat = np.asarray(lat_pts, dtype=float)
    lon = np.asarray(lon_pts, dtype=float)
    val = np.asarray(values, dtype=float)

    ok = np.isfinite(val) & np.isfinite(lat) & np.isfinite(lon)
    lat, lon, val = lat[ok], lon[ok], val[ok]
    if lat.size == 0:
        return lat, lon, val

    lat1, lon1 = np.unique(lat), np.unique(lon)
    z = np.full((len(lat1), len(lon1)), np.nan)
    la = {v: i for i, v in enumerate(lat1)}
    li = {v: i for i, v in enumerate(lon1)}
    z[[la[x] for x in lat], [li[y] for y in lon]] = val

    if factor > 1:
        nl, no = len(lat1), len(lon1)
        nl2, no2 = nl - nl % factor, no - no % factor
        block = z[:nl2, :no2].reshape(nl2 // factor, factor, no2 // factor, factor)
        z = np.ma.array(block, mask=np.isnan(block)).mean(axis=(1, 3)).filled(np.nan)
        lat1, lon1 = lat1[:nl2][::factor], lon1[:no2][::factor]

    lon2d, lat2d = np.meshgrid(lon1, lat1, indexing="ij")
    xo, yo = lon2d.ravel(), lat2d.ravel()
    zo = z.T.ravel()
    keep = ~np.isnan(zo)
    return yo[keep], xo[keep], zo[keep]


def add_cell_layer(
    m: folium.Map,
    lat,
    lon,
    value,
    label: str,
    units: str,
    max_radius: float = 8.0,
    min_radius: float = 2.5,
    vmin: float | None = None,
    vmax: float | None = None,
) -> folium.Map:
    """Draw ``value`` per cell as colored, value-sized circles on map ``m``.

    Color and radius are mapped on the log1p scale so heavy-tailed tonnage
    fields stay readable; a hover tooltip shows the raw value. A
    ``LinearColormap`` legend is attached to the map. Pass ``vmin``/``vmax`` to
    keep the colour scale fixed across maps (e.g. a per-variable scale across
    all composite windows); otherwise it adapts to this map's 2-98 percentile.

    Each cell is a plain ``folium.CircleMarker`` (not named, so it isn't
    listed in the layer control): folium emits the ``L.control.layers`` object
    before the declared ``var`` of any *named* child, which Leaflet turns into
    an ``undefined`` overlay and kills the whole map script. Unnamed circle
    markers sidestep that breakage entirely.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    val = np.asarray(value, dtype=float)
    ok = np.isfinite(val) & np.isfinite(lat) & np.isfinite(lon)
    lat, lon, val = lat[ok], lon[ok], val[ok]
    if val.size == 0:
        return m

    if vmin is None or vmax is None:
        lo, hi = np.nanpercentile(val, [2, 98])
    else:
        lo, hi = float(vmin), float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(val)), float(np.nanmax(val))
    lo, hi = float(lo), float(hi)

    logv = np.log1p(np.maximum(val, 0.0))
    lmin, lmax = np.log1p(max(lo, 0.0)), np.log1p(max(hi, 0.0))
    if lmax <= lmin:
        lmin, lmax = 0.0, 1.0
    lc = np.clip(logv, lmin, lmax)

    cm = LinearColormap(colors=_COLOR_RAMP, vmin=lmin, vmax=lmax)
    norm = (lc - lmin) / (lmax - lmin)
    radii = min_radius + (max_radius - min_radius) * np.sqrt(norm)

    for i in range(val.size):
        folium.CircleMarker(
            location=[float(lat[i]), float(lon[i])],
            radius=float(radii[i]),
            color=cm(float(lc[i])),
            weight=0.5,
            fill=True,
            fillColor=cm(float(lc[i])),
            fillOpacity=0.72,
            tooltip=f"{label}: {val[i]:.3f} {units}",
        ).add_to(m)

    cm.caption = f"{label} ({units})"
    m.add_child(cm)
    return m


def _ramp_rgb(norm: np.ndarray) -> np.ndarray:
    """Vectorized RGB (0-255) sampling of ``_COLOR_RAMP`` at ``norm`` in [0,1]."""
    ramp = np.array(
        [[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in _COLOR_RAMP], dtype=float
    )
    t = np.clip(np.nan_to_num(np.asarray(norm, dtype=float), nan=0.0), 0.0, 1.0) * (
        len(ramp) - 1
    )
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(ramp) - 1)
    f = t - i0
    return (ramp[i0] + (ramp[i1] - ramp[i0]) * f[..., None]).astype(np.uint8)


def field_overlay(
    m: folium.Map,
    lat,
    lon,
    value,
    label: str,
    units: str,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha_low: float = 0.18,
    alpha_high: float = 0.85,
) -> folium.Map:
    """Draw ``value`` as a semi-transparent raster field over map ``m``.

    Instead of thousands of marker circles, the grid is rendered once as an
    RGBA image (values colored by the same ramp; opacity scaled by value) and
    glued to the map with follows the same color-scale logic as
    ``add_cell_layer`` (log1p, fixed or adaptive scale), so both styles agree.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    val = np.asarray(value, dtype=float)
    ok = np.isfinite(val) & np.isfinite(lat) & np.isfinite(lon)
    lat, lon, val = lat[ok], lon[ok], val[ok]
    if val.size == 0:
        return m

    lat1 = np.unique(lat)
    lon1 = np.unique(lon)
    la = {v: i for i, v in enumerate(lat1)}
    li = {v: i for i, v in enumerate(lon1)}
    z = np.full((len(lat1), len(lon1)), np.nan)
    z[[la[x] for x in lat], [li[y] for y in lon]] = val

    if vmin is None or vmax is None:
        lo, hi = np.nanpercentile(val, [2, 98])
    else:
        lo, hi = float(vmin), float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(val)), float(np.nanmax(val))
    lo, hi = float(lo), float(hi)

    logz = np.log1p(np.maximum(z, 0.0))
    lmin, lmax = np.log1p(max(lo, 0.0)), np.log1p(max(hi, 0.0))
    if lmax <= lmin:
        lmin, lmax = 0.0, 1.0
    norm = np.clip((logz - lmin) / (lmax - lmin), 0.0, 1.0)

    rgb = _ramp_rgb(norm)
    alpha = (alpha_low + (alpha_high - alpha_low) * norm) * 255.0
    alpha = np.where(np.isnan(z), 0.0, alpha)
    rgba = np.dstack([rgb, alpha.astype(np.uint8)])

    from folium.raster_layers import ImageOverlay

    ImageOverlay(
        rgba,
        bounds=[[float(lat1[0]), float(lon1[0])], [float(lat1[-1]), float(lon1[-1])]],
        origin="lower",
        pixelated=True,
        control=False,
    ).add_to(m)

    stops = np.linspace(0.0, 1.0, 11)
    legend_colors = ["#%02x%02x%02x" % tuple(c) for c in _ramp_rgb(stops)]
    cm = LinearColormap(colors=legend_colors, vmin=lo, vmax=hi)
    cm.caption = f"{label} ({units})"
    m.add_child(cm)
    return m


def integrated_map(
    m: folium.Map,
    lat,
    lon,
    value,
    label: str,
    units: str,
    vmin: float | None = None,
    vmax: float | None = None,
    width: int = 1280,
) -> folium.Map:
    """Integrate ``value`` into the sea itself as a smooth gradient.

    Each grid cell first becomes a flat color plateau which is then given a
    2px seam-soften (instead of full-width interpolation, which read as
    blurry), so the field looks like a crisp semi-continuous gradient.  The
    raster is fully opaque: it can never show "empty" water, because every
    sea pixel carries a color.  On top of that raster a *vector* land cover
    (the real coastline from ``data/ph_land.json``) is drawn, so the gradient
    never bleeds onto land at any zoom regardless of raster resolution.
    ``width`` is the raster resolution in pixels.
    """
    import base64
    import io

    from PIL import Image

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    val = np.asarray(value, dtype=float)
    ok = np.isfinite(val) & np.isfinite(lat) & np.isfinite(lon)
    lat, lon, val = lat[ok], lon[ok], val[ok]
    if val.size == 0:
        return m

    lat1 = np.unique(lat)
    lon1 = np.unique(lon)
    la = {v: i for i, v in enumerate(lat1)}
    li = {v: i for i, v in enumerate(lon1)}
    z = np.full((len(lat1), len(lon1)), np.nan)
    z[[la[x] for x in lat], [li[y] for y in lon]] = val

    if vmin is None or vmax is None:
        lo, hi = np.nanpercentile(val, [2, 98])
    else:
        lo, hi = float(vmin), float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(val)), float(np.nanmax(val))
    lo, hi = float(lo), float(hi)

    logz = np.log1p(np.maximum(z, 0.0))
    lmin, lmax = np.log1p(max(lo, 0.0)), np.log1p(max(hi, 0.0))
    if lmax <= lmin:
        lmin, lmax = 0.0, 1.0
    norm = np.clip((logz - lmin) / (lmax - lmin), 0.0, 1.0)
    rgb = Image.fromarray(_ramp_rgb(norm), "RGB")

    lat_span = float(lat1[-1] - lat1[0])
    lon_span = float(lon1[-1] - lon1[0])
    H = max(64, int(width * lat_span / lon_span))
    W = int(width)
    # Rows run south->north in the grid; the map image is displayed with
    # row 0 at the north edge, so flip before encoding (the JPEG branch of
    # folium's encoder applies no origin flip of its own).  To keep the
    # gradient readable instead of smeared, first snap each cell to a flat
    # color plateau (nearest-neighbour) and then soften only the ~2px seams,
    # rather than interpolating across entire cells (which is what made the
    # old result look "blurred by so much").
    from PIL import ImageFilter

    rgb_smooth = np.asarray(
        rgb.resize((W, H), Image.NEAREST).transpose(Image.FLIP_TOP_BOTTOM)
        .filter(ImageFilter.GaussianBlur(2.0)),
        dtype=np.uint8,
    )

    buf = io.BytesIO()
    Image.fromarray(rgb_smooth, "RGB").save(buf, format="JPEG", quality=88)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    from folium.raster_layers import ImageOverlay

    ImageOverlay(
        url,
        bounds=[[float(lat1[0]), float(lon1[0])], [float(lat1[-1]), float(lon1[-1])]],
        control=False,
        opacity=1.0,
        pixelated=False,
    ).add_to(m)

    # Vector land cover ABOVE the raster: real coastline, crisp at every zoom,
    # and it guarantees the heat gradient is only ever visible over water.
    _land_layer(m)

    stops = np.linspace(0.0, 1.0, 11)
    legend_colors = ["#%02x%02x%02x" % tuple(c) for c in _ramp_rgb(stops)]
    cm = LinearColormap(colors=legend_colors, vmin=lo, vmax=hi)
    cm.caption = f"{label} ({units})"
    m.add_child(cm)
    return m


def _land_layer(m: folium.Map) -> folium.Map:
    """Draw a tinted land cover (real coastline) on top of a heat raster."""
    data = {
        "type": "MultiPolygon",
        "coordinates": [
            [ring for ring in poly if len(ring) >= 3] for poly in _ph_land_polys()
        ],
    }
    folium.GeoJson(
        data,
        name="Land",
        control=False,
        style_function=lambda _: {
            "weight": 0.7,
            "color": "#000000",
            "opacity": 0.55,
            "fillColor": "#1f2a18",
            "fillOpacity": 0.8,
        },
    ).add_to(m)
    return m


def squares_map(
    m: folium.Map,
    lat,
    lon,
    value,
    label: str,
    units: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> folium.Map:
    """Square/mosaic style: each grid cell is a flat, crisp colored block.

    Like the smooth gradient it is covered by the real-coastline land vector
    (``_land_layer``), so cells only ever show over water.  This reuses the
    transparent mesh renderer with fully opaque alpha.
    """
    field_overlay(
        m, lat, lon, value, label, units,
        vmin=vmin, vmax=vmax, alpha_low=1.0, alpha_high=1.0,
    )
    _land_layer(m)
    return m


def html_for(m: folium.Map) -> str:
    """Render ``m`` to a self-contained HTML string for embedding.

    Embed via ``st.iframe``. The branca colormap legend requires d3 in the
    browser, so inject it up front.
    """
    if _D3_LIBS not in m.get_root().header.render():
        m.get_root().header.add_child(folium.Element(_D3_LIBS))
    return m.get_root().render()