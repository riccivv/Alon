"""OceanForecast Philippines -- Streamlit application.

Run with ``streamlit run app.py`` after building the cube and (optionally)
training the per-species models.

Tabs:
    1. Ocean map      -- animated environmental fields (SST, chl, PAR, Kd490, indices)
    2. Yield forecast -- per-species prediction for the next 8-day window
    3. FMA dashboard  -- region rankings + condition trends
    4. Sustainability -- MSY-proxy flags per region
"""

from __future__ import annotations

import logging

import streamlit as st

from src.config import APP_SUBTITLE, APP_TITLE

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="\U0001F41F",  # fish
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html(
    """
    <style>
      /* Reclaim the vertical space above the title: remove the default top
         padding and collapse the app header/toolbar row. */
      [data-testid="stHeader"] { display: none; }
      #MainMenu, footer { visibility: hidden; }
      [data-testid="stMainBlockContainer"], .block-container {
          padding-top: 0.25rem;
          padding-bottom: 1rem;
      }
    </style>
    """
)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

from app.components.state import get_cube  # noqa: E402

try:
    cube = get_cube()
except Exception:  # get_cube calls st.stop() on missing artifacts
    st.stop()

tab_ocean, tab_forecast, tab_dashboard, tab_sustainability = st.tabs(
    ["Ocean map", "Yield forecast", "FMA dashboard", "Sustainability"]
)

with st.sidebar:
    st.header("Study domain")
    st.caption("Philippine EEZ (approx.) -- grid cells 0.1 deg (~11 km)")
    if st.toggle("Developer notes", value=False):
        st.markdown(
            "- Sources: NASA MODIS-Aqua L3m 8-day 9km composites 2015-2025\n"
            "- Static fields: ETOPO 2022 bathymetry, distance to coast\n"
            "- Forecast baseline: latest 8-day composite; next-window projection\n"
            "- Label strategy: BFAR/PSA catches when present, else habitat suitability"
        )

with tab_ocean:
    from app.components.ocean_map import ocean_tab

    ocean_tab(cube)

with tab_forecast:
    from app.components.forecast_map import forecast_tab

    forecast_tab(cube)

with tab_dashboard:
    from app.components.dashboard import dashboard_tab

    dashboard_tab(cube)

with tab_sustainability:
    from app.components.sustainability import sustainability_tab

    sustainability_tab(cube)

st.sidebar.markdown("---")
st.sidebar.caption("MVP build -- oceanographic fisheries forecasting (PH)")