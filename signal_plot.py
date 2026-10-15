"""Plotly signal plots with per-joint legend choices retained across rerenders."""
from functools import lru_cache
import html
import json

from plotly.offline import get_plotlyjs
import streamlit as st
from streamlit import runtime


@lru_cache(maxsize=1)
def _plotly_script():
    return get_plotlyjs().encode("utf-8")


def show_signal_plot(figure, *, key):
    if not runtime.exists():
        return st.plotly_chart(figure, width="stretch", key=key)

    figure.update_layout(template="plotly_white", showlegend=True)

    # The native Streamlit chart is recreated when its figure data changes.
    # Save legend visibility by signal name in the browser, independently of
    # the episode, trace order and temporary absence of a joint.
    media = runtime.get_instance().media_file_mgr
    script_url = media.add(_plotly_script(), "application/javascript", "joint-plotly-library")
    storage_key = json.dumps(f"joint-plot-visibility:{key}")
    after_plot = """
    const graph = document.getElementById('{plot_id}');
    const storageKey = STORAGE_KEY;
    let saved = {};
    try { saved = JSON.parse(sessionStorage.getItem(storageKey) || '{}'); } catch (_) {}
    function saveVisibility() {
        for (const trace of graph.data) saved[trace.name] = trace.visible ?? true;
        try { sessionStorage.setItem(storageKey, JSON.stringify(saved)); } catch (_) {}
    }
    Plotly.restyle(graph, {visible: graph.data.map(trace => saved[trace.name] ?? true)})
        .then(() => graph.on('plotly_restyle', saveVisibility));
    """.replace("STORAGE_KEY", storage_key)
    page = figure.to_html(
        include_plotlyjs=False, full_html=True, div_id=key,
        config={"displaylogo": False, "responsive": True}, post_script=after_plot,
    ).replace("<head>", '<head><style>body{margin:0}</style><script src="'
              + html.escape(str(script_url), quote=True) + '"></script>')
    st.iframe(page, height="content")
