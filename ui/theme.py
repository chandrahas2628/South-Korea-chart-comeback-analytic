"""Visual theme: CSS for the Streamlit shell and a shared Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio

BG, CARD, BORDER = "#0B1220", "#111A2E", "#1F2A44"
TEXT, MUTED = "#E6EAF2", "#8B97B0"
ACCENT, TEAL, AMBER, ROSE, VIOLET = "#5B8DEF", "#34D3A5", "#F5A524", "#F2677A", "#9B8AFB"
SEQ = ["#5B8DEF", "#34D3A5", "#F5A524", "#9B8AFB", "#F2677A", "#6CB6D9"]
PATTERN_COLORS = {"Strong / Sustainable": TEAL, "Explosive / Short-lived": AMBER,
                  "Moderate / Stable": ACCENT, "Weak / Short-lived": "#6B7694", "Not scored": "#3A4560"}

CSS = f"""
<style>
html, body, [class*="css"] {{ font-family: Inter, -apple-system, "Segoe UI", Roboto, sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; font-weight: 650; }}
section[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {BORDER}; }}
.kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; height: 100%; }}
.kpi .label {{ color: {MUTED}; font-size: 0.74rem; text-transform: uppercase; letter-spacing: .06em; }}
.kpi .value {{ color: {TEXT}; font-size: 1.65rem; font-weight: 650; line-height: 1.25; margin-top: 2px; }}
.kpi .sub {{ color: {MUTED}; font-size: 0.78rem; margin-top: 2px; }}
.sec {{ margin: 1.4rem 0 .4rem 0; padding-bottom: .35rem; border-bottom: 1px solid {BORDER}; }}
.sec .t {{ font-size: 1.05rem; font-weight: 650; }}
.sec .s {{ color: {MUTED}; font-size: .85rem; }}
.insight {{ background: {CARD}; border: 1px solid {BORDER}; border-left: 3px solid {ACCENT}; border-radius: 8px;
           padding: 10px 14px; margin-bottom: 8px; font-size: .9rem; }}
.insight .c {{ color: {ACCENT}; font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; }}
.callout {{ background: #0F1A33; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 16px; font-size: .9rem; }}
.pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; border:1px solid {BORDER}; color:{MUTED}; }}
div[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 10px 14px; }}
</style>
"""


def register_plotly_template() -> None:
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=12),
        colorway=SEQ, margin=dict(l=50, r=20, t=55, b=45),
        title=dict(font=dict(size=15), x=0.0, xanchor="left"),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font_color=TEXT),
    )
    pio.templates["atlantic"] = tpl
    pio.templates.default = "atlantic"
