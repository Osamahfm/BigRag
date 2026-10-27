"""Static design system; all dynamic HTML is escaped."""
from html import escape

import streamlit as st


def install_theme():
    st.markdown('''<style>
    :root { --atlas-accent:#66e0c2; --atlas-muted:#8b9bb7; --atlas-border:#243047; }
    .stApp { background: radial-gradient(ellipse at 88% 0%, #13263a 0, #0b1220 50%); }
    .block-container { max-width:1320px; padding-top:4.2rem; padding-bottom:4rem; }
    h1,h2,h3 { letter-spacing:-.035em; font-weight:650!important; }
    h1 { font-size:2.8rem!important; line-height:1.14!important; }
    [data-testid="stSidebar"] { background:#10192a; border-right:1px solid var(--atlas-border); }
    [data-testid="stSidebarUserContent"] { padding-top:1.5rem; }
    [data-testid="stSidebar"] hr { margin:.85rem 0; }
    [data-testid="stAppDeployButton"] { display:none; }
    [data-testid="stSidebar"] .block-container { padding-top:1rem; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap:.2rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label { padding:.5rem .7rem; border-radius:8px; width:100%; }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#1c3044; color:#90e8d0; }
    [data-testid="stMetric"] { padding:1.05rem 1.3rem; border:1px solid var(--atlas-border); border-radius:12px; background:#121d30; }
    [data-testid="stMetricLabel"] { color:var(--atlas-muted); font-size:.8rem; }
    [data-testid="stMetricValue"] { font-size:1.95rem; font-weight:600; }
    [data-testid="stChatMessage"] { border:1px solid var(--atlas-border); background:#111c2d; border-radius:14px; padding:1.2rem; }
    [data-testid="stChatInput"] { border:1px solid #36546a; border-radius:14px; }
    [data-testid="stExpander"] { border-color:var(--atlas-border); border-radius:10px; }
    [data-testid="stTabs"] [role="tablist"] { gap:1.1rem; border-bottom:1px solid var(--atlas-border); }
    .atlas-brand { display:flex; align-items:center; gap:12px; margin:0 0 1.5rem; }
    .atlas-symbol { color:#082622; background:#6fe1c6; width:40px; height:40px; border-radius:11px; display:grid; place-items:center; font-size:24px; font-weight:800; }
    .atlas-brand-name { font-size:23px; font-weight:750; letter-spacing:-.05em; }
    .atlas-kicker { color:#70dcc4; font-size:10px; letter-spacing:.15em; font-weight:650; text-transform:uppercase; }
    .atlas-topline { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--atlas-border); padding-bottom:18px; margin-bottom:26px; }
    .atlas-breadcrumb { color:var(--atlas-muted); font-size:12px; letter-spacing:.04em; }
    .atlas-status { border:1px solid #2a4e4d; color:#8ce7cb; padding:5px 11px; border-radius:99px; font-size:11px; background:#122d2d; }
    .atlas-card { padding:22px; border:1px solid var(--atlas-border); background:#121e31; border-radius:13px; min-height:150px; }
    .atlas-card h4 { margin:8px 0; font-size:17px; color:#e7eef9; }
    .atlas-card p { color:var(--atlas-muted); font-size:13px; margin:0; line-height:1.6; }
    .atlas-card-icon { font-size:11px; color:#7fe1c5; font-weight:700; letter-spacing:.12em; }
    .atlas-empty { padding:34px; border:1px dashed #34516a; background:#101c2c; border-radius:14px; margin:16px 0; }
    .atlas-empty h3 { font-size:22px; margin:0 0 10px; }
    .atlas-empty p { color:var(--atlas-muted); max-width:650px; }
    .atlas-footer { font-size:11px; color:#8192ad; padding-top:16px; }
    .atlas-source-tag { font-family:monospace; font-size:11px; padding:4px 7px; color:#a4eedb; background:#203c3a; border-radius:5px; display:inline-block; }
    .atlas-answer-rtl { direction:rtl; text-align:right; }
    @media(max-width:800px) { h1 {font-size:2rem!important;} .block-container {padding-top:4.2rem;} .atlas-topline {margin-bottom:20px;} }
    </style>''', unsafe_allow_html=True)


def brand():
    st.markdown('''<div class="atlas-brand"><div class="atlas-symbol">A</div><div>
    <div class="atlas-brand-name">Atlas</div><div class="atlas-kicker">Knowledge workspace</div></div></div>''', unsafe_allow_html=True)


def header(page, backend, ready):
    status = "Gemini configured" if backend == "gemini" and ready else "Offline demo" if backend == "demo" else "API configuration needed"
    st.markdown(f'''<div class="atlas-topline"><span class="atlas-breadcrumb">ATLAS / {escape(page.upper())}</span>
    <span class="atlas-status">● {escape(status)}</span></div>''', unsafe_allow_html=True)


def card(kicker, title, body):
    st.markdown(f'''<div class="atlas-card"><div class="atlas-card-icon">{escape(kicker)}</div>
    <h4>{escape(title)}</h4><p>{escape(body)}</p></div>''', unsafe_allow_html=True)


def empty_state(title, body):
    st.markdown(f'<div class="atlas-empty"><h3>{escape(title)}</h3><p>{escape(body)}</p></div>', unsafe_allow_html=True)
