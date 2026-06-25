import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import time
import joblib

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="NIDS — Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
.stApp { background: #0a0e1a; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e2d3d; }
[data-testid="stSidebar"] * { color: #8b9ab1 !important; }

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2332 100%);
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 20px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.blue::before   { background: linear-gradient(90deg, #3b82f6, transparent); }
.kpi-card.red::before    { background: linear-gradient(90deg, #ef4444, transparent); }
.kpi-card.green::before  { background: linear-gradient(90deg, #22c55e, transparent); }
.kpi-card.purple::before { background: linear-gradient(90deg, #a855f7, transparent); }
.kpi-icon {
    width: 40px; height: 40px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; margin-bottom: 12px;
}
.kpi-icon.blue   { background: #3b82f620; }
.kpi-icon.red    { background: #ef444420; }
.kpi-icon.green  { background: #22c55e20; }
.kpi-icon.purple { background: #a855f720; }
.kpi-label { font-size: 0.75rem; color: #4b6080; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
.kpi-delta { font-size: 0.75rem; margin-top: 4px; }
.kpi-delta.up   { color: #22c55e; }
.kpi-delta.down { color: #ef4444; }

/* Panel cards */
.panel {
    background: #0d1b2a;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.panel-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Alert items */
.alert-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid #1e2d3d;
}
.alert-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
}
.alert-title { font-size: 0.82rem; font-weight: 600; color: #e2e8f0; }
.alert-meta  { font-size: 0.72rem; color: #4b6080; margin-top: 2px; }
.severity-high     { color: #ef4444; font-size: 0.7rem; font-weight: 600; }
.severity-medium   { color: #f59e0b; font-size: 0.7rem; font-weight: 600; }
.severity-low      { color: #22c55e; font-size: 0.7rem; font-weight: 600; }
.severity-critical { color: #a855f7; font-size: 0.7rem; font-weight: 600; }

/* Prediction table */
.pred-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.pred-table th {
    text-align: left; padding: 8px 12px;
    color: #4b6080; font-weight: 500;
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid #1e2d3d;
}
.pred-table td { padding: 10px 12px; border-bottom: 1px solid #111827; color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
.badge-normal   { background: #22c55e20; color: #22c55e; border: 1px solid #22c55e40; }
.badge-dos      { background: #ef444420; color: #ef4444; border: 1px solid #ef444440; }
.badge-probe    { background: #f59e0b20; color: #f59e0b; border: 1px solid #f59e0b40; }
.badge-r2l      { background: #f9731620; color: #f97316; border: 1px solid #f9731640; }
.badge-u2r      { background: #a855f720; color: #a855f7; border: 1px solid #a855f740; }
.badge-malicious { background: #ef444420; color: #ef4444; border: 1px solid #ef444440; }
.badge-benign   { background: #22c55e20; color: #22c55e; border: 1px solid #22c55e40; }

/* Confidence bar */
.conf-bar-wrap { display: flex; align-items: center; gap: 8px; }
.conf-bar { height: 4px; border-radius: 2px; flex: 1; background: #1e2d3d; }
.conf-fill { height: 100%; border-radius: 2px; }

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #4b6080 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 10px 16px !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    margin-bottom: 2px !important;
    width: 100% !important;
    background-image: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #111827 !important;
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] .stButton > button:focus {
    background: #1e3a5f !important;
    color: #3b82f6 !important;
    box-shadow: none !important;
    border: none !important;
}

/* Tabs — dark theme */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e2d3d;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4b6080;
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-size: 0.85rem;
    font-weight: 500;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #0d1b2a !important;
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 16px; }

/* Status badge */
.status-active {
    display: inline-flex; align-items: center; gap: 6px;
    background: #22c55e15; border: 1px solid #22c55e30;
    color: #22c55e; border-radius: 20px;
    padding: 4px 12px; font-size: 0.75rem; font-weight: 600;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #22c55e;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Column overflow guard */
[data-testid="column"] { overflow: hidden; }

/* Compact form inputs in the narrow right column */
[data-testid="column"]:last-child .stSelectbox > div,
[data-testid="column"]:last-child .stNumberInput > div,
[data-testid="column"]:last-child .stTextInput > div {
    font-size: 0.8rem !important;
}
[data-testid="column"]:last-child label {
    font-size: 0.78rem !important;
    margin-bottom: 2px !important;
}

/* Input form */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: #111827 !important;
    border: 1px solid #1e2d3d !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 10px 24px !important;
}

/* Flash alert animation */
.flash-alert { animation: flash 1s infinite; }
@keyframes flash { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* Comparison table */
.cmp-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.cmp-table th {
    text-align: left; padding: 10px 14px;
    color: #4b6080; font-weight: 500;
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 2px solid #1e2d3d;
}
.cmp-table td { padding: 12px 14px; border-bottom: 1px solid #111827; color: #94a3b8; font-family: 'JetBrains Mono', monospace; }
.cmp-table tr:last-child td { border-bottom: none; }
.winner-badge {
    display: inline-block; padding: 2px 10px;
    background: #22c55e20; color: #22c55e;
    border: 1px solid #22c55e40; border-radius: 20px;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.05em; margin-left: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "total_analyzed" not in st.session_state:
    st.session_state.total_analyzed = random.randint(140000, 160000)
if "malicious_count" not in st.session_state:
    st.session_state.malicious_count = random.randint(400, 500)
if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"


# ── Helper functions ──────────────────────────────────────────────────────────
def get_badge(prediction):
    badges = {
        "Normal": "badge-normal", "DoS": "badge-dos",
        "Probe": "badge-probe",   "R2L": "badge-r2l", "U2R": "badge-u2r"
    }
    return badges.get(prediction, "badge-normal")


def get_severity(prediction):
    sev = {"Normal": "LOW", "Probe": "MEDIUM", "DoS": "HIGH",
           "R2L": "HIGH", "U2R": "CRITICAL"}
    return sev.get(prediction, "LOW")


def get_severity_class(sev):
    return {"LOW": "severity-low", "MEDIUM": "severity-medium",
            "HIGH": "severity-high", "CRITICAL": "severity-critical"}.get(sev, "severity-low")


def random_ip():
    return f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def conf_color(conf):
    if conf > 0.8: return "#22c55e"
    if conf > 0.5: return "#f59e0b"
    return "#ef4444"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px 16px;border-bottom:1px solid #1e2d3d;margin-bottom:16px">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#3b82f6,#6366f1);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem">🛡️</div>
            <div>
                <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0">NIDS</div>
                <div style="font-size:0.68rem;color:#4b6080">ML Network Intrusion Detector</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("🏠", "Dashboard"),
        ("📡", "Live Traffic"),
        ("🚨", "Alerts"),
        ("🤖", "ML Model"),
        ("📊", "Traffic Analysis"),
        ("📋", "Logs"),
        ("⚙️", "Settings"),
    ]
    for icon, label in nav_items:
        if st.sidebar.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.active_page = label

    st.sidebar.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="background:#111827;border:1px solid #1e2d3d;border-radius:8px;padding:12px">
        <div style="font-size:0.7rem;color:#4b6080;margin-bottom:4px">STUDENT PROJECT</div>
        <div style="font-size:0.78rem;color:#94a3b8">ML Based Network<br>Intrusion Detection</div>
    </div>
    """, unsafe_allow_html=True)


# ── Top bar ───────────────────────────────────────────────────────────────────
col_title, col_status, col_time = st.columns([2, 1, 1])
with col_title:
    st.markdown("""
    <div style="padding:16px 0 8px">
        <div style="font-size:1.4rem;font-weight:700;color:#e2e8f0">Dashboard</div>
        <div style="font-size:0.8rem;color:#4b6080">Real-time overview of network and threats</div>
    </div>
    """, unsafe_allow_html=True)
with col_status:
    st.markdown(f"""
    <div style="padding:20px 0 8px;text-align:right">
        <span class="status-active"><span class="status-dot"></span> Model Status: Active</span>
    </div>
    """, unsafe_allow_html=True)
with col_time:
    now = datetime.now()
    st.markdown(f"""
    <div style="padding:20px 0 8px;text-align:right">
        <div style="font-size:0.9rem;font-weight:600;color:#e2e8f0;font-family:'JetBrains Mono',monospace">{now.strftime('%I:%M:%S %p')}</div>
        <div style="font-size:0.72rem;color:#4b6080">{now.strftime('%B %d, %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr style="border:none;border-top:1px solid #1e2d3d;margin:0 0 8px">', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dash, tab_compare = st.tabs(["📊 Dashboard", "🤖 Model Comparison"])

# ── NSL-KDD preset profiles (all 41 features) ─────────────────────────────────
_PROTOCOLS = ["tcp", "udp", "icmp"]
_SERVICES  = ["http", "ftp", "ssh", "smtp", "dns", "ftp_data", "telnet", "private", "domain_u", "pop_3", "other"]
_FLAGS     = ["SF", "S0", "REJ", "RSTO", "SH", "RSTR", "S1", "S2", "S3", "OTH"]

PRESETS = {
    "Custom": {},
    "Normal Traffic": {
        "protocol": "tcp",   "service": "http",    "flag": "SF",
        "src_bytes": 232,    "dst_bytes": 8153,    "logged_in": 1, "duration": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "num_compromised": 0, "root_shell": 0, "su_attempted": 0,
        "num_root": 0, "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 8,   "srv_count": 8,   "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0,  "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.03, "dst_host_srv_diff_host_rate": 0.04,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
    },
    "DoS Attack": {
        "protocol": "tcp",   "service": "http",    "flag": "S0",
        "src_bytes": 0,      "dst_bytes": 0,       "logged_in": 0, "duration": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "num_compromised": 0, "root_shell": 0, "su_attempted": 0,
        "num_root": 0, "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 511, "srv_count": 511, "serror_rate": 1.0, "srv_serror_rate": 1.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0,  "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
    },
    "Port Scan (Probe)": {
        "protocol": "tcp",   "service": "private", "flag": "REJ",
        "src_bytes": 0,      "dst_bytes": 0,       "logged_in": 0, "duration": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "num_compromised": 0, "root_shell": 0, "su_attempted": 0,
        "num_root": 0, "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1,   "srv_count": 18,  "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 1.0, "srv_rerror_rate": 0.06, "same_srv_rate": 0.06,
        "diff_srv_rate": 0.06, "srv_diff_host_rate": 1.0,
        "dst_host_count": 255, "dst_host_srv_count": 18,
        "dst_host_same_srv_rate": 0.07, "dst_host_diff_srv_rate": 0.06,
        "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 1.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.05, "dst_host_srv_rerror_rate": 0.0,
    },
    "R2L Attack": {
        "protocol": "udp",   "service": "ftp",     "flag": "SF",
        "src_bytes": 0,      "dst_bytes": 0,       "logged_in": 0, "duration": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 5, "num_compromised": 0, "root_shell": 0, "su_attempted": 0,
        "num_root": 0, "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 4,   "srv_count": 4,   "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 4,
        "dst_host_same_srv_rate": 0.02, "dst_host_diff_srv_rate": 0.06,
        "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
    },
    "U2R Attack": {
        "protocol": "tcp",   "service": "telnet",  "flag": "SF",
        "src_bytes": 1408,   "dst_bytes": 3664,    "logged_in": 1, "duration": 16,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 2,
        "num_failed_logins": 0, "num_compromised": 4, "root_shell": 1, "su_attempted": 0,
        "num_root": 4, "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1,   "srv_count": 1,   "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 6,   "dst_host_srv_count": 6,
        "dst_host_same_srv_rate": 1.0,  "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.17, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
    },
}

def _idx(lst, val, default=0):
    try:
        return lst.index(val)
    except ValueError:
        return default


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    # ── KPI Cards ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    benign_count = st.session_state.total_analyzed - st.session_state.malicious_count
    accuracy = 77.65

    with k1:
        st.markdown(f"""
        <div class="kpi-card blue">
            <div class="kpi-icon blue">📊</div>
            <div class="kpi-label">Total Packets Analyzed</div>
            <div class="kpi-value">{st.session_state.total_analyzed:,}</div>
            <div class="kpi-delta up">↑ 12.5% from last hour</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card red">
            <div class="kpi-icon red">🛡️</div>
            <div class="kpi-label">Malicious Detections</div>
            <div class="kpi-value">{st.session_state.malicious_count:,}</div>
            <div class="kpi-delta down">↑ 18.7% from last hour</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card green">
            <div class="kpi-icon green">✅</div>
            <div class="kpi-label">Benign Detections</div>
            <div class="kpi-value">{benign_count:,}</div>
            <div class="kpi-delta up">↑ 11.3% from last hour</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card purple">
            <div class="kpi-icon purple">🎯</div>
            <div class="kpi-label">Detection Accuracy</div>
            <div class="kpi-value">{accuracy}%</div>
            <div class="kpi-delta up">↑ 2.4% from yesterday</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    chart_col, donut_col, alerts_col = st.columns([1.8, 1, 1])

    with chart_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📡 Traffic Overview (Last 1 Hour)</div>', unsafe_allow_html=True)

        times     = [datetime.now() - timedelta(minutes=i) for i in range(60, 0, -1)]
        total     = [random.randint(2000, 5000) for _ in range(60)]
        benign_t  = [int(t * random.uniform(0.85, 0.95)) for t in total]
        malicious = [t - b for t, b in zip(total, benign_t)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=total, name="Total Traffic",
                                 line=dict(color="#3b82f6", width=2), fill=None))
        fig.add_trace(go.Scatter(x=times, y=benign_t, name="Benign",
                                 line=dict(color="#22c55e", width=1.5)))
        fig.add_trace(go.Scatter(x=times, y=malicious, name="Malicious",
                                 line=dict(color="#ef4444", width=1.5)))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=220, margin=dict(t=10, b=10, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1, font=dict(color="#94a3b8", size=11)),
            xaxis=dict(showgrid=False, color="#4b6080", tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#1e2d3d", color="#4b6080", tickfont=dict(size=10)),
            font=dict(color="#94a3b8")
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with donut_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🎯 Attack Distribution</div>', unsafe_allow_html=True)

        attack_labels = ["DoS", "Probe", "R2L", "U2R", "Normal"]
        attack_values = [41.4, 23.1, 15.7, 9.3, 10.5]
        attack_colors = ["#ef4444", "#f59e0b", "#f97316", "#a855f7", "#22c55e"]

        fig2 = go.Figure(go.Pie(
            labels=attack_labels, values=attack_values,
            hole=0.6, marker_colors=attack_colors,
            textinfo="none",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=220, margin=dict(t=10, b=10, l=0, r=0),
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8", size=10), orientation="v", x=1, y=0.5),
            annotations=[dict(
                text=f"<b style='color:#e2e8f0'>{st.session_state.malicious_count}</b>"
                     f"<br><span style='color:#4b6080;font-size:10px'>Total</span>",
                x=0.5, y=0.5, font_size=14, showarrow=False, font=dict(color="#e2e8f0")
            )]
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with alerts_col:
        st.markdown('<div class="panel" style="height:280px;overflow-y:auto">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🚨 Recent Alerts</div>', unsafe_allow_html=True)

        alerts = [
            ("DoS Attack Detected",  random_ip(), "High",     "#ef4444"),
            ("Port Scan Detected",   random_ip(), "Medium",   "#f59e0b"),
            ("R2L Attack Detected",  random_ip(), "High",     "#ef4444"),
            ("DoS Attack Detected",  random_ip(), "High",     "#ef4444"),
            ("Unusual Activity",     random_ip(), "Low",      "#22c55e"),
            ("U2R Attempt Detected", random_ip(), "Critical", "#a855f7"),
        ]
        for title, ip, sev, color in alerts:
            sev_cls = get_severity_class(sev.upper())
            st.markdown(f"""
            <div class="alert-item">
                <div class="alert-dot" style="background:{color}"></div>
                <div>
                    <div class="alert-title">{title}</div>
                    <div class="alert-meta">Source IP: {ip} &nbsp;
                        <span class="{sev_cls}">Severity: {sev}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Prediction Table + Input Form ─────────────────────────────────────────
    table_col, form_col = st.columns([2.6, 1])

    with form_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🔍 Analyze Connection</div>', unsafe_allow_html=True)

        preset = st.selectbox("⚡ Quick Test Profile", list(PRESETS.keys()))
        p = PRESETS[preset]

        threshold = st.slider("⚠️ Alert Threshold", 0.0, 1.0, 0.5, 0.05)

        st.markdown("<hr style='border:none;border-top:1px solid #1e2d3d;margin:8px 0'>",
                    unsafe_allow_html=True)

        protocol  = st.selectbox("Protocol", _PROTOCOLS,
                                  index=_idx(_PROTOCOLS, p.get("protocol", "tcp")))
        service   = st.selectbox("Service",  _SERVICES,
                                  index=_idx(_SERVICES,  p.get("service",  "http")))
        flag      = st.selectbox("Flag",     _FLAGS,
                                  index=_idx(_FLAGS,     p.get("flag",     "SF")))
        src_bytes = st.number_input("Src Bytes",   min_value=0, value=int(p.get("src_bytes", 232)))
        dst_bytes = st.number_input("Dst Bytes",   min_value=0, value=int(p.get("dst_bytes", 8153)))
        logged_in = st.selectbox("Logged In", [1, 0],
                                  index=0 if p.get("logged_in", 1) == 1 else 1)
        duration  = st.number_input("Duration (s)", min_value=0, value=int(p.get("duration", 0)))

        analyze = st.button("🔍 Analyze Traffic", use_container_width=True)

        if analyze:
            payload = {
                "duration":        duration,      "protocol_type":   protocol,
                "service":         service,        "flag":            flag,
                "src_bytes":       src_bytes,      "dst_bytes":       dst_bytes,
                "logged_in":       logged_in,
                "land":                        p.get("land", 0),
                "wrong_fragment":              p.get("wrong_fragment", 0),
                "urgent":                      p.get("urgent", 0),
                "hot":                         p.get("hot", 0),
                "num_failed_logins":           p.get("num_failed_logins", 0),
                "num_compromised":             p.get("num_compromised", 0),
                "root_shell":                  p.get("root_shell", 0),
                "su_attempted":                p.get("su_attempted", 0),
                "num_root":                    p.get("num_root", 0),
                "num_file_creations":          p.get("num_file_creations", 0),
                "num_shells":                  p.get("num_shells", 0),
                "num_access_files":            p.get("num_access_files", 0),
                "num_outbound_cmds":           p.get("num_outbound_cmds", 0),
                "is_host_login":               p.get("is_host_login", 0),
                "is_guest_login":              p.get("is_guest_login", 0),
                "count":                       p.get("count", 8),
                "srv_count":                   p.get("srv_count", 8),
                "serror_rate":                 p.get("serror_rate", 0.0),
                "srv_serror_rate":             p.get("srv_serror_rate", 0.0),
                "rerror_rate":                 p.get("rerror_rate", 0.0),
                "srv_rerror_rate":             p.get("srv_rerror_rate", 0.0),
                "same_srv_rate":               p.get("same_srv_rate", 1.0),
                "diff_srv_rate":               p.get("diff_srv_rate", 0.0),
                "srv_diff_host_rate":          p.get("srv_diff_host_rate", 0.0),
                "dst_host_count":              p.get("dst_host_count", 255),
                "dst_host_srv_count":          p.get("dst_host_srv_count", 255),
                "dst_host_same_srv_rate":      p.get("dst_host_same_srv_rate", 1.0),
                "dst_host_diff_srv_rate":      p.get("dst_host_diff_srv_rate", 0.0),
                "dst_host_same_src_port_rate": p.get("dst_host_same_src_port_rate", 0.0),
                "dst_host_srv_diff_host_rate": p.get("dst_host_srv_diff_host_rate", 0.0),
                "dst_host_serror_rate":        p.get("dst_host_serror_rate", 0.0),
                "dst_host_srv_serror_rate":    p.get("dst_host_srv_serror_rate", 0.0),
                "dst_host_rerror_rate":        p.get("dst_host_rerror_rate", 0.0),
                "dst_host_srv_rerror_rate":    p.get("dst_host_srv_rerror_rate", 0.0),
            }
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                    pred   = result["prediction"]
                    conf   = result["confidence"]
                    risk   = result["risk_level"]
                    desc   = result["description"]

                    risk_colors = {"LOW": "#22c55e", "MEDIUM": "#f59e0b",
                                   "HIGH": "#ef4444", "CRITICAL": "#a855f7"}
                    rc = risk_colors.get(risk, "#94a3b8")

                    is_threat    = pred != "Normal"
                    above_thresh = conf >= threshold

                    if is_threat and above_thresh:
                        # High-confidence malicious — flashing red alert
                        st.markdown(f"""
                        <div class="flash-alert" style="background:#1a0a0a;border:2px solid #ef4444;border-radius:10px;padding:16px;margin-top:12px">
                            <div style="font-size:0.7rem;color:#ef444499;margin-bottom:4px;letter-spacing:0.08em">🚨 THREAT DETECTED</div>
                            <div style="font-size:1.3rem;font-weight:700;color:#ef4444">{pred}</div>
                            <div style="color:#ef4444;font-size:0.75rem;font-weight:600;margin:4px 0">{risk} RISK</div>
                            <div style="font-size:0.75rem;color:#4b6080;margin-top:8px">{desc}</div>
                            <div style="font-size:0.75rem;color:#94a3b8;margin-top:8px">
                                Confidence: <strong style="color:#ef4444">{conf*100:.1f}%</strong>
                                &nbsp;·&nbsp; Threshold: {threshold*100:.0f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif is_threat and not above_thresh:
                        # Below threshold — yellow caution
                        st.markdown(f"""
                        <div style="background:#1a1400;border:1px solid #f59e0b80;border-radius:10px;padding:16px;margin-top:12px">
                            <div style="font-size:0.7rem;color:#f59e0b99;margin-bottom:4px;letter-spacing:0.08em">⚠️ LOW CONFIDENCE</div>
                            <div style="font-size:1.1rem;font-weight:700;color:#f59e0b">{pred}</div>
                            <div style="font-size:0.78rem;color:#f59e0b;margin:6px 0 8px">Manual review recommended</div>
                            <div style="font-size:0.75rem;color:#4b6080">{desc}</div>
                            <div style="font-size:0.75rem;color:#94a3b8;margin-top:8px">
                                Confidence: <strong style="color:#f59e0b">{conf*100:.1f}%</strong>
                                &nbsp;·&nbsp; Threshold: {threshold*100:.0f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Benign result
                        st.markdown(f"""
                        <div style="background:#111827;border:1px solid {rc}40;border-radius:10px;padding:16px;margin-top:12px">
                            <div style="font-size:0.7rem;color:#4b6080;margin-bottom:4px">PREDICTION RESULT</div>
                            <div style="font-size:1.3rem;font-weight:700;color:{rc}">{pred}</div>
                            <div style="color:{rc};font-size:0.75rem;font-weight:600;margin:4px 0">{risk} RISK</div>
                            <div style="font-size:0.75rem;color:#4b6080;margin-top:8px">{desc}</div>
                            <div style="font-size:0.75rem;color:#94a3b8;margin-top:8px">Confidence: <strong style="color:{rc}">{conf*100:.1f}%</strong></div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.session_state.predictions.insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "src_ip": random_ip(), "dst_ip": random_ip(),
                        "protocol": protocol.upper(),
                        "prediction": pred, "confidence": conf
                    })
                    st.session_state.total_analyzed += 1
                    if pred != "Normal":
                        st.session_state.malicious_count += 1

            except Exception as e:
                st.error(f"API error: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    with table_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📋 Latest Traffic Predictions</div>', unsafe_allow_html=True)

        sim_preds  = []
        attack_types = ["Normal", "Normal", "Normal", "DoS", "Probe", "Normal", "R2L", "Normal", "DoS", "U2R"]
        for i in range(10):
            pred = attack_types[i % len(attack_types)]
            sim_preds.append({
                "time": (datetime.now() - timedelta(seconds=i*30)).strftime("%H:%M:%S"),
                "src_ip": random_ip(), "dst_ip": random_ip(),
                "protocol": random.choice(["TCP", "UDP", "ICMP"]),
                "prediction": pred,
                "confidence": round(random.uniform(0.75, 0.99), 2)
            })

        all_preds = (st.session_state.predictions + sim_preds)[:12]

        rows = ""
        for row in all_preds:
            pred  = row["prediction"]
            conf  = row["confidence"]
            badge = get_badge(pred)
            label = "Malicious" if pred != "Normal" else "Benign"
            lb    = "badge-malicious" if pred != "Normal" else "badge-benign"
            cc    = conf_color(conf)
            bar_w = int(conf * 100)
            rows += f"""
            <tr>
                <td style="color:#4b6080">{row['time']}</td>
                <td>{row['src_ip']}</td>
                <td>{row['dst_ip']}</td>
                <td style="color:#94a3b8">{row['protocol']}</td>
                <td><span class="badge {lb}">{label}</span></td>
                <td><span class="badge {badge}">{pred}</span></td>
                <td>
                    <div class="conf-bar-wrap">
                        <span style="color:{cc};min-width:36px">{int(conf*100)}%</span>
                        <div class="conf-bar"><div class="conf-fill" style="width:{bar_w}%;background:{cc}"></div></div>
                    </div>
                </td>
            </tr>"""

        st.markdown(f"""
        <table class="pred-table">
            <thead><tr>
                <th>Time</th><th>Source IP</th><th>Destination IP</th>
                <th>Protocol</th><th>Status</th><th>Attack Type</th><th>Confidence</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Intelligence ────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔬 Model Intelligence — Top 15 Feature Importances</div>',
                unsafe_allow_html=True)

    try:
        _xgb    = joblib.load("models/xgboost.pkl")
        _fnames = joblib.load("models/feature_names.pkl")
        _imp    = _xgb.feature_importances_

        fi_df = (
            pd.DataFrame({"feature": _fnames, "importance": _imp})
            .sort_values("importance", ascending=False)
            .head(15)
            .sort_values("importance", ascending=True)
        )

        fig_fi = go.Figure(go.Bar(
            x=fi_df["importance"],
            y=fi_df["feature"],
            orientation="h",
            marker=dict(
                color=fi_df["importance"],
                colorscale=[[0, "#1e3a5f"], [1, "#3b82f6"]],
                showscale=False,
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1b2a",
            height=380, margin=dict(t=8, b=8, l=0, r=24),
            xaxis=dict(showgrid=True, gridcolor="#1e2d3d", color="#4b6080",
                       tickfont=dict(size=10), zeroline=False),
            yaxis=dict(showgrid=False, color="#94a3b8",
                       tickfont=dict(size=11), automargin=True),
            font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    except FileNotFoundError:
        st.markdown(
            '<div style="color:#4b6080;padding:24px;text-align:center;font-size:0.82rem">'
            'Train the model first — <code>models/xgboost.pkl</code> or '
            '<code>models/feature_names.pkl</code> not found.</div>',
            unsafe_allow_html=True,
        )
    except Exception as _e:
        st.markdown(
            f'<div style="color:#ef4444;padding:12px;font-size:0.8rem">Could not load feature importances: {_e}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Hardcoded training results ─────────────────────────────────────────────
    _metrics = {
        "Accuracy":    [0.7612, 0.7765],
        "F1 Weighted": [0.7149, 0.7463],
        "DoS Recall":  [0.81,   0.80],
        "Probe Recall":[0.68,   0.73],
        "R2L Recall":  [0.02,   0.13],
        "U2R Recall":  [0.06,   0.28],
    }

    # ── Summary header ─────────────────────────────────────────────────────────
    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        st.markdown("""
        <div style="margin-bottom:20px">
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0">RandomForest vs XGBoost</div>
            <div style="font-size:0.8rem;color:#4b6080;margin-top:4px">
                Trained on NSL-KDD dataset · 5-class classification · SMOTE balanced
            </div>
        </div>
        """, unsafe_allow_html=True)
    with hdr_right:
        st.markdown("""
        <div style="text-align:right;padding-top:4px">
            <span style="font-size:0.72rem;color:#4b6080">DEPLOYED MODEL</span><br>
            <span style="font-size:0.95rem;font-weight:700;color:#3b82f6">XGBoost</span>
            <span class="winner-badge">★ WINNER</span>
        </div>
        """, unsafe_allow_html=True)

    cmp_table_col, cmp_chart_col = st.columns([1, 1.6])

    # ── Metrics table ──────────────────────────────────────────────────────────
    with cmp_table_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📋 Metrics Breakdown</div>', unsafe_allow_html=True)

        rf_vals  = [_metrics[m][0] for m in _metrics]
        xgb_vals = [_metrics[m][1] for m in _metrics]

        table_rows = ""
        for metric, (rf_v, xgb_v) in zip(_metrics.keys(), zip(rf_vals, xgb_vals)):
            winner_col = "#3b82f6" if xgb_v >= rf_v else "#94a3b8"
            rf_col     = "#94a3b8" if xgb_v >= rf_v else "#3b82f6"
            delta      = xgb_v - rf_v
            delta_html = (
                f'<span style="color:#22c55e;font-size:0.68rem">▲ {delta:+.4f}</span>'
                if delta > 0 else
                f'<span style="color:#ef4444;font-size:0.68rem">▼ {delta:.4f}</span>'
            )
            table_rows += f"""
            <tr>
                <td style="color:#94a3b8;font-family:Inter,sans-serif">{metric}</td>
                <td style="color:{rf_col}">{rf_v:.4f}</td>
                <td style="color:{winner_col};font-weight:600">{xgb_v:.4f}</td>
                <td>{delta_html}</td>
            </tr>"""

        st.markdown(f"""
        <table class="cmp-table">
            <thead><tr>
                <th>Metric</th>
                <th>RandomForest</th>
                <th>XGBoost <span class="winner-badge">★ WINNER</span></th>
                <th>Δ</th>
            </tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:16px;padding:12px;background:#111827;border-radius:8px;border:1px solid #1e2d3d">
            <div style="font-size:0.72rem;color:#4b6080;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em">Key Takeaway</div>
            <div style="font-size:0.78rem;color:#94a3b8;line-height:1.5">
                XGBoost outperforms RandomForest on overall accuracy (+1.5%) and F1 (+3.1%).
                The biggest gap is on <span style="color:#f97316">R2L</span> (+11pp) and
                <span style="color:#a855f7">U2R</span> (+22pp) recall — rare attack classes
                where boosting's iterative correction has the most impact.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Grouped bar chart ──────────────────────────────────────────────────────
    with cmp_chart_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 Side-by-Side Performance</div>', unsafe_allow_html=True)

        metric_names = list(_metrics.keys())
        rf_scores    = [_metrics[m][0] for m in metric_names]
        xgb_scores   = [_metrics[m][1] for m in metric_names]

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="RandomForest",
            x=metric_names,
            y=rf_scores,
            marker_color="#4b6080",
            marker_line_color="#1e2d3d",
            marker_line_width=1,
            hovertemplate="%{x}: %{y:.4f}<extra>RandomForest</extra>",
        ))
        fig_cmp.add_trace(go.Bar(
            name="XGBoost ★",
            x=metric_names,
            y=xgb_scores,
            marker_color="#3b82f6",
            marker_line_color="#1e3a5f",
            marker_line_width=1,
            hovertemplate="%{x}: %{y:.4f}<extra>XGBoost</extra>",
        ))
        fig_cmp.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0d1b2a",
            height=360,
            margin=dict(t=10, b=10, l=0, r=0),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                font=dict(color="#94a3b8", size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                showgrid=False, color="#4b6080",
                tickfont=dict(size=11), tickangle=-15,
            ),
            yaxis=dict(
                showgrid=True, gridcolor="#1e2d3d",
                color="#4b6080", tickfont=dict(size=10),
                range=[0, 1.05], tickformat=".0%",
            ),
            font=dict(color="#94a3b8"),
            bargap=0.25,
            bargroupgap=0.05,
        )
        # Reference line at 1.0
        fig_cmp.add_hline(y=1.0, line_dash="dot", line_color="#1e2d3d", line_width=1)
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

        # Per-class recall spotlight
        st.markdown("""
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px">
            <div style="background:#111827;border:1px solid #1e2d3d;border-radius:8px;padding:10px 14px;flex:1;min-width:100px">
                <div style="font-size:0.68rem;color:#4b6080;margin-bottom:4px">DoS RECALL</div>
                <div style="font-size:0.9rem;font-weight:700;color:#ef4444">RF 81% · XGB 80%</div>
                <div style="font-size:0.68rem;color:#4b6080">Both strong on majority class</div>
            </div>
            <div style="background:#111827;border:1px solid #1e2d3d;border-radius:8px;padding:10px 14px;flex:1;min-width:100px">
                <div style="font-size:0.68rem;color:#4b6080;margin-bottom:4px">R2L RECALL</div>
                <div style="font-size:0.9rem;font-weight:700;color:#f97316">RF 2% · XGB 13%</div>
                <div style="font-size:0.68rem;color:#22c55e">XGBoost +11pp improvement</div>
            </div>
            <div style="background:#111827;border:1px solid #1e2d3d;border-radius:8px;padding:10px 14px;flex:1;min-width:100px">
                <div style="font-size:0.68rem;color:#4b6080;margin-bottom:4px">U2R RECALL</div>
                <div style="font-size:0.9rem;font-weight:700;color:#a855f7">RF 6% · XGB 28%</div>
                <div style="font-size:0.68rem;color:#22c55e">XGBoost +22pp improvement</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#1e2d3d;font-size:0.72rem;padding:16px 0">
    NIDS · ML-Based Network Intrusion Detection System · Built with XGBoost · NSL-KDD Dataset · FastAPI · Streamlit
</div>
""", unsafe_allow_html=True)
