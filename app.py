from __future__ import annotations

import base64
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from utils.tracking import (
    apply_focus_effect,
    choose_target_from_click,
    draw_boxes,
    enhance_low_light,
    find_bbox_for_track,
    find_bbox_and_id_by_proximity,
    get_candidate_boxes,
    load_model,
    AppearanceMatcher,
)
from utils.video import iter_frames, make_video_writer, open_video, read_frame_at

# ─── Page Config ─────────────────────────────────────────────────────────────
LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

st.set_page_config(
    page_title="Bulls-Eye",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Import Google Font ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ─────────────────────────────────────────────────── */
:root {
    --bg-primary: #0E1117;
    --bg-secondary: #161B22;
    --bg-tertiary: #1C2128;
    --bg-card: rgba(22, 27, 34, 0.65);
    --accent: #00D4FF;
    --accent-dim: rgba(0, 212, 255, 0.15);
    --accent-glow: rgba(0, 212, 255, 0.35);
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --text-muted: #484F58;
    --border: rgba(240, 246, 252, 0.06);
    --border-accent: rgba(0, 212, 255, 0.2);
    --success: #3FB950;
    --warning: #D29922;
    --error: #F85149;
    --radius: 12px;
    --radius-sm: 8px;
    --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global ─────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0E1117 0%, #0D1520 35%, #0E1117 100%) !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(22,27,34,0.97) 0%, rgba(13,21,32,0.98) 100%) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    transition: width 0.35s cubic-bezier(0.4,0,0.2,1), min-width 0.35s cubic-bezier(0.4,0,0.2,1) !important;
    overflow: hidden !important;
}

/* Hide the native Streamlit sidebar collapse button */
[data-testid="stSidebar"] button[aria-label="Close"],
[data-testid="collapsedControl"],
button[kind="headerNoPadding"],
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* Collapsed sidebar state */
[data-testid="stSidebar"].sidebar-collapsed {
    width: 80px !important;
    min-width: 80px !important;
}

[data-testid="stSidebar"].sidebar-collapsed [data-testid="stSidebarContent"] {
    overflow: hidden !important;
    padding: 0.5rem 0 !important;
}

/* Hide ALL content inside collapsed sidebar except brand-header */
[data-testid="stSidebar"].sidebar-collapsed [data-testid="stSidebarContent"] [data-testid="stVerticalBlock"] > div {
    display: none !important;
}

/* Show ONLY the first block (brand-header) */
[data-testid="stSidebar"].sidebar-collapsed [data-testid="stSidebarContent"] [data-testid="stVerticalBlock"] > div:first-child {
    display: block !important;
}

/* Extra safety: hide specific Streamlit widget types in collapsed mode */
[data-testid="stSidebar"].sidebar-collapsed [data-testid="stRadio"],
[data-testid="stSidebar"].sidebar-collapsed [data-testid="stFileUploader"],
[data-testid="stSidebar"].sidebar-collapsed [data-testid="stSlider"],
[data-testid="stSidebar"].sidebar-collapsed [data-testid="stCheckbox"],
[data-testid="stSidebar"].sidebar-collapsed [data-testid="stExpander"],
[data-testid="stSidebar"].sidebar-collapsed .stButton,
[data-testid="stSidebar"].sidebar-collapsed .section-label,
[data-testid="stSidebar"].sidebar-collapsed .sidebar-collapsible {
    display: none !important;
}

/* Logo always visible, styled as toggle button */
.brand-header {
    cursor: pointer !important;
    user-select: none;
}

.brand-header img, .brand-header .logo-emoji {
    transition: transform 0.3s ease, filter 0.3s ease;
}

.brand-header:hover img, .brand-header:hover .logo-emoji {
    transform: scale(1.1);
    filter: drop-shadow(0 0 20px var(--accent)) brightness(1.1);
}

/* In collapsed state: center the logo, hide title/subtitle */
[data-testid="stSidebar"].sidebar-collapsed .brand-header {
    padding: 1.2rem 0;
    border-bottom: none;
    margin-bottom: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

[data-testid="stSidebar"].sidebar-collapsed .brand-title,
[data-testid="stSidebar"].sidebar-collapsed .brand-subtitle,
[data-testid="stSidebar"].sidebar-collapsed .toggle-hint {
    display: none;
}

[data-testid="stSidebar"].sidebar-collapsed .brand-header img {
    width: 44px;
    height: 44px;
    margin-bottom: 0;
}

[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
    font-size: 0.88rem;
    color: var(--text-secondary);
}

/* ── Sidebar Header ─────────────────────────────────────────────────── */
.brand-header {
    text-align: center;
    padding: 1.2rem 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

.brand-header img {
    width: 64px;
    height: 64px;
    margin-bottom: 0.6rem;
    filter: drop-shadow(0 0 12px var(--accent-glow));
    transition: all 0.3s ease;
}

.brand-title {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 0.35em;
    color: var(--text-primary);
    margin: 0;
    background: linear-gradient(135deg, #E6EDF3 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    transition: opacity 0.2s ease;
}

.brand-subtitle {
    font-size: 0.7rem;
    font-weight: 400;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 0.25rem;
    transition: opacity 0.2s ease;
}

/* Toggle hint arrow */
.toggle-hint {
    font-size: 0.6rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
    opacity: 0.5;
    transition: opacity 0.2s ease;
}
.brand-header:hover .toggle-hint {
    opacity: 1;
}

/* ── Section Dividers ───────────────────────────────────────────────── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 0.8rem 0 0.4rem;
    border-top: 1px solid var(--border);
    margin-top: 0.6rem;
}

.section-label .sec-icon {
    font-size: 0.82rem;
}

/* ── Buttons ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1rem !important;
    transition: var(--transition) !important;
    text-transform: none !important;
    letter-spacing: 0.02em !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 16px var(--accent-glow), inset 0 0 8px rgba(0,212,255,0.05) !important;
    transform: translateY(-1px);
}

[data-testid="stSidebar"] .stButton > button:active {
    transform: translateY(0px);
}

/* Primary buttons */
[data-testid="stSidebar"] button[kind="primary"],
button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2) 0%, rgba(0,212,255,0.08) 100%) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.35) 0%, rgba(0,212,255,0.15) 100%) !important;
    box-shadow: 0 0 24px var(--accent-glow) !important;
}

/* Main area buttons */
.stMainBlockContainer .stButton > button {
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: var(--transition) !important;
}

.stMainBlockContainer .stButton > button:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 16px var(--accent-glow) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(63,185,80,0.15) 0%, rgba(63,185,80,0.05) 100%) !important;
    border: 1px solid var(--success) !important;
    color: var(--success) !important;
    font-weight: 600 !important;
}

.stDownloadButton > button:hover {
    background: rgba(63,185,80,0.25) !important;
    box-shadow: 0 0 20px rgba(63,185,80,0.3) !important;
}

/* ── Sliders ────────────────────────────────────────────────────────── */
[data-testid="stSlider"] label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}

[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* ── Checkboxes ─────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label {
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    color: var(--text-secondary) !important;
    transition: var(--transition);
}

[data-testid="stCheckbox"] label:hover {
    color: var(--text-primary) !important;
}

/* ── File Uploader ──────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border-accent) !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem !important;
    background: rgba(0, 212, 255, 0.02) !important;
    transition: var(--transition);
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: rgba(0, 212, 255, 0.04) !important;
}

[data-testid="stFileUploader"] section > button {
    color: var(--accent) !important;
}

[data-testid="stFileUploader"] small {
    color: var(--text-muted) !important;
}

/* ── Expander ───────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--bg-card) !important;
    backdrop-filter: blur(8px) !important;
}

[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}

[data-testid="stExpander"] summary:hover {
    color: var(--text-primary) !important;
}

/* ── Status Badges ──────────────────────────────────────────────────── */
.status-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    padding: 0.65rem 0;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.03em;
    border: 1px solid;
    line-height: 1.4;
}

.status-pill.tracking {
    background: rgba(63, 185, 80, 0.1);
    border-color: rgba(63, 185, 80, 0.3);
    color: var(--success);
}

.status-pill.idle {
    background: rgba(210, 153, 34, 0.1);
    border-color: rgba(210, 153, 34, 0.3);
    color: var(--warning);
}

.status-pill.frame {
    background: rgba(0, 212, 255, 0.08);
    border-color: rgba(0, 212, 255, 0.2);
    color: var(--accent);
}

.status-pill.matcher {
    background: rgba(139, 148, 158, 0.08);
    border-color: rgba(139, 148, 158, 0.2);
    color: var(--text-secondary);
}

.pulse-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}

.pulse-dot.green { background: var(--success); }
.pulse-dot.amber { background: var(--warning); }

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 currentColor; }
    50% { opacity: 0.6; box-shadow: 0 0 6px 2px currentColor; }
}

/* ── Main area title ────────────────────────────────────────────────── */
.main-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.75rem;
}

/* ── Preview frame wrapper ──────────────────────────────────────────── */
.preview-wrapper {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--bg-secondary);
    box-shadow: 0 4px 32px rgba(0,0,0,0.3), 0 0 1px var(--border);
    transition: var(--transition);
}

.preview-wrapper:hover {
    border-color: var(--border-accent);
    box-shadow: 0 4px 40px rgba(0,0,0,0.4), 0 0 30px rgba(0,212,255,0.03);
}

/* ── Info/Warning/Success/Error alerts ──────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-size: 0.82rem !important;
    font-family: 'Inter', sans-serif !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress bar ───────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background-color: var(--accent) !important;
    border-radius: 4px !important;
}

/* ── Tooltip hint text ──────────────────────────────────────────────── */
.click-hint {
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-muted);
    padding: 0.5rem 0;
    font-style: italic;
}

/* ── Hide streamlit branding ────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { visibility: hidden; height: 0; }
footer { visibility: hidden; }

/* ── Divider override ───────────────────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    margin: 0.5rem 0 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── Sidebar Toggle JavaScript ───────────────────────────────────────────────
import streamlit.components.v1 as components

SIDEBAR_TOGGLE_JS = """
<script>
(function() {
    const doc = window.parent.document;
    function initSidebarToggle() {
        const sidebar = doc.querySelector('[data-testid="stSidebar"]');
        const toggle = doc.getElementById('sidebar-toggle');
        if (!sidebar || !toggle) {
            setTimeout(initSidebarToggle, 300);
            return;
        }
        // Restore state from localStorage
        if (localStorage.getItem('bullseye_sidebar_collapsed') === 'true') {
            sidebar.classList.add('sidebar-collapsed');
        }
        // Only attach once
        if (toggle.dataset.toggleBound) return;
        toggle.dataset.toggleBound = 'true';
        toggle.addEventListener('click', function(e) {
            e.stopPropagation();
            e.preventDefault();
            sidebar.classList.toggle('sidebar-collapsed');
            const collapsed = sidebar.classList.contains('sidebar-collapsed');
            localStorage.setItem('bullseye_sidebar_collapsed', collapsed);
            // Update hint text
            const hint = toggle.querySelector('.toggle-hint');
            if (hint) {
                hint.textContent = collapsed ? 'click to expand' : 'click logo to collapse';
            }
        });
        // Update hint on load
        const hint = toggle.querySelector('.toggle-hint');
        if (hint && sidebar.classList.contains('sidebar-collapsed')) {
            hint.textContent = 'click to expand';
        }
    }
    // Run after a short delay to let Streamlit render
    setTimeout(initSidebarToggle, 500);
    // Also observe for Streamlit re-renders
    const observer = new MutationObserver(function() {
        initSidebarToggle();
    });
    observer.observe(doc.body, { childList: true, subtree: true });
})();
</script>
"""
components.html(SIDEBAR_TOGGLE_JS, height=0, width=0)


# ─── Helper: encode logo to base64 for HTML ─────────────────────────────────
def get_logo_b64() -> str:
    if LOGO_PATH.exists():
        data = LOGO_PATH.read_bytes()
        return base64.b64encode(data).decode()
    return ""


# ─── Helper functions (unchanged logic) ─────────────────────────────────────
def check_lap() -> None:
    try:
        import lap  # noqa: F401
    except ModuleNotFoundError:
        try:
            import lapx  # noqa: F401
        except ModuleNotFoundError:
            st.error("Missing dependency 'lap' or 'lapx' required for ByteTrack. Install with: pip install lapx")
            st.stop()


def init_state() -> None:
    defaults = {
        "preview_started": False,
        "playing": False,
        "current_frame": 0,
        "selected_track_id": None,
        "selected_point": None,
        "selection_frame": None,
        "last_frame_index": None,
        "last_display_frame": None,
        "pending_click": None,
        "pending_click_frame": None,
        "last_click": None,
        "last_bbox": None,
        "lock_target": False,
        "target_embedding": None,
        # Live camera state
        "live_frozen_frame": None,
        "live_selected_track_id": None,
        "live_target_bbox": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def ensure_preview_model(reset: bool = False):
    if reset or "preview_model" not in st.session_state:
        st.session_state.preview_model = load_model("yolov8n.pt")
    return st.session_state.preview_model


@st.cache_resource
def get_appearance_matcher():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        matcher = AppearanceMatcher(use_pretrained=True)
    except ModuleNotFoundError:
        matcher = AppearanceMatcher(use_pretrained=False)
    return matcher


init_state()
check_lap()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Brand header ──
    logo_b64 = get_logo_b64()
    if logo_b64:
        st.markdown(
            f"""
            <div class="brand-header" id="sidebar-toggle" title="Click to collapse/expand sidebar">
                <img src="data:image/png;base64,{logo_b64}" alt="Bulls-Eye Logo">
                <div class="brand-title">BULLS-EYE</div>
                <div class="brand-subtitle">Click · Track · Focus</div>
                <div class="toggle-hint">click logo to collapse</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="brand-header" id="sidebar-toggle" title="Click to collapse/expand sidebar">
                <div class="logo-emoji" style="font-size:2.5rem; margin-bottom:0.4rem;">🎯</div>
                <div class="brand-title">BULLS-EYE</div>
                <div class="brand-subtitle">Click · Track · Focus</div>
                <div class="toggle-hint">click logo to collapse</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Collapsible content wrapper (everything below the logo) ──
    st.markdown('<div class="sidebar-collapsible">', unsafe_allow_html=True)

    # ── Source Mode Section ──
    st.markdown('<div class="section-label"><span class="sec-icon">📡</span> SOURCE</div>', unsafe_allow_html=True)
    source_mode = st.radio(
        "Input source",
        ["📁 Upload Video", "📷 Live Camera"],
        horizontal=True,
        label_visibility="collapsed",
    )
    is_live_mode = source_mode == "📷 Live Camera"

    uploaded = None
    if not is_live_mode:
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")

    # ── Playback Section ──
    st.markdown('<div class="section-label"><span class="sec-icon">🎬</span> PLAYBACK</div>', unsafe_allow_html=True)

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            video_path = tmp.name

        cap, meta = open_video(video_path)
        cap.release()

        start_frame = st.slider(
            "Start frame",
            min_value=0,
            max_value=max(0, meta.frame_count - 1),
            value=st.session_state.current_frame,
            step=1,
            disabled=st.session_state.preview_started,
        )

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            preview_fps = st.slider("Preview FPS", min_value=1, max_value=20, value=5)
        with pcol2:
            playback_speed = st.slider("Speed (x)", min_value=0.25, max_value=3.0, value=1.0, step=0.25)

        # ── Control Buttons ──
        bcol1, bcol2, bcol3 = st.columns(3)
        start_clicked = bcol1.button("▶ Start", disabled=st.session_state.preview_started, use_container_width=True)
        play_pause = bcol2.button(
            "⏸ Pause" if st.session_state.playing else "▶ Play",
            disabled=not st.session_state.preview_started,
            use_container_width=True,
        )
        reset_clicked = bcol3.button("↺ Reset", disabled=not st.session_state.preview_started, use_container_width=True)
    elif is_live_mode:
        # Live camera controls
        lcol1, lcol2 = st.columns(2)
        with lcol1:
            start_cam = st.button(
                "📷 Start Camera",
                disabled=st.session_state.get("live_playing", False),
                use_container_width=True,
                type="primary",
            )
        with lcol2:
            stop_cam = st.button(
                "⏹ Stop Camera",
                disabled=not st.session_state.get("live_playing", False),
                use_container_width=True,
            )
        if start_cam:
            st.session_state.live_playing = True
            st.session_state.live_model = None  # Reset model for fresh tracker
            st.session_state.live_selected_track_id = None
            st.session_state.live_target_bbox = None
            st.session_state.target_embedding = None
            st.session_state.live_last_click = None
            st.session_state.live_pending_click = None
            st.rerun()
        if stop_cam:
            st.session_state.live_playing = False
            st.session_state.live_model = None
            st.rerun()

        # Placeholder values for video controls (unused in live mode)
        start_frame = 0
        preview_fps = 5
        playback_speed = 1.0
        start_clicked = False
        play_pause = False
        reset_clicked = False
        video_path = None
        meta = None
    else:
        start_frame = 0
        preview_fps = 5
        playback_speed = 1.0
        start_clicked = False
        play_pause = False
        reset_clicked = False
        video_path = None
        meta = None

    # ── Tracking Section ──
    st.markdown('<div class="section-label"><span class="sec-icon">🎯</span> TRACKING</div>', unsafe_allow_html=True)
    show_boxes = st.checkbox("Show detections", value=True)
    lock_target_widget = st.checkbox(
        "Lock target (ignore clicks)",
        value=st.session_state.lock_target,
        key="lock_target_widget",
    )
    st.session_state.lock_target = lock_target_widget

    appearance_match = st.checkbox("Appearance matching", value=True)
    if appearance_match:
        appearance_strictness = st.slider(
            "Match strictness",
            min_value=0.35,
            max_value=0.75,
            value=0.55,
            step=0.05,
        )
    else:
        appearance_strictness = 0.55

    # ── Enhancement Section ──
    st.markdown('<div class="section-label"><span class="sec-icon">⚡</span> ENHANCEMENT</div>', unsafe_allow_html=True)
    low_light = st.checkbox("Low-light enhance", value=False)
    adaptive_blur = st.checkbox("GrabCut mask", value=False, help="Sharper subject edges, slower processing.")
    fast_motion = st.checkbox("Fast motion mode", value=False)
    if fast_motion:
        fast_motion_tolerance = st.slider("Motion tolerance", min_value=1.0, max_value=3.0, value=2.0, step=0.25)
    else:
        fast_motion_tolerance = 2.0

    # ── Output Section ──
    st.markdown('<div class="section-label"><span class="sec-icon">💾</span> OUTPUT</div>', unsafe_allow_html=True)
    save_output = st.checkbox("Save output video", value=True)

    # ── Close collapsible wrapper ──
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Main Area ───────────────────────────────────────────────────────────────

if is_live_mode:
    # ═══════════════════════════════════════════════════════════════════════
    # LIVE CAMERA MODE
    # ═══════════════════════════════════════════════════════════════════════

    # Initialize live-specific session state
    if "live_playing" not in st.session_state:
        st.session_state.live_playing = False
    if "live_model" not in st.session_state:
        st.session_state.live_model = None
    if "live_last_click" not in st.session_state:
        st.session_state.live_last_click = None
    if "live_pending_click" not in st.session_state:
        st.session_state.live_pending_click = None

    # Appearance matcher for live mode
    matcher = get_appearance_matcher() if appearance_match else None
    keep_threshold = max(0.2, appearance_strictness - 0.1)
    switch_threshold = appearance_strictness

    matcher_mode = "—"
    if appearance_match and matcher is not None:
        matcher_mode = "torch" if matcher.mode == "torch" else "histogram"

    # ── Live model (persisted in session state) ──
    def ensure_live_model(reset=False):
        if reset or st.session_state.live_model is None:
            st.session_state.live_model = load_model("yolov8n.pt")
        return st.session_state.live_model

    # ── Start / Stop controls ──
    if not st.session_state.live_playing:
        st.markdown('<div class="main-title">LIVE CAMERA</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                        padding: 4rem 2rem; text-align:center; border: 1px dashed var(--border-accent);
                        border-radius: var(--radius); background: rgba(0,212,255,0.01);">
                <div style="font-size:2.5rem; margin-bottom:0.8rem; opacity:0.25;">📷</div>
                <div style="font-size:0.88rem; font-weight:500; color:var(--text-secondary); margin-bottom:0.3rem;">
                    Camera ready
                </div>
                <div style="font-size:0.75rem; color:var(--text-muted);">
                    Press <strong>▶ Start Camera</strong> in the sidebar to begin live tracking
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Active live stream ──
    model_live = ensure_live_model()

    # Capture a single frame from the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam. Make sure your camera is connected and not in use by another app.")
        st.session_state.live_playing = False
        st.stop()

    ok, frame = cap.read()
    cap.release()

    if not ok:
        st.error("❌ Could not read frame from webcam.")
        st.session_state.live_playing = False
        st.stop()

    # ── Apply low-light enhancement ──
    tracking_frame = enhance_low_light(frame) if low_light else frame

    # ── Run YOLO tracking ──
    results = model_live.track(
        tracking_frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )
    result = results[0]

    # ── Track selected target ──
    selected_track_id = st.session_state.get("live_selected_track_id")
    bbox = None

    if selected_track_id is not None:
        bbox = find_bbox_for_track(result, selected_track_id)

        # Fast-motion fallback: proximity search
        if bbox is None and fast_motion and st.session_state.get("live_target_bbox") is not None:
            last_bbox = st.session_state.live_target_bbox
            bbox_w = max(1, last_bbox[2] - last_bbox[0])
            bbox_h = max(1, last_bbox[3] - last_bbox[1])
            max_dist = max(bbox_w, bbox_h) * fast_motion_tolerance
            bbox, new_id = find_bbox_and_id_by_proximity(result, last_bbox, max_dist)
            if new_id is not None:
                st.session_state.live_selected_track_id = new_id
                selected_track_id = new_id

    # ── Appearance matching ──
    target_embedding = st.session_state.get("target_embedding")
    if appearance_match and matcher is not None and target_embedding is not None:
        if bbox is not None:
            current_emb = matcher.embed_crop(tracking_frame, bbox)
            sim = matcher.cosine_similarity(current_emb, target_embedding) if current_emb is not None else -1.0
            if sim < keep_threshold:
                candidates = get_candidate_boxes(result, max_candidates=5)
                best_bbox, best_id, best_sim = matcher.best_match(tracking_frame, candidates, target_embedding)
                if best_bbox is not None and best_sim >= switch_threshold:
                    bbox = best_bbox
                    st.session_state.live_selected_track_id = best_id
                    selected_track_id = best_id
        else:
            candidates = get_candidate_boxes(result, max_candidates=5)
            best_bbox, best_id, best_sim = matcher.best_match(tracking_frame, candidates, target_embedding)
            if best_bbox is not None and best_sim >= switch_threshold:
                bbox = best_bbox
                st.session_state.live_selected_track_id = best_id
                selected_track_id = best_id

    # ── Apply focus effect ──
    preview_frame = tracking_frame
    if selected_track_id is not None and bbox is not None:
        preview_frame = apply_focus_effect(tracking_frame, bbox, use_grabcut=adaptive_blur)
        st.session_state.live_target_bbox = bbox
    else:
        st.session_state.live_target_bbox = None

    # ── Draw detection boxes ──
    if show_boxes:
        preview_frame = draw_boxes(preview_frame, result)

    rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # ── Status bar ──
    st.markdown('<div class="main-title">LIVE CAMERA</div>', unsafe_allow_html=True)

    if selected_track_id is not None:
        track_status_html = f'<span class="status-pill tracking"><span class="pulse-dot green"></span> Tracking ID {selected_track_id}</span>'
    else:
        track_status_html = '<span class="status-pill idle"><span class="pulse-dot amber"></span> No target</span>'

    matcher_html = ""
    if appearance_match:
        matcher_html = f'<span class="status-pill matcher">Matcher: {matcher_mode}</span>'

    st.markdown(
        f"""
        <div class="status-container">
            {track_status_html}
            <span class="status-pill frame">📷 Live</span>
            {matcher_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Video Frame (clickable!) ──
    st.markdown('<div class="preview-wrapper">', unsafe_allow_html=True)
    coords = streamlit_image_coordinates(pil_img, key="live-click")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="click-hint">Click on a detected subject to track it · Click elsewhere to switch target</div>', unsafe_allow_html=True)

    # ── Click handling (same as video mode) ──
    if coords and st.session_state.lock_target:
        st.info("🔒 Target is locked. Disable **Lock target** in the sidebar to switch focus.")

    if coords and st.session_state.live_pending_click is None and not st.session_state.lock_target:
        click_x = int(coords["x"])
        click_y = int(coords["y"])
        click_key = (click_x, click_y)
        if st.session_state.live_last_click != click_key:
            st.session_state.live_last_click = click_key
            st.session_state.live_pending_click = {"x": click_x, "y": click_y}
            st.rerun()

    if st.session_state.live_pending_click is not None:
        click = st.session_state.live_pending_click
        selection = choose_target_from_click(result, click["x"], click["y"])
        if selection is None:
            st.warning("No detection under the click. Please click directly on the object.")
        else:
            st.session_state.live_selected_track_id = selection.track_id
            st.session_state.live_target_bbox = selection.bbox
            # Compute appearance embedding for matching
            if appearance_match and matcher is not None:
                embedding = matcher.embed_crop(tracking_frame, selection.bbox)
                if embedding is not None:
                    st.session_state.target_embedding = embedding
                else:
                    st.warning("Could not compute appearance embedding for this selection.")
            else:
                st.session_state.target_embedding = None
            st.success(f"🎯 Now tracking ID **{selection.track_id}**")
        st.session_state.live_pending_click = None

    # ── Continuous rerun for live feed ──
    if st.session_state.live_playing:
        time.sleep(1.0 / 10)  # ~10 FPS refresh
        st.rerun()

    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# VIDEO UPLOAD MODE (existing logic)
# ═══════════════════════════════════════════════════════════════════════════
if uploaded is None:
    # Empty state
    st.markdown(
        """
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                    padding: 6rem 2rem; text-align:center;">
            <div style="font-size:3.5rem; margin-bottom:1rem; opacity:0.3;">🎯</div>
            <div style="font-size:1.1rem; font-weight:600; color:var(--text-secondary); margin-bottom:0.5rem;">
                No video loaded
            </div>
            <div style="font-size:0.82rem; color:var(--text-muted); max-width:400px;">
                Upload a video in the sidebar to begin. Click any detected subject to keep it sharp while blurring everything else.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Handle button actions ────────────────────────────────────────────────────
if start_clicked:
    st.session_state.preview_started = True
    st.session_state.playing = True
    st.session_state.current_frame = start_frame
    st.session_state.selected_track_id = None
    st.session_state.selected_point = None
    st.session_state.selection_frame = None
    st.session_state.last_frame_index = None
    st.session_state.last_display_frame = None
    st.session_state.pending_click = None
    st.session_state.pending_click_frame = None
    st.session_state.last_click = None
    st.session_state.last_bbox = None
    st.session_state.lock_target = False
    st.session_state.target_embedding = None
    ensure_preview_model(reset=True)
    st.rerun()

if play_pause:
    st.session_state.playing = not st.session_state.playing
    st.rerun()

if reset_clicked:
    st.session_state.preview_started = False
    st.session_state.playing = False
    st.session_state.current_frame = start_frame
    st.session_state.selected_track_id = None
    st.session_state.selected_point = None
    st.session_state.selection_frame = None
    st.session_state.last_frame_index = None
    st.session_state.last_display_frame = None
    st.session_state.pending_click = None
    st.session_state.pending_click_frame = None
    st.session_state.last_click = None
    st.session_state.last_bbox = None
    st.session_state.lock_target = False
    st.session_state.target_embedding = None
    if "preview_model" in st.session_state:
        del st.session_state.preview_model
    st.rerun()

if not st.session_state.preview_started:
    st.markdown('<div class="main-title">LIVE PREVIEW</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                    padding: 4rem 2rem; text-align:center; border: 1px dashed var(--border-accent);
                    border-radius: var(--radius); background: rgba(0,212,255,0.01);">
            <div style="font-size:2rem; margin-bottom:0.8rem; opacity:0.25;">▶</div>
            <div style="font-size:0.88rem; font-weight:500; color:var(--text-secondary); margin-bottom:0.3rem;">
                Ready to preview
            </div>
            <div style="font-size:0.75rem; color:var(--text-muted);">
                Press <strong>▶ Start</strong> in the sidebar to begin streaming frames
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ─── Active Preview ──────────────────────────────────────────────────────────
base_fps = meta.fps or 30.0
effective_fps = base_fps * playback_speed
frame_step = max(1, int(round(effective_fps / preview_fps)))
matcher = get_appearance_matcher() if appearance_match else None
keep_threshold = max(0.2, appearance_strictness - 0.1)
switch_threshold = appearance_strictness

# Matcher mode warning
matcher_mode = "—"
if appearance_match and matcher is not None:
    matcher_mode = "torch" if matcher.mode == "torch" else "histogram"
    if matcher.mode != "torch":
        st.warning("Appearance model unavailable. Falling back to color-histogram matching.")

current_frame = st.session_state.current_frame

if st.session_state.pending_click is not None:
    current_frame = st.session_state.pending_click_frame or current_frame

reset_tracker = False
if st.session_state.last_frame_index is not None:
    expected_next = st.session_state.last_frame_index + frame_step
    if current_frame != expected_next:
        reset_tracker = True

model_preview = ensure_preview_model(reset=reset_tracker)

cap, _ = open_video(video_path)
ok, frame = read_frame_at(cap, current_frame)
cap.release()

if not ok:
    st.session_state.playing = False
    st.warning("Reached the end of the video.")
    st.stop()

tracking_frame = enhance_low_light(frame) if low_light else frame

results_preview = model_preview.track(
    tracking_frame,
    persist=True,
    tracker="bytetrack.yaml",
    verbose=False,
)
result_preview = results_preview[0]

selected_track_id = st.session_state.selected_track_id

bbox = None
if selected_track_id is not None:
    bbox = find_bbox_for_track(result_preview, selected_track_id)
    if bbox is None and fast_motion and st.session_state.last_bbox is not None:
        last_bbox = st.session_state.last_bbox
        bbox_w = max(1, last_bbox[2] - last_bbox[0])
        bbox_h = max(1, last_bbox[3] - last_bbox[1])
        max_distance = max(bbox_w, bbox_h) * fast_motion_tolerance
        bbox, new_id = find_bbox_and_id_by_proximity(result_preview, last_bbox, max_distance)
        if new_id is not None:
            st.session_state.selected_track_id = new_id
            selected_track_id = new_id

target_embedding = st.session_state.target_embedding
if appearance_match and matcher is not None and target_embedding is not None:
    if bbox is not None:
        current_emb = matcher.embed_crop(tracking_frame, bbox)
        sim = matcher.cosine_similarity(current_emb, target_embedding) if current_emb is not None else -1.0
        if sim < keep_threshold:
            candidates = get_candidate_boxes(result_preview, max_candidates=5)
            best_bbox, best_id, best_sim = matcher.best_match(tracking_frame, candidates, target_embedding)
            if best_bbox is not None and best_sim >= switch_threshold:
                bbox = best_bbox
                st.session_state.selected_track_id = best_id
                selected_track_id = best_id
    else:
        candidates = get_candidate_boxes(result_preview, max_candidates=5)
        best_bbox, best_id, best_sim = matcher.best_match(tracking_frame, candidates, target_embedding)
        if best_bbox is not None and best_sim >= switch_threshold:
            bbox = best_bbox
            st.session_state.selected_track_id = best_id
            selected_track_id = best_id

preview_frame = tracking_frame
if selected_track_id is not None:
    preview_frame = apply_focus_effect(tracking_frame, bbox, use_grabcut=adaptive_blur)
    st.session_state.last_bbox = bbox
else:
    st.session_state.last_bbox = None

if show_boxes:
    preview_frame = draw_boxes(preview_frame, result_preview)

rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(rgb)

# ── Status Bar ──
st.markdown('<div class="main-title">LIVE PREVIEW</div>', unsafe_allow_html=True)

track_status_html = ""
if selected_track_id is not None:
    track_status_html = f'<span class="status-pill tracking"><span class="pulse-dot green"></span> Tracking ID {selected_track_id}</span>'
else:
    track_status_html = '<span class="status-pill idle"><span class="pulse-dot amber"></span> No target</span>'

frame_html = f'<span class="status-pill frame">Frame {current_frame} / {meta.frame_count - 1}</span>'
play_state = "Playing" if st.session_state.playing else "Paused"
play_icon = "▶" if st.session_state.playing else "⏸"

matcher_html = ""
if appearance_match:
    matcher_html = f'<span class="status-pill matcher">Matcher: {matcher_mode}</span>'

st.markdown(
    f"""
    <div class="status-container">
        {track_status_html}
        {frame_html}
        <span class="status-pill" style="background:rgba(139,148,158,0.05); border-color:var(--border); color:var(--text-muted);">{play_icon} {play_state}</span>
        {matcher_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Video Frame ──
st.markdown('<div class="preview-wrapper">', unsafe_allow_html=True)
coords = streamlit_image_coordinates(pil_img, key="preview-click")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="click-hint">Click on a detected subject to track it · Click elsewhere to switch target</div>', unsafe_allow_html=True)

# ── Click handling (logic unchanged) ──
if coords and st.session_state.lock_target:
    st.info("🔒 Target is locked. Disable **Lock target** in the sidebar to switch focus.")

if coords and st.session_state.pending_click is None and not st.session_state.lock_target:
    click_x = int(coords["x"])
    click_y = int(coords["y"])
    click_frame = st.session_state.last_display_frame or current_frame
    click_key = (click_x, click_y, click_frame)
    if st.session_state.last_click != click_key:
        st.session_state.last_click = click_key
        st.session_state.pending_click = {"x": click_x, "y": click_y}
        st.session_state.pending_click_frame = click_frame
        st.session_state.playing = False
        st.rerun()

if st.session_state.pending_click is not None:
    click = st.session_state.pending_click
    selection = choose_target_from_click(result_preview, click["x"], click["y"])
    if selection is None:
        st.warning("No detection under the click. Please click directly on the object.")
    else:
        st.session_state.selected_track_id = selection.track_id
        st.session_state.selected_point = (click["x"], click["y"])
        st.session_state.selection_frame = current_frame
        st.session_state.last_bbox = selection.bbox
        if appearance_match and matcher is not None:
            embedding = matcher.embed_crop(tracking_frame, selection.bbox)
            if embedding is None:
                st.warning("Could not compute appearance embedding for this selection.")
            else:
                st.session_state.target_embedding = embedding
        else:
            st.session_state.target_embedding = None
        st.success(f"🎯 Now tracking ID **{selection.track_id}**")
    st.session_state.pending_click = None
    st.session_state.pending_click_frame = None

st.session_state.last_frame_index = current_frame
st.session_state.last_display_frame = current_frame

if st.session_state.playing:
    if current_frame >= meta.frame_count - 1:
        st.session_state.playing = False
        st.info("Reached the end of the video.")
    else:
        next_frame = current_frame + frame_step
        st.session_state.current_frame = min(next_frame, meta.frame_count - 1)
        time.sleep(1.0 / max(1, preview_fps))
        st.rerun()

# ─── Process & Save ──────────────────────────────────────────────────────────
if save_output:
    st.markdown("---")
    process = st.button(
        "⚙ Process & Save Video",
        type="primary",
        disabled=st.session_state.selected_point is None,
        use_container_width=True,
    )
    if process:
        selection_frame = st.session_state.selection_frame
        if selection_frame is None:
            st.error("Select a target before processing.")
            st.stop()

        click_x, click_y = st.session_state.selected_point

        progress = st.progress(0.0, text="Processing…")

        model = load_model("yolov8n.pt")
        cap_process, _ = open_video(video_path)

        ok_first, frame_first = read_frame_at(cap_process, selection_frame)
        if not ok_first:
            cap_process.release()
            st.error("Could not read the selected frame for processing.")
            st.stop()

        tracking_first = enhance_low_light(frame_first) if low_light else frame_first
        first_results = model.track(
            tracking_first,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        first_result = first_results[0]

        selection = choose_target_from_click(first_result, click_x, click_y)
        if selection is None:
            cap_process.release()
            st.error("No detection under the click. Please click directly on the object.")
            st.stop()

        output_path = Path(tempfile.mkstemp(suffix=".mp4")[1])
        writer = make_video_writer(str(output_path), meta)

        processed = apply_focus_effect(tracking_first, selection.bbox, use_grabcut=adaptive_blur)
        writer.write(processed)

        processed_frames = 1
        total_frames = max(1, meta.frame_count - selection_frame)
        current_track_id = selection.track_id
        last_bbox = selection.bbox
        target_embedding = None
        if appearance_match and matcher is not None:
            target_embedding = matcher.embed_crop(tracking_first, selection.bbox)

        for _, frame in iter_frames(cap_process, start=selection_frame + 1):
            tracking_frame = enhance_low_light(frame) if low_light else frame
            results = model.track(
                tracking_frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            result = results[0]
            bbox = find_bbox_for_track(result, current_track_id)
            if bbox is None and fast_motion and last_bbox is not None:
                bbox_w = max(1, last_bbox[2] - last_bbox[0])
                bbox_h = max(1, last_bbox[3] - last_bbox[1])
                max_distance = max(bbox_w, bbox_h) * fast_motion_tolerance
                bbox, new_id = find_bbox_and_id_by_proximity(result, last_bbox, max_distance)
                if new_id is not None:
                    current_track_id = new_id
            if appearance_match and matcher is not None and target_embedding is not None:
                if bbox is not None:
                    current_emb = matcher.embed_crop(tracking_frame, bbox)
                    sim = matcher.cosine_similarity(current_emb, target_embedding) if current_emb is not None else -1.0
                    if sim < keep_threshold:
                        candidates = get_candidate_boxes(result, max_candidates=5)
                        best_bbox, best_id, best_sim = matcher.best_match(
                            tracking_frame, candidates, target_embedding
                        )
                        if best_bbox is not None and best_sim >= switch_threshold:
                            bbox = best_bbox
                            current_track_id = best_id
                else:
                    candidates = get_candidate_boxes(result, max_candidates=5)
                    best_bbox, best_id, best_sim = matcher.best_match(
                        tracking_frame, candidates, target_embedding
                    )
                    if best_bbox is not None and best_sim >= switch_threshold:
                        bbox = best_bbox
                        current_track_id = best_id
            if bbox is not None:
                last_bbox = bbox
            processed = apply_focus_effect(tracking_frame, bbox, use_grabcut=adaptive_blur)
            writer.write(processed)

            processed_frames += 1
            progress.progress(min(1.0, processed_frames / total_frames), text="Processing…")

        cap_process.release()
        writer.release()

        progress.progress(1.0, text="Done!")

        with open(output_path, "rb") as f:
            st.download_button(
                "⬇ Download Processed Video",
                data=f,
                file_name=f"processed_{Path(uploaded.name).stem}.mp4",
                mime="video/mp4",
            )

        st.success("✅ Processing complete.")
