"""
src/core/report_generator.py

Academic PDF Research Report Generator for The Butterfly Algorithm.
Produces university-grade single-simulation multi-layer technical reports using ReportLab
and thread-safe Matplotlib (Agg backend) charts grounded strictly in persisted JSON data.

Key guarantees:
  - Uses ONLY saved backend simulation data; never fabricates numbers or reruns LLM inference.
  - Never mutates the source simulation JSON record.
  - Explicitly labels FAILED / PARTIALLY COMPLETED simulations and N/A metrics.
  - Explicitly distinguishes model-generated synthetic estimates from empirical observations.
  - Persists PDFs canonically under `data/reports/<simulation_id>.pdf` with source hash caching.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np

from src.core.forecasting_engine import ensure_forecast_fields
from src.core.forecast_summarizer import compress_summary_for_pdf, extract_event_date
from src.core.schemas import CORE_RESEARCH_QUESTION

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

REPORTS_DIR = os.getenv("BUTTERFLY_REPORTS_DIR", "data/reports")

# Page geometry: Letter (612 x 792 pt), 0.55in L/R margins -> 532.8 pt content width
PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN_X = 0.55 * inch
MARGIN_TOP = 0.66 * inch
MARGIN_BOTTOM = 0.56 * inch
PAGE1_TOP_MARGIN = 0.40 * inch
PAGE1_BOTTOM_MARGIN = 0.46 * inch
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X

# Institutional Palette
NAVY = colors.HexColor("#0F172A")
SLATE_DARK = colors.HexColor("#1E293B")
SLATE_MID = colors.HexColor("#475569")
SLATE_LIGHT = colors.HexColor("#94A3B8")
BORDER_GRAY = colors.HexColor("#CBD5E1")
BG_SUBTLE = colors.HexColor("#F8FAFC")
BG_TINT = colors.HexColor("#F1F5F9")
CYAN_ACCENT = colors.HexColor("#0891B2")
CYAN_SOFT = colors.HexColor("#ECFEFF")
VIOLET_ACCENT = colors.HexColor("#6D28D9")
EMERALD = colors.HexColor("#047857")
EMERALD_BG = colors.HexColor("#ECFDF5")
AMBER = colors.HexColor("#B45309")
AMBER_BG = colors.HexColor("#FFFBEB")
ROSE = colors.HexColor("#BE123C")
ROSE_BG = colors.HexColor("#FFF1F2")


class ReportGenerationError(RuntimeError):
    """Raised when a report cannot be built due to missing/corrupted simulation data."""


def _esc(val: Any) -> str:
    """Safely escapes arbitrary LLM/user strings for ReportLab XML Paragraphs with clean punctuation."""
    if val is None:
        return ""
    s = str(val)
    # Strip ASCII control characters other than \n and \t
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", s)
    # Collapse accidental doubled words like "raw raw"
    s = re.sub(r"\b(?!that\b|had\b|very\b)([A-Za-z]{2,})\s+\1\b", r"\1", s, flags=re.IGNORECASE)
    # Fix missing sentence break before Causal mechanism: / Shift: labels in stored verdicts
    s = re.sub(r"([a-z0-9])\s+(Causal mechanism:)", r"\1. \2", s)
    # Normalize accidental broken punctuation artifacts (preserving '...')
    s = re.sub(r"(?<!\.)\.\.(?!\.)", ".", s)
    s = re.sub(r"\.\s*;", ";", s)
    s = re.sub(r",\s*\.", ".", s)
    return xml_escape(s).replace("\n", "<br/>")


def _fmt_pct(val: Optional[float], digits: int = 1) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val * 100:.{digits}f}%"


def _fmt_num(val: Optional[float], digits: int = 3, signed: bool = False) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(float(val))


def _join_clean(items: Any, sep: str = "; ") -> str:
    """Joins sentence/clause items without creating awkward '.;' artifacts."""
    cleaned: List[str] = []
    for raw in (items or []):
        s = str(raw or "").strip()
        if not s:
            continue
        if s.endswith(".") and not s.endswith("..."):
            s = s[:-1].strip()
        if s:
            cleaned.append(s)
    return sep.join(cleaned)


def _status_display(status: Optional[str]) -> Tuple[str, colors.Color, colors.Color]:
    s = (status or "completed").lower()
    if s == "failed":
        return ("FAILED", ROSE, ROSE_BG)
    if s == "partially_completed":
        return ("PARTIALLY COMPLETED", AMBER, AMBER_BG)
    if s in ("running", "queued"):
        return (s.upper(), CYAN_ACCENT, CYAN_SOFT)
    return ("COMPLETED", EMERALD, EMERALD_BG)


def _safe_slug(text: str, max_len: int = 40) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", (text or "Simulation").strip()).strip("_")
    if not clean:
        clean = "Simulation"
    return clean[:max_len].strip("_") or "Simulation"


def build_download_filename(sim_data: Dict[str, Any], sim_id: str) -> str:
    title = (
        sim_data.get("event_title")
        or (sim_data.get("config") or {}).get("event_title")
        or sim_data.get("signal_text")
        or "Simulation"
    )
    slug = _safe_slug(str(title), max_len=36)
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "", sim_id)
    return f"Butterfly_Research_Report_{slug}_{safe_id}.pdf"


def get_report_paths(sim_id: str, reports_dir: Optional[str] = None) -> Tuple[str, str]:
    base_dir = reports_dir or REPORTS_DIR
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "", sim_id)
    return (
        os.path.join(base_dir, f"{safe_id}.pdf"),
        os.path.join(base_dir, f"{safe_id}.meta.json"),
    )


def compute_source_signature(session_path: str) -> str:
    h = hashlib.sha256()
    with open(session_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_report_status(
    sim_id: str,
    session_path: Optional[str] = None,
    reports_dir: Optional[str] = None,
) -> Dict[str, Any]:
    pdf_path, meta_path = get_report_paths(sim_id, reports_dir=reports_dir)
    exists = os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 0
    meta: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
        except Exception:
            meta = {}

    up_to_date = exists
    if exists and session_path and os.path.isfile(session_path):
        try:
            cur_hash = compute_source_signature(session_path)
            if meta.get("source_sha256") and meta.get("source_sha256") != cur_hash:
                up_to_date = False
        except Exception:
            pass

    return {
        "simulation_id": sim_id,
        "report_exists": exists,
        "up_to_date": up_to_date,
        "pdf_path": pdf_path if exists else None,
        "file_size_bytes": os.path.getsize(pdf_path) if exists else 0,
        "generated_at": meta.get("generated_at"),
        "filename": meta.get("filename"),
    }


# ── Canvas with Consistent Academic Headers, Footers & Page X of Y ────────────

class _NumberedResearchCanvas(canvas.Canvas):
    def __init__(self, *args, header_title: str = "", sim_id: str = "", status_label: str = "", generated_ts: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []
        self._header_title = header_title
        self._sim_id = sim_id
        self._status_label = status_label
        self._generated_ts = generated_ts

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_chrome(num_pages)
            super().showPage()
        super().save()

    def _draw_chrome(self, total_pages: int):
        page_num = self._pageNumber
        self.saveState()

        # Footer on every page positioned safely below both Page 1 (0.46in) and Later (0.56in) frames
        footer_line_y = 0.38 * inch
        footer_text_y = 0.23 * inch
        self.setStrokeColor(BORDER_GRAY)
        self.setLineWidth(0.5)
        self.line(MARGIN_X, footer_line_y, PAGE_WIDTH - MARGIN_X, footer_line_y)

        self.setFont("Helvetica", 7.3)
        self.setFillColor(SLATE_MID)
        self.drawString(
            MARGIN_X,
            footer_text_y,
            f"The Butterfly Algorithm  |  ID: {self._sim_id}  |  Generated: {self._generated_ts}",
        )
        self.drawRightString(
            PAGE_WIDTH - MARGIN_X,
            footer_text_y,
            f"Page {page_num} of {total_pages}",
        )

        # Running header on pages 2..N positioned safely above Later frame (topMargin = 0.66in)
        if page_num > 1:
            rule_y = PAGE_HEIGHT - 0.50 * inch
            text_y = PAGE_HEIGHT - 0.42 * inch
            self.setStrokeColor(BORDER_GRAY)
            self.setLineWidth(0.5)
            self.line(MARGIN_X, rule_y, PAGE_WIDTH - MARGIN_X, rule_y)

            self.setFont("Helvetica-Bold", 7.6)
            self.setFillColor(NAVY)
            short_title = (self._header_title[:68] + "...") if len(self._header_title) > 70 else self._header_title
            self.drawString(MARGIN_X, text_y, short_title)

            self.setFont("Helvetica", 7.3)
            self.setFillColor(SLATE_MID)
            self.drawRightString(
                PAGE_WIDTH - MARGIN_X,
                text_y,
                f"Status: {self._status_label}",
            )

        self.restoreState()


# ── Slot Partition Helper (mirrors backend/frontend logic cleanly) ────────────

def _partition_slot_map(micros: List[Dict[str, Any]], cfg: Dict[str, Any], diag_models: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], str, str]:
    ma_name = cfg.get("model_a") or (diag_models.get("model_a") or {}).get("model") or "Model A"
    mb_name = cfg.get("model_b") or (diag_models.get("model_b") or {}).get("model") or "Model B"

    slot_a: Dict[str, Dict[str, Any]] = {}
    slot_b: Dict[str, Dict[str, Any]] = {}
    has_explicit = any(m.get("model_slot") in ("model_a", "model_b") for m in micros)

    if has_explicit:
        for m in micros:
            arch = str(m.get("archetype") or "")
            if m.get("model_slot") == "model_a" and arch not in slot_a:
                slot_a[arch] = m
            elif m.get("model_slot") == "model_b" and arch not in slot_b:
                slot_b[arch] = m
    else:
        for idx, m in enumerate(micros):
            arch = str(m.get("archetype") or "")
            if m.get("model_name") == ma_name and arch not in slot_a:
                slot_a[arch] = m
            elif m.get("model_name") == mb_name and arch not in slot_b:
                slot_b[arch] = m
            elif idx < 5 and arch not in slot_a:
                slot_a[arch] = m
            elif arch not in slot_b:
                slot_b[arch] = m

    return slot_a, slot_b, str(ma_name), str(mb_name)


def _is_valid_micro(m: Optional[Dict[str, Any]]) -> bool:
    if not m:
        return False
    st = m.get("execution_status")
    if st is not None:
        return st == "success"
    return bool(str(m.get("primary_narrative") or "").strip())


# ── Thread-Safe Matplotlib Chart Builders (OOP Figure API) ────────────────────

def _fig_to_rl_image(fig: Figure, width_pts: float) -> Image:
    canvas_agg = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canvas_agg.print_png(buf)
    buf.seek(0)
    w_in, h_in = fig.get_size_inches()
    aspect = h_in / w_in if w_in > 0 else 0.5
    return Image(buf, width=width_pts, height=width_pts * aspect)


def _chart_micro_stance(last_iter: Dict[str, Any], cfg: Dict[str, Any], diag_models: Dict[str, Any]) -> Image:
    micros = last_iter.get("micro_interpretations") or []
    slot_a, slot_b, ma_name, mb_name = _partition_slot_map(micros, cfg, diag_models)

    arch_order = [
        ("institutional_analyst", "Institutional\nAnalyst"),
        ("skeptic_debunker", "Skeptic\nDebunker"),
        ("pattern_seeker", "Pattern\nSeeker"),
        ("doomer", "Doomer"),
        ("amplifier_influencer", "Amplifier\nInfluencer"),
    ]
    labels = [lbl for _, lbl in arch_order]
    y = np.arange(len(arch_order))
    height = 0.34

    vals_a = [float(slot_a[k]["stance"]) if _is_valid_micro(slot_a.get(k)) else 0.0 for k, _ in arch_order]
    vals_b = [float(slot_b[k]["stance"]) if _is_valid_micro(slot_b.get(k)) else 0.0 for k, _ in arch_order]
    valid_a = [_is_valid_micro(slot_a.get(k)) for k, _ in arch_order]
    valid_b = [_is_valid_micro(slot_b.get(k)) for k, _ in arch_order]

    fig = Figure(figsize=(7.6, 3.35), dpi=200)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    bars_a = ax.barh(y + height / 2, vals_a, height, label=f"Model A ({ma_name})", color="#0891B2", alpha=0.88)
    bars_b = ax.barh(y - height / 2, vals_b, height, label=f"Model B ({mb_name})", color="#7C3AED", alpha=0.85)

    for idx, ok in enumerate(valid_a):
        if not ok:
            bars_a[idx].set_color("#E11D48")
            bars_a[idx].set_alpha(0.35)
            ax.text(0.03, y[idx] + height / 2, "FAILED", va="center", ha="left", fontsize=7, color="#BE123C", fontweight="bold")
        else:
            v = vals_a[idx]
            ax.text(v + (0.03 if v >= 0 else -0.03), y[idx] + height / 2, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color="#0E7490")

    for idx, ok in enumerate(valid_b):
        if not ok:
            bars_b[idx].set_color("#E11D48")
            bars_b[idx].set_alpha(0.35)
            ax.text(0.03, y[idx] - height / 2, "FAILED", va="center", ha="left", fontsize=7, color="#BE123C", fontweight="bold")
        else:
            v = vals_b[idx]
            ax.text(v + (0.03 if v >= 0 else -0.03), y[idx] - height / 2, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color="#5B21B6")

    ax.axvline(0.0, color="#64748B", linewidth=0.9, linestyle="-")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color="#1E293B")
    ax.set_xlim(-1.18, 1.18)
    ax.set_xlabel("Stance Score [-1.0 Skeptical/Critical  to  +1.0 Amplifying/Supportive]", fontsize=8, color="#334155")
    ax.set_title(f"Final Iteration ({last_iter.get('iteration', '?')}): Dual-Model Stance Comparison by Archetype", fontsize=9.5, fontweight="bold", color="#0F172A", pad=26)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=2, fontsize=7.5, frameon=True, facecolor="#FFFFFF")
    fig.tight_layout()
    return _fig_to_rl_image(fig, CONTENT_WIDTH)


def _chart_meso_network(meso: Dict[str, Any], iter_num: int) -> Image:
    nodes = meso.get("nodes") or []
    edges = meso.get("edges") or []
    node_map = {n["id"]: n for n in nodes}

    fig = Figure(figsize=(8.0, 3.9), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    ax_net = fig.add_axes([0.01, 0.11, 0.52, 0.79])
    ax_bar = fig.add_axes([0.66, 0.13, 0.32, 0.72])

    # Left panel: Weighted 8-node directed graph (adjusted vertical coordinates prevent bottom legend collision)
    pos = {
        "external_media_broadcast": (0.11, 0.54),
        "institutional_analyst": (0.39, 0.89),
        "skeptic_debunker": (0.41, 0.66),
        "pattern_seeker": (0.39, 0.42),
        "doomer": (0.41, 0.17),
        "amplifier_influencer": (0.64, 0.78),
        "mainstream_public": (0.84, 0.61),
        "engaged_subcultures": (0.84, 0.25),
    }
    short_labels = {
        "external_media_broadcast": "External\nMedia",
        "institutional_analyst": "Institutional\nAnalyst",
        "skeptic_debunker": "Skeptic\nDebunker",
        "pattern_seeker": "Pattern\nSeeker",
        "doomer": "Doomer",
        "amplifier_influencer": "Amplifier\nInfluencer",
        "mainstream_public": "Mainstream\nPublic",
        "engaged_subcultures": "Engaged\nSubcultures",
    }
    chan_colors = {
        "external_broadcast": "#0891B2",
        "institutional_trust": "#059669",
        "peer_contagion": "#7C3AED",
    }

    for e in edges:
        src, dst = e.get("source"), e.get("target")
        if src not in pos or dst not in pos:
            continue
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        col = chan_colors.get(e.get("channel_type", "peer_contagion"), "#64748B")
        w = float(e.get("weight", 0.5))
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle="arc3,rad=0.08",
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.7 + w * 1.4,
            color=col,
            alpha=0.58,
            shrinkA=16,
            shrinkB=16,
        )
        ax_net.add_patch(arrow)

    for nid, (px, py) in pos.items():
        n = node_map.get(nid, {})
        act = float(n.get("current_activation", 0.0) or 0.0)
        act_clamped = max(0.0, min(1.0, act))
        # Color ramp from cool slate to deep cyan/amber
        fill = matplotlib.colormaps["YlGnBu"](0.20 + 0.75 * act_clamped)
        circ = Circle((px, py), radius=0.064, facecolor=fill, edgecolor="#0F172A", linewidth=1.1, zorder=4)
        ax_net.add_patch(circ)
        txt_col = "#FFFFFF" if act_clamped >= 0.48 else "#0F172A"
        ax_net.text(px, py + 0.004, f"{act * 100:.0f}%", ha="center", va="center", fontsize=7.0, fontweight="bold", color=txt_col, zorder=5)
        ax_net.text(
            px,
            py - 0.078,
            short_labels.get(nid, nid),
            ha="center",
            va="top",
            fontsize=6.0,
            color="#1E293B",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="#FFFFFF", edgecolor="none", alpha=0.84),
        )

    ax_net.set_xlim(0.0, 1.0)
    ax_net.set_ylim(-0.02, 1.0)
    ax_net.axis("off")
    ax_net.set_title(f"8-Node Meso Influence Graph (Round {iter_num})", fontsize=9, fontweight="bold", color="#0F172A", pad=8)

    # Bottom figure caption strip outside network coordinates so Doomer label never overlaps
    fig.text(0.015, 0.025, "Edges: Cyan=External Broadcast | Green=Institutional Trust | Violet=Peer Contagion", fontsize=6.2, color="#475569")

    # Right panel: Normalized [0, 1] Activation vs Raw Inflow
    ordered = sorted(nodes, key=lambda item: float(item.get("current_activation", 0.0)))
    names = [short_labels.get(n["id"], n["id"]).replace("\n", " ") for n in ordered]
    acts = [float(n.get("current_activation", 0.0) or 0.0) for n in ordered]
    raws = [float(n.get("raw_influence", n.get("current_activation", 0.0)) or 0.0) for n in ordered]
    yy = np.arange(len(ordered))

    ax_bar.set_facecolor("#F8FAFC")
    ax_bar.barh(yy + 0.16, raws, height=0.30, color="#94A3B8", alpha=0.65, label="Raw Inflow")
    ax_bar.barh(yy - 0.16, acts, height=0.30, color="#0891B2", alpha=0.90, label="Normalized [0,1]")
    ax_bar.axvline(1.0, color="#E11D48", linestyle="--", linewidth=0.8, alpha=0.75)
    ax_bar.set_yticks(yy)
    ax_bar.set_yticklabels(names, fontsize=6.4, color="#1E293B")
    max_x = max([1.15] + [r * 1.14 for r in raws])
    ax_bar.set_xlim(0.0, max_x)
    ax_bar.set_xlabel("Score (Red line = 1.0 ceiling)", fontsize=7, color="#334155")
    ax_bar.set_title("Normalized Activation vs Raw Inflow", fontsize=8.5, fontweight="bold", color="#0F172A", pad=22)
    ax_bar.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_bar.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=2, fontsize=6.3, frameon=True, facecolor="#FFFFFF")

    return _fig_to_rl_image(fig, CONTENT_WIDTH)


def _chart_macro_trajectory(iterations: List[Dict[str, Any]], traj: Dict[str, Any]) -> Image:
    steps = [int(it.get("iteration", i + 1)) for i, it in enumerate(iterations)]
    cum = [float((it.get("macro_state") or {}).get("cumulative_adoption", 0.0)) * 100 for it in iterations]
    act = [float((it.get("macro_state") or {}).get("active_adoption", 0.0)) * 100 for it in iterations]
    attn = [float((it.get("macro_state") or {}).get("attention_indicator", 0.0)) * 100 for it in iterations]
    vel = [float((it.get("macro_state") or {}).get("adoption_velocity", 0.0)) * 100 for it in iterations]
    ratios = [float((it.get("macro_state") or {}).get("attention_vs_adoption_ratio", 0.0)) for it in iterations]

    fig = Figure(figsize=(7.6, 3.55), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    ax1 = fig.add_subplot(121)
    ax1.set_facecolor("#F8FAFC")
    ax1.plot(steps, cum, marker="o", linewidth=2.0, color="#0891B2", label="Cumul. Sim. Diffusion N(t) %")
    ax1.plot(steps, act, marker="s", linewidth=2.0, color="#059669", label="Active Sim. Adoption A(t) %")
    ax1.plot(steps, attn, marker="^", linewidth=1.8, linestyle="--", color="#D97706", label="Sim. Attention Index %")
    ax1.fill_between(steps, act, cum, color="#94A3B8", alpha=0.18, label="Simulated Churn Loss")
    ax1.set_xticks(steps)
    ax1.set_xlabel("Iteration Cycle", fontsize=8)
    ax1.set_ylabel("Simulated Share (%)", fontsize=8)
    ax1.set_ylim(0, max([15.0] + cum + attn) * 1.24)
    ax1.set_title("Simulated Diffusion vs Attention Proxy", fontsize=9, fontweight="bold", color="#0F172A")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(bbox_to_anchor=(0.5, -0.21), loc="upper center", ncol=2, fontsize=6.2, frameon=True, facecolor="#FFFFFF")

    ax2 = fig.add_subplot(122)
    ax2.set_facecolor("#F8FAFC")
    bars = ax2.bar(steps, vel, width=0.45, color="#6366F1", alpha=0.85, label="Sim. Velocity dN/dt (%)")
    for b, v in zip(bars, vel):
        ax2.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.15, f"{v:+.2f}%", ha="center", va="bottom", fontsize=7, color="#312E81")
    ax2_t = ax2.twinx()
    line_r = ax2_t.plot(steps, ratios, marker="D", color="#E11D48", linewidth=1.8, label="Sim. Attn/Adoption Ratio (x)")
    ax2_t.set_ylabel("Sim. Attention / Adoption Ratio (x)", fontsize=7.5, color="#BE123C")
    ax2.set_xticks(steps)
    ax2.set_xlabel("Iteration Cycle", fontsize=8)
    ax2.set_ylabel("Sim. Velocity dN/dt (%)", fontsize=8, color="#312E81")
    ax2.set_ylim(min([0.0] + vel) * 1.1, max([5.0] + vel) * 1.30)
    ax2_t.set_ylim(0, max([2.0] + ratios) * 1.28)
    ax2.set_title("Simulated Diffusion Velocity & Hype Gap", fontsize=9, fontweight="bold", color="#0F172A")
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    handles2 = [bars] + line_r
    labels2 = [h.get_label() for h in handles2]
    ax2.legend(handles2, labels2, bbox_to_anchor=(0.5, -0.21), loc="upper center", ncol=1, fontsize=6.2, frameon=True, facecolor="#FFFFFF")

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    return _fig_to_rl_image(fig, CONTENT_WIDTH)


def _chart_feedback_evolution(iterations: List[Dict[str, Any]]) -> Image:
    steps = [int(it.get("iteration", i + 1)) for i, it in enumerate(iterations)]
    supported, inferred, unsupported, failed = [], [], [], []
    pol, spread = [], []

    for it in iterations:
        fb = it.get("feedback_state") or {}
        audit = fb.get("claim_verification_audit") or {}
        supported.append(int(audit.get("supported", 0)))
        inferred.append(int(audit.get("inferred", 0)))
        unsupported.append(int(audit.get("unsupported", 0)))
        failed.append(int(audit.get("failed_interpretations", 0)))
        pol.append(float(fb.get("polarization_index", 0.0)))
        spread.append(float((it.get("meso_state") or {}).get("effective_spread_potential", 0.0)))

    fig = Figure(figsize=(7.6, 2.85), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    ax1 = fig.add_subplot(121)
    ax1.set_facecolor("#F8FAFC")
    s_arr = np.array(supported)
    i_arr = np.array(inferred)
    u_arr = np.array(unsupported)
    f_arr = np.array(failed)
    ax1.bar(steps, s_arr, width=0.45, color="#059669", label="Supported")
    ax1.bar(steps, i_arr, width=0.45, bottom=s_arr, color="#D97706", label="Inferred")
    ax1.bar(steps, u_arr, width=0.45, bottom=s_arr + i_arr, color="#E11D48", label="Unsupported")
    if f_arr.sum() > 0:
        ax1.bar(steps, f_arr, width=0.45, bottom=s_arr + i_arr + u_arr, color="#475569", label="Failed")
    ax1.set_xticks(steps)
    ax1.set_xlabel("Iteration Cycle", fontsize=8)
    ax1.set_ylabel("Interpretation Slots", fontsize=8)
    ax1.set_title("Claim Verification Audit by Round", fontsize=9, fontweight="bold", color="#0F172A")
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(loc="upper right", fontsize=6.5, frameon=True)

    ax2 = fig.add_subplot(122)
    ax2.set_facecolor("#F8FAFC")
    ax2.plot(steps, pol, marker="o", linewidth=2.0, color="#DC2626", label="Polarization Index [0..1]")
    ax2.plot(steps, spread, marker="s", linewidth=2.0, color="#0891B2", label="Effective Spread Potential [0..1]")
    ax2.set_xticks(steps)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("Iteration Cycle", fontsize=8)
    ax2.set_ylabel("Normalized Score [0.0, 1.0]", fontsize=8)
    ax2.set_title("Polarization & Network Spread Evolution", fontsize=9, fontweight="bold", color="#0F172A")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(loc="best", fontsize=6.5, frameon=True)

    fig.tight_layout()
    return _fig_to_rl_image(fig, CONTENT_WIDTH)


def _chart_butterfly_causal_diagram(pathways: List[Dict[str, Any]], event_title: str) -> Image:
    """Renders the Butterfly-Effect Multi-Order Causal Diagram highlighting weak/speculative links."""
    fig = Figure(figsize=(8.2, 4.75), dpi=210)
    fig.patch.set_facecolor("#FFFFFF")
    ax = fig.add_axes([0.01, 0.04, 0.98, 0.89])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    if not pathways:
        ax.text(
            0.5,
            0.52,
            "Causal Butterfly-Effect Diagram Unavailable\n(Insufficient Valid Model Interpretations to Trace Multi-Order Pathways)",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#BE123C",
        )
        return _fig_to_rl_image(fig, CONTENT_WIDTH)

    # Column header labels
    col_x = [0.09, 0.30, 0.51, 0.72, 0.91]
    col_titles = ["INITIAL EVENT", "1ST ORDER", "2ND ORDER", "3RD ORDER", "SYSTEMIC / FEEDBACK"]
    for cx, title in zip(col_x, col_titles):
        ax.text(cx, 0.968, title, ha="center", va="top", fontsize=6.8, fontweight="bold", color="#0F172A")

    # Root Event Node on far left
    root_short = textwrap.shorten(re.sub(r"\s+", " ", event_title or "").strip(), width=48, placeholder="...")
    root_label = textwrap.fill(root_short, width=16)
    root_box = FancyBboxPatch(
        (0.015, 0.39),
        0.15,
        0.18,
        boxstyle="round,pad=0.012",
        facecolor="#0F172A",
        edgecolor="#0891B2",
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(root_box)
    ax.text(0.09, 0.48, root_label, ha="center", va="center", fontsize=6.2, fontweight="bold", color="#FFFFFF", zorder=5)

    row_ys = [0.77, 0.56, 0.35, 0.14]
    pathway_cols = ["#059669", "#0891B2", "#D97706", "#DC2626"]

    for p_idx, pw in enumerate(pathways[:4]):
        ry = row_ys[p_idx]
        p_col = pathway_cols[p_idx % len(pathway_cols)]
        steps = pw.get("steps") or []
        raw_pw_name = str(pw.get("name") or f"Pathway {p_idx + 1}")
        wrapped_pw = textwrap.shorten(f"P{p_idx + 1}: {raw_pw_name}", width=92, placeholder="...")
        ax.text(0.19, ry + 0.066, wrapped_pw, ha="left", va="bottom", fontsize=5.8, fontweight="bold", color=p_col)

        # Arrow from root to step 1
        arr0 = FancyArrowPatch(
            (0.17, 0.48),
            (0.215, ry),
            connectionstyle="arc3,rad=0.04",
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.1,
            color=p_col,
            alpha=0.65,
        )
        ax.add_patch(arr0)

        prev_x = None
        for s_idx, step in enumerate(steps[:4]):
            cx = col_x[s_idx + 1]
            is_weak = bool(step.get("is_weak_link"))
            is_missing = bool(step.get("is_missing_link"))
            ev_stat = str(step.get("evidence_status") or "inferred")
            box_bg = "#FDF4FF" if is_missing else "#FFF1F2" if ev_stat == "unsupported" else "#FFFBEB" if is_weak else "#F0FDFA"
            edge_col = "#7E22CE" if is_missing else "#BE123C" if ev_stat == "unsupported" else "#D97706" if is_weak else p_col
            line_style = ":" if is_missing else "--" if is_weak else "-"

            bw, bh = 0.165, 0.112
            box = FancyBboxPatch(
                (cx - bw / 2, ry - bh / 2),
                bw,
                bh,
                boxstyle="round,pad=0.008",
                facecolor=box_bg,
                edgecolor=edge_col,
                linestyle=line_style,
                linewidth=1.2,
                zorder=4,
            )
            ax.add_patch(box)

            raw_t = str(step.get("title") or step.get("effect") or "")
            clipped_t = textwrap.shorten(raw_t, width=74, placeholder="...")
            short_t = textwrap.fill(clipped_t, width=24)
            conf_val = float(step.get("confidence", 0.5) or 0.5)
            wk_tag = " [MISSING LINK]" if is_missing else (" [WEAK]" if is_weak else "")
            ax.text(
                cx,
                ry + 0.015,
                short_t,
                ha="center",
                va="center",
                fontsize=4.7,
                color="#0F172A",
                linespacing=1.05,
                zorder=5,
            )
            ax.text(
                cx,
                ry - 0.039,
                f"Conf {conf_val:.0%}{wk_tag}",
                ha="center",
                va="center",
                fontsize=4.8,
                fontweight="bold",
                color=edge_col,
                zorder=5,
            )

            if prev_x is not None:
                arr = FancyArrowPatch(
                    (prev_x + bw / 2 + 0.003, ry),
                    (cx - bw / 2 - 0.003, ry),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=1.1,
                    linestyle=line_style,
                    color=edge_col,
                    alpha=0.80,
                )
                ax.add_patch(arr)
            prev_x = cx

    ax.text(
        0.02,
        0.015,
        "Solid = Grounded Link  |  Dashed Amber = Multi-Assumption Weak Link  |  Dashed Crimson = Unsupported Jump  |  Dotted Purple = Missing Causal Link",
        fontsize=5.8,
        color="#475569",
    )
    ax.set_title("Multi-Order Causal Butterfly-Effect Pathway Diagram (Solid vs Flagged Weak Links)", fontsize=9.2, fontweight="bold", color="#0F172A", pad=4)
    return _fig_to_rl_image(fig, CONTENT_WIDTH)


# ── Main Public Report Generation Entry Point ─────────────────────────────────

def generate_simulation_pdf_report(
    sim_data: Dict[str, Any],
    session_path: Optional[str] = None,
    reports_dir: Optional[str] = None,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """
    Builds (or returns cached) institutional research PDF for one persisted simulation.
    Never modifies the underlying simulation file and never calls any LLM inference.
    """
    if not isinstance(sim_data, dict):
        raise ReportGenerationError("Simulation payload is malformed.")

    sim_id = str(sim_data.get("simulation_id") or "").strip()
    if not sim_id:
        raise ReportGenerationError("Missing simulation_id in persisted simulation record.")

    iterations = sim_data.get("iterations")
    if not isinstance(iterations, list) or len(iterations) == 0:
        raise ReportGenerationError(
            f"Cannot generate 4-layer research report for simulation '{sim_id}': 0 execution iterations recorded."
        )

    out_dir = reports_dir or REPORTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    pdf_path, meta_path = get_report_paths(sim_id, reports_dir=out_dir)

    source_sha = compute_source_signature(session_path) if (session_path and os.path.isfile(session_path)) else None
    download_filename = build_download_filename(sim_data, sim_id)

    # Check cache: reuse only if PDF exists AND source SHA matches AND force_regenerate is False
    if not force_regenerate and os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 0:
        existing_meta: Dict[str, Any] = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    existing_meta = json.load(mf)
            except Exception:
                existing_meta = {}
        if source_sha is None or existing_meta.get("source_sha256") == source_sha:
            return {
                "simulation_id": sim_id,
                "event_title": sim_data.get("event_title") or (sim_data.get("config") or {}).get("event_title") or sim_id,
                "report_exists": True,
                "cached": True,
                "pdf_path": pdf_path,
                "filename": existing_meta.get("filename") or download_filename,
                "file_size_bytes": os.path.getsize(pdf_path),
                "generated_at": existing_meta.get("generated_at"),
                "download_url": f"/api/simulations/{sim_id}/report/download",
            }

    # Gather structured fields cleanly (computes forecast bundle in-memory if older JSON lacks it)
    sim_view = ensure_forecast_fields(sim_data)
    iterations = sim_view.get("iterations") or iterations
    cfg = sim_view.get("config") or {}
    diag = sim_view.get("diagnostics") or {}
    diag_models = diag.get("selected_models") or {}
    signal_text = str(sim_view.get("signal_text") or cfg.get("signal") or "")
    event_title = str(sim_view.get("event_title") or cfg.get("event_title") or signal_text[:64] or sim_id)
    research_question = str(sim_view.get("research_question") or CORE_RESEARCH_QUESTION)
    source_url = cfg.get("source_url") or sim_view.get("source_url")
    created_ts = str(sim_view.get("timestamp") or "Unknown")
    generated_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    generated_iso = datetime.utcnow().isoformat() + "Z"

    sim_status = str(sim_view.get("status") or "completed")
    status_label, status_fg, status_bg = _status_display(sim_status)
    succ_cnt = sim_view.get("successful_interpretations_count")
    exp_cnt = sim_view.get("expected_interpretations_count")
    if succ_cnt is None or exp_cnt is None:
        all_m = [m for it in iterations for m in (it.get("micro_interpretations") or [])]
        exp_cnt = len(all_m)
        succ_cnt = sum(1 for m in all_m if _is_valid_micro(m))

    model_a = str(cfg.get("model_a") or (diag_models.get("model_a") or {}).get("model") or "Unknown")
    prov_a = str(cfg.get("provider_a") or (diag_models.get("model_a") or {}).get("provider") or "Unknown")
    model_b = str(cfg.get("model_b") or (diag_models.get("model_b") or {}).get("model") or "Unknown")
    prov_b = str(cfg.get("provider_b") or (diag_models.get("model_b") or {}).get("provider") or "Unknown")

    first_iter = iterations[0]
    last_iter = iterations[-1]
    last_meso = last_iter.get("meso_state") or {}
    last_macro = last_iter.get("macro_state") or {}
    last_fb = last_iter.get("feedback_state") or {}
    traj = sim_view.get("trajectory") or {}
    agr_details = sim_view.get("model_agreement_details") or {}
    conv_details = sim_view.get("narrative_convergence_details") or {}
    agr_score = sim_view.get("model_agreement_score")
    conv_score = sim_view.get("narrative_convergence_score")
    failed_models = sim_view.get("failed_models") or []
    model_diagnostics = sim_view.get("model_diagnostics") or []
    gpu_info = diag.get("gpu") or {}
    is_mock = bool(diag.get("is_mock_only")) or (prov_a.lower() == "mock" and prov_b.lower() == "mock")

    causal_pathways: List[Dict[str, Any]] = sim_view.get("causal_pathways") or []
    order_effects: Dict[str, Any] = sim_view.get("order_effects_summary") or {}
    stakeholder_reactions: List[Dict[str, Any]] = sim_view.get("stakeholder_reactions") or []
    scenarios: List[Dict[str, Any]] = sim_view.get("scenarios") or []
    time_horizon_fc: Dict[str, Any] = sim_view.get("time_horizon_forecast") or {}
    epistemic_bd: Dict[str, Any] = sim_view.get("epistemic_breakdown") or {}
    final_verdict: Dict[str, Any] = sim_view.get("final_verdict") or {}

    # ── Styles ────────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()
    st_p1_kicker = ParagraphStyle("P1Kicker", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8.2, leading=10.0, textColor=CYAN_ACCENT, spaceAfter=1)
    st_p1_title = ParagraphStyle("P1Title", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=14.2, leading=17.0, textColor=NAVY, spaceAfter=3)
    st_p1_sub = ParagraphStyle("P1Sub", parent=base["Normal"], fontName="Helvetica", fontSize=7.8, leading=10.0, textColor=SLATE_MID, spaceAfter=4)
    st_p1_h = ParagraphStyle("P1H", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8.7, leading=10.8, textColor=NAVY, spaceBefore=4, spaceAfter=2, keepWithNext=True)
    st_p1_summary = ParagraphStyle("P1Summary", parent=base["Normal"], fontName="Helvetica", fontSize=8.1, leading=11.0, textColor=SLATE_DARK, alignment=TA_JUSTIFY)
    st_p1_cell = ParagraphStyle("P1Cell", parent=base["Normal"], fontName="Helvetica", fontSize=7.4, leading=9.4, textColor=SLATE_DARK)
    st_p1_cell_b = ParagraphStyle("P1CellB", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.4, leading=9.4, textColor=NAVY)
    st_p1_chain_hdr = ParagraphStyle("P1ChainHdr", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=6.8, leading=8.4, textColor=colors.white, alignment=TA_CENTER)
    st_p1_chain_txt = ParagraphStyle("P1ChainTxt", parent=base["Normal"], fontName="Helvetica", fontSize=6.9, leading=8.8, textColor=SLATE_DARK, alignment=TA_LEFT)
    st_p1_scen_num = ParagraphStyle("P1ScenNum", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=11.5, leading=13.5, textColor=NAVY, alignment=TA_CENTER)
    st_p1_scen_lbl = ParagraphStyle("P1ScenLbl", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.2, leading=9.0, textColor=SLATE_MID, alignment=TA_CENTER)
    st_p1_scen_note = ParagraphStyle("P1ScenNote", parent=base["Normal"], fontName="Helvetica", fontSize=6.8, leading=8.6, textColor=SLATE_DARK, alignment=TA_CENTER)
    st_p1_disc = ParagraphStyle("P1Disc", parent=base["Normal"], fontName="Helvetica", fontSize=7.1, leading=9.2, textColor=SLATE_MID, alignment=TA_JUSTIFY)

    st_h1 = ParagraphStyle("SecH1", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=NAVY, spaceBefore=14, spaceAfter=6, keepWithNext=True)
    st_h2 = ParagraphStyle("SecH2", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=SLATE_DARK, spaceBefore=9, spaceAfter=4, keepWithNext=True)
    st_body = ParagraphStyle("Body", parent=base["Normal"], fontName="Helvetica", fontSize=9.2, leading=13.5, textColor=SLATE_DARK, alignment=TA_JUSTIFY, spaceAfter=6)
    st_callout = ParagraphStyle("Callout", parent=base["Normal"], fontName="Helvetica", fontSize=8.6, leading=12.4, textColor=SLATE_DARK, spaceAfter=4)
    st_caption = ParagraphStyle("Caption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=7.8, leading=10.2, textColor=SLATE_MID, alignment=TA_CENTER, spaceBefore=2, spaceAfter=8)
    st_tbl_hdr = ParagraphStyle("TblHdr", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.8, leading=10, textColor=colors.white)
    st_tbl_cell = ParagraphStyle("TblCell", parent=base["Normal"], fontName="Helvetica", fontSize=7.8, leading=10.2, textColor=SLATE_DARK)
    st_tbl_mono = ParagraphStyle("TblMono", parent=base["Normal"], fontName="Courier", fontSize=7.4, leading=9.8, textColor=NAVY)
    st_formula = ParagraphStyle("Formula", parent=base["Normal"], fontName="Courier", fontSize=8.0, leading=11.5, textColor=NAVY, spaceAfter=4)

    def _p(txt: Any, style: ParagraphStyle = st_tbl_cell) -> Paragraph:
        return Paragraph(_esc(txt), style)

    def _p_bullets(items: Any, empty_fallback: str = "None recorded", max_items: Optional[int] = None) -> Paragraph:
        raw_list = [str(x).strip() for x in (items or []) if str(x or "").strip()]
        if max_items is not None:
            raw_list = raw_list[:max_items]
        if not raw_list:
            return Paragraph(f"• {_esc(empty_fallback)}", st_tbl_cell)
        html = "<br/>".join(f"• {_esc(item)}" for item in raw_list)
        return Paragraph(html, st_tbl_cell)

    def _ph(txt: Any) -> Paragraph:
        return Paragraph(_esc(txt), st_tbl_hdr)

    def _styled_table(rows: List[List[Any]], col_widths: List[float], repeat_header: bool = True, long_tbl: bool = False) -> Table:
        cls = LongTable if long_tbl else Table
        tbl = cls(rows, colWidths=col_widths, repeatRows=1 if repeat_header else 0)
        tbl.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_SUBTLE]),
                ("GRID", (0, 0), (-1, -1), 0.45, BORDER_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5.5),
            ])
        )
        return tbl

    def _callout_box(title: str, body_html: str, border_col: colors.Color = CYAN_ACCENT, bg_col: colors.Color = BG_TINT) -> Table:
        content = Paragraph(f"<b>{_esc(title)}</b><br/>{body_html}", st_callout)
        box = Table([[content]], colWidths=[CONTENT_WIDTH])
        box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), bg_col),
                ("BOX", (0, 0), (-1, -1), 0.75, border_col),
                ("LINEBEFORE", (0, 0), (0, -1), 3.5, border_col),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ])
        )
        return box

    def _clip_p1(txt: Any, max_chars: int) -> str:
        s = re.sub(r"\s+", " ", str(txt or "")).strip()
        if len(s) <= max_chars:
            return s
        out = textwrap.shorten(s, width=max_chars, placeholder="...")
        return re.sub(r"(?:\s+(?:of|for|to|in|on|with|and|or|&))+\.\.\.$", "...", out, flags=re.IGNORECASE)

    def _clip_p1_sentence(txt: Any, max_chars: int) -> str:
        s = re.sub(r"\s+", " ", str(txt or "")).strip()
        if len(s) <= max_chars:
            return s
        sents = re.split(r"(?<=[.!?])\s+", s)
        if len(sents) >= 2 and 34 <= len(sents[0]) <= max_chars:
            acc = sents[0]
            for nxt in sents[1:]:
                if len(acc) + 1 + len(nxt) <= max_chars:
                    acc += " " + nxt
                else:
                    break
            return acc
        return _clip_p1(s, max_chars)

    def _clean_p1_verdict_text(txt: Any) -> str:
        s = str(txt or "").strip()
        s = re.sub(r"^\[[^\]]+\]\s*", "", s)
        s = re.sub(r"^(?:Downside\s+harm\s+risk|Stabilizing\s+outcome)\s*\([^)]*\)\s*:\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*\(Note:\s*benign competition[^)]*\)", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*\(simulated churn[^)]*\)", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*\((?:Hypothesis|Tail Hypothesis(?:; unverified)?)\)", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s).strip()
        return (s[0].upper() + s[1:]) if s and s[0].islower() else s

    def _short_scen_subtitle(title: str) -> str:
        s = re.sub(r"^(?:Baseline|Positive|Negative|Extreme(?:\s+Tail-Risk)?)\s+Trajectory\s*[—–\-:]\s*", "", str(title or "").strip(), flags=re.IGNORECASE)
        return _clip_p1(s or title or "Unavailable", 58)

    def _fmt_weight_exact(w: Optional[float]) -> str:
        if w is None or (isinstance(w, float) and math.isnan(w)):
            return "N/A"
        return f"{float(w) * 100.0:.1f}%"

    def _hamilton_four_weights(sc_by_key: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        keys = ("baseline", "positive", "negative", "extreme")
        raw_vals: List[Tuple[str, float]] = []
        for k in keys:
            w = (sc_by_key.get(k) or {}).get("conditional_weight")
            if w is None or (isinstance(w, float) and math.isnan(w)):
                return {kk: "N/A" for kk in keys}
            raw_vals.append((k, float(w) * 1000.0))  # tenths of a percent, total = 1000
        floors = {k: int(math.floor(v)) for k, v in raw_vals}
        diff = 1000 - sum(floors.values())
        if -4 <= diff <= 4:
            ranked = sorted(raw_vals, key=lambda kv: kv[1] - floors[kv[0]], reverse=(diff > 0))
            for idx in range(abs(diff)):
                k = ranked[idx % len(ranked)][0]
                floors[k] += 1 if diff > 0 else -1
        return {k: f"{floors[k] / 10.0:.1f}%" for k in keys}

    # Gather Page 1 executive fields from authoritative FinalVerdict & Scenarios
    tr_cls = final_verdict.get("triggers_classified") or {}
    eg_cls = final_verdict.get("evidence_gaps_classified") or {}
    mm_verdict = final_verdict.get("monitoring_matrix") or (final_verdict.get("traceability_refs") or {}).get("monitoring_matrix") or []
    initial_event_date = final_verdict.get("initial_event_date") or extract_event_date(signal_text) or "Not specified"
    forecast_summary_txt = str(final_verdict.get("forecast_summary") or final_verdict.get("executive_summary") or "").strip()
    summary_meta = final_verdict.get("forecast_summary_meta") or {}
    compact_chain = final_verdict.get("causal_chain_compact") or {}

    scen_by_type_top = {str(sc.get("scenario_type") or "").lower(): sc for sc in scenarios}
    four_w_map = _hamilton_four_weights(scen_by_type_top)

    leading_scen = max(
        scenarios,
        key=lambda sc: float(sc.get("conditional_weight") or 0.0),
    ) if scenarios else {}
    lead_type_key = str(leading_scen.get("scenario_type") or "baseline").lower()
    lead_type_lbl = lead_type_key.capitalize()
    lead_w_str = four_w_map.get(lead_type_key) if lead_type_key in four_w_map else _fmt_weight_exact(leading_scen.get("conditional_weight"))
    lead_title = str(leading_scen.get("title") or "")
    lead_why = str(leading_scen.get("summary") or leading_scen.get("rationale") or final_verdict.get("executive_summary") or "")
    if lead_title and lead_title.lower() not in lead_why.lower():
        why_leads_txt = f"{_short_scen_subtitle(lead_title)}: {lead_why}"
    else:
        why_leads_txt = lead_why

    short_imp = _join_clean((leading_scen.get("short_term_consequences") or time_horizon_fc.get("short_term_weeks") or [])[:1]) or "Contained initial review"
    long_imp = _join_clean((leading_scen.get("long_term_consequences") or time_horizon_fc.get("long_term_quarters") or [])[:1]) or "Limited structural change unless evidence escalates"

    chain_nodes = [
        ("1. INITIAL EVENT", compact_chain.get("initial_event") or event_title),
        ("2. IMMEDIATE", compact_chain.get("immediate_reaction") or final_verdict.get("most_likely_immediate_outcome") or "Unavailable"),
        ("3. SECONDARY", compact_chain.get("secondary_consequence") or final_verdict.get("most_plausible_secondary_cascade") or "Unavailable"),
        ("4. THIRD-ORDER", compact_chain.get("third_order_effect") or _join_clean((order_effects.get("third_order_and_systemic") or [])[:1]) or "Contingent"),
        ("5. SYSTEMIC", f"{lead_type_lbl} regime ({lead_w_str})" if scenarios else (compact_chain.get("systemic_outcome") or "Unavailable")),
    ]

    verdict_conf = final_verdict.get("verdict_confidence")
    conf_str = _fmt_pct(verdict_conf) if verdict_conf is not None else "N/A"
    conf_band_str = str(final_verdict.get("confidence_band") or ("Unavailable" if sim_status == "failed" else "Heuristic"))
    top_turning_pt = (
        (tr_cls.get("escalation") or [])[:1]
        or (tr_cls.get("acceleration") or [])[:1]
        or (final_verdict.get("key_triggers_and_turning_points") or [])[:1]
        or ["Verification findings clarifying initial claims"]
    )[0]
    top_ev_gap = (
        summary_meta.get("uncertainty_disclosure")
        or (eg_cls.get("missing_source_evidence") or [])[:1]
        or (final_verdict.get("evidence_and_information_gaps") or [])[:1]
        or ["Unverified source claims require independent corroboration"]
    )
    if isinstance(top_ev_gap, list):
        top_ev_gap = top_ev_gap[0]

    # Exact mandated Methodological Disclaimer (Verbatim per Phase 6.4 specification)
    verbatim_disclaimer = (
        "This forecast is generated from multi-model structured behavioral interpretations, network propagation proxies, "
        "and simulated diffusion dynamics. Scenario weights represent heuristic exploratory indices rather than "
        "empirically calibrated statistical probabilities. Confirmed facts are separated from unverified claims and "
        "hypothetical future developments."
    )

    def _build_page1_flowables(summary_str: str, compact_mode: bool = False) -> List[Any]:
        p1: List[Any] = []
        # Header
        p1.append(Paragraph("THE BUTTERFLY ALGORITHM", st_p1_kicker))
        p1.append(Paragraph("Future Impact Forecast &amp; Final Verdict", st_p1_title))
        p1.append(Paragraph(f"<b>Subject:</b> {_esc(_clip_p1(event_title, 95))}", st_p1_sub))
        p1.append(HRFlowable(width="100%", thickness=1.4, color=NAVY, spaceAfter=4))

        # 1. Event Information
        p1.append(Paragraph("1. Event Information", st_p1_h))
        ev_info_rows = [
            [
                Paragraph(f"<b>Event:</b> {_esc(_clip_p1(event_title, 78))}", st_p1_cell),
                Paragraph(f"<b>Initial Event Date:</b> {_esc(initial_event_date)}", st_p1_cell),
            ],
            [
                Paragraph(f"<b>Simulation ID:</b> <font name='Courier' size='7.2'>{_esc(sim_id)}</font>", st_p1_cell),
                Paragraph(f"<b>Analysis Date (UTC):</b> {_esc(generated_ts)}", st_p1_cell),
            ],
            [
                Paragraph(f"<b>Status:</b> {_esc(status_label)} ({succ_cnt}/{exp_cnt} slots, {len(iterations)}/{cfg.get('max_iterations', len(iterations))} iter)", st_p1_cell),
                Paragraph(f"<b>Models:</b> {_esc(model_a)} ({_esc(prov_a)}) | {_esc(model_b)} ({_esc(prov_b)})", st_p1_cell),
            ],
        ]
        ev_tbl = Table(ev_info_rows, colWidths=[CONTENT_WIDTH * 0.54, CONTENT_WIDTH * 0.46])
        ev_tbl.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), BG_SUBTLE),
                ("BOX", (0, 0), (-1, -1), 0.55, BORDER_GRAY),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, BORDER_GRAY),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5 if compact_mode else 3.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 if compact_mode else 3.2),
                ("LEFTPADDING", (0, 0), (-1, -1), 5.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5.0),
            ])
        )
        p1.append(ev_tbl)

        # 2. Executive Forecast Summary
        p1.append(Paragraph("2. Executive Forecast Summary", st_p1_h))
        sum_box = Table([[Paragraph(_esc(summary_str), st_p1_summary)]], colWidths=[CONTENT_WIDTH])
        sum_box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CYAN_SOFT),
                ("BOX", (0, 0), (-1, -1), 0.75, CYAN_ACCENT),
                ("LINEBEFORE", (0, 0), (0, -1), 3.0, CYAN_ACCENT),
                ("TOPPADDING", (0, 0), (-1, -1), 4.0 if compact_mode else 5.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.0 if compact_mode else 5.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 7.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7.5),
            ])
        )
        p1.append(sum_box)

        # 3. Most Plausible Future Trajectory
        p1.append(Paragraph("3. Most Plausible Future Trajectory", st_p1_h))
        traj_max_why = 155 if compact_mode else 198
        traj_max_hl = 95 if compact_mode else 118
        traj_rows = [
            [
                Paragraph("<b>Leading Scenario &amp; Heuristic Weight</b>", st_p1_cell_b),
                Paragraph(
                    f"<b>{_esc(lead_type_lbl)} ({_esc(lead_w_str)} heuristic weight, non-empirical)</b> — {_esc(_clip_p1(why_leads_txt, traj_max_why))}",
                    st_p1_cell,
                ),
            ],
            [
                Paragraph("<b>Short- vs Longer-Term Implications</b>", st_p1_cell_b),
                Paragraph(
                    f"<b>Near-term:</b> {_esc(_clip_p1(short_imp, traj_max_hl))} &nbsp;|&nbsp; <b>Longer-term:</b> {_esc(_clip_p1(long_imp, traj_max_hl))}",
                    st_p1_cell,
                ),
            ],
        ]
        traj_tbl = Table(traj_rows, colWidths=[132.0, CONTENT_WIDTH - 132.0])
        traj_tbl.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [BG_SUBTLE, colors.white]),
                ("BOX", (0, 0), (-1, -1), 0.55, BORDER_GRAY),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, BORDER_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5 if compact_mode else 3.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 if compact_mode else 3.2),
                ("LEFTPADDING", (0, 0), (-1, -1), 5.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5.0),
            ])
        )
        p1.append(traj_tbl)

        # 4. Key Causal Chain
        p1.append(Paragraph("4. Key Causal Chain", st_p1_h))
        cw = CONTENT_WIDTH / 5.0
        chain_lim = 64 if compact_mode else 82
        chain_hdr_row = [Paragraph(hdr, st_p1_chain_hdr) for hdr, _ in chain_nodes]
        chain_body_row = [Paragraph(_esc(_clip_p1(body, chain_lim)), st_p1_chain_txt) for _, body in chain_nodes]
        chain_tbl = Table([chain_hdr_row, chain_body_row], colWidths=[cw] * 5)
        chain_tbl.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 1), (-1, 1), BG_SUBTLE),
                ("GRID", (0, 0), (-1, -1), 0.45, BORDER_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, 0), 2.6),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 2.6),
                ("TOPPADDING", (0, 1), (-1, 1), 3.0 if compact_mode else 3.6),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 3.0 if compact_mode else 3.6),
                ("LEFTPADDING", (0, 0), (-1, -1), 4.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4.0),
            ])
        )
        p1.append(chain_tbl)

        # 5. Four Scenario Weights
        p1.append(Paragraph("5. Four Scenario Weights (Heuristic Exploratory Indices — Not Statistical Probabilities)", st_p1_h))
        scen_cols: List[Any] = []
        ordered_types = [
            ("baseline", "BASELINE", BG_TINT, NAVY),
            ("positive", "POSITIVE", EMERALD_BG, EMERALD),
            ("negative", "NEGATIVE", AMBER_BG, AMBER),
            ("extreme", "EXTREME", ROSE_BG, ROSE),
        ]
        for st_key, st_disp, bg_c, fg_c in ordered_types:
            sc_obj = scen_by_type_top.get(st_key, {})
            w_disp = four_w_map.get(st_key, "N/A")
            sc_title = _short_scen_subtitle(sc_obj.get("title") or sc_obj.get("summary") or "Unavailable")
            cell_tbl = Table(
                [
                    [Paragraph(f"<font color='{fg_c.hexval()}'>{st_disp}</font>", st_p1_scen_lbl)],
                    [Paragraph(f"<font color='{fg_c.hexval()}'>{_esc(w_disp)}</font>", st_p1_scen_num)],
                    [Paragraph(_esc(sc_title), st_p1_scen_note)],
                ],
                colWidths=[(CONTENT_WIDTH / 4.0) - 4.0],
                rowHeights=[12.0, 16.5, 22.5],
            )
            cell_tbl.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), bg_c),
                    ("BOX", (0, 0), (-1, -1), 0.55, fg_c),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
                ])
            )
            scen_cols.append(cell_tbl)
        scen_strip = Table([scen_cols], colWidths=[CONTENT_WIDTH / 4.0] * 4)
        scen_strip.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
            ])
        )
        p1.append(scen_strip)

        # 6. Final Verdict & Uncertainty
        p1.append(Paragraph("6. Final Verdict &amp; Uncertainty", st_p1_h))
        fv_lim = 112 if compact_mode else 138
        downside_clean = _clean_p1_verdict_text(final_verdict.get("most_dangerous_downside_risk") or "Unavailable")
        stab_clean = _clean_p1_verdict_text(final_verdict.get("most_plausible_stabilizing_outcome") or "Unavailable")
        gap_clean = _clean_p1_verdict_text(top_ev_gap)
        fv_rows = [
            [
                Paragraph("<b>Confidence Score / Index</b>", st_p1_cell_b),
                Paragraph(
                    f"{_esc(conf_str)} — {_esc(conf_band_str)} &nbsp;|&nbsp; Uncertainty Index: {_fmt_num(final_verdict.get('uncertainty'), 2)} "
                    f"&nbsp;|&nbsp; Valid Slots: {succ_cnt}/{exp_cnt}",
                    st_p1_cell,
                ),
            ],
            [
                Paragraph("<b>Main Downside Risk</b>", st_p1_cell_b),
                Paragraph(_esc(_clip_p1_sentence(downside_clean, fv_lim)), st_p1_cell),
            ],
            [
                Paragraph("<b>Main Stabilizing Possibility</b>", st_p1_cell_b),
                Paragraph(_esc(_clip_p1_sentence(stab_clean, fv_lim)), st_p1_cell),
            ],
            [
                Paragraph("<b>Most Important Turning Point</b>", st_p1_cell_b),
                Paragraph(_esc(_clip_p1_sentence(top_turning_pt, fv_lim)), st_p1_cell),
            ],
            [
                Paragraph("<b>Biggest Evidence Gap</b>", st_p1_cell_b),
                Paragraph(_esc(_clip_p1_sentence(gap_clean, fv_lim)), st_p1_cell),
            ],
        ]
        fv_tbl = Table(fv_rows, colWidths=[128.0, CONTENT_WIDTH - 128.0])
        fv_tbl.setStyle(
            TableStyle([
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, BG_SUBTLE]),
                ("BOX", (0, 0), (-1, -1), 0.55, BORDER_GRAY),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, BORDER_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.3 if compact_mode else 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3 if compact_mode else 3.0),
                ("LEFTPADDING", (0, 0), (-1, -1), 5.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5.0),
            ])
        )
        p1.append(fv_tbl)

        # 7. Methodological Disclaimer
        p1.append(Paragraph("7. Methodological Disclaimer", st_p1_h))
        disc_tbl = Table([[Paragraph(_esc(verbatim_disclaimer), st_p1_disc)]], colWidths=[CONTENT_WIDTH])
        disc_tbl.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), BG_TINT),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
                ("TOPPADDING", (0, 0), (-1, -1), 3.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
                ("LEFTPADDING", (0, 0), (-1, -1), 6.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6.0),
            ])
        )
        p1.append(disc_tbl)
        return p1

    def _measure_page1_height(items: List[Any]) -> float:
        total_h = 0.0
        for item in items:
            _, h = item.wrap(CONTENT_WIDTH, 9999.0)
            total_h += h + float(getattr(item, "spaceBefore", 0.0) or 0.0) + float(getattr(item, "spaceAfter", 0.0) or 0.0)
        return total_h

    page1_avail_h = PAGE_HEIGHT - PAGE1_TOP_MARGIN - PAGE1_BOTTOM_MARGIN - 6.0
    p1_flowables = _build_page1_flowables(forecast_summary_txt, compact_mode=False)
    if _measure_page1_height(p1_flowables) > page1_avail_h:
        compressed_sum = compress_summary_for_pdf(forecast_summary_txt, max_words=150)
        p1_flowables = _build_page1_flowables(compressed_sum, compact_mode=True)
    if _measure_page1_height(p1_flowables) > page1_avail_h:
        compressed_sum = compress_summary_for_pdf(forecast_summary_txt, max_words=132)
        p1_flowables = _build_page1_flowables(compressed_sum, compact_mode=True)

    # Build Story: Page 1 Executive Forecast followed cleanly by Detailed Technical Report (Pages 2..N)
    story: List[Any] = []
    story.extend(p1_flowables)
    story.append(NextPageTemplate("Later"))
    story.append(PageBreak())

    # ── SECTION 1: DETAILED EXECUTIVE VERDICT & Q1-Q8 ASSESSMENT (PAGE 2) ─────
    story.append(Paragraph("Detailed Technical Report", st_p1_kicker))
    story.append(Paragraph("1. Executive Verdict", st_h1))
    agr_str = _fmt_pct(agr_score) if agr_score is not None else "N/A (fewer than 2 valid paired models)"
    conv_str = (
        f"{_fmt_pct(conv_score)} ({conv_details.get('direction', 'measured')})"
        if conv_score is not None
        else "N/A (insufficient valid outputs)"
    )
    story.append(
        _callout_box(
            final_verdict.get("headline") or "Executive Verdict Summary",
            f"<b>Core Research Question:</b> {_esc(research_question)}<br/>"
            f"<b>Initial Signal Anchor:</b> {_esc(signal_text)}<br/>"
            f"<b>Confidence / Uncertainty:</b> {conf_str} — {_esc(conf_band_str)} &nbsp;|&nbsp; "
            f"Uncertainty Index: {_fmt_num(final_verdict.get('uncertainty'), 2)} &nbsp;|&nbsp; "
            f"Source URL: {_esc(source_url or 'Signal-only analysis')}",
            border_col=ROSE if sim_status == "failed" else AMBER if sim_status == "partially_completed" else NAVY,
            bg_col=ROSE_BG if sim_status == "failed" else AMBER_BG if sim_status == "partially_completed" else BG_TINT,
        )
    )
    story.append(Spacer(1, 8))

    trig_summary_str = _join_clean(
        [
            f"{cat.capitalize()}: {tr_cls[cat][0]}"
            for cat in ("acceleration", "escalation", "stabilization", "reversal", "falsification")
            if tr_cls.get(cat)
        ]
    ) or _join_clean((final_verdict.get("key_triggers_and_turning_points") or ["Unavailable"])[:3])

    gap_summary_str = _join_clean(
        [
            f"{k.replace('_', ' ').title()}: {eg_cls[k][0]}"
            for k in ("missing_source_evidence", "model_disagreement", "unsupported_causal_links")
            if eg_cls.get(k)
        ]
    ) or _join_clean((final_verdict.get("evidence_and_information_gaps") or ["Unavailable"])[:3])

    top_mon_str = "; ".join(
        f"[{m.get('scenario_type', 'ALL').upper()}] {m.get('measurable_variable') or m.get('indicator', '')}"
        for m in mm_verdict[:3]
    ) or "; ".join((final_verdict.get("recommended_monitoring_actions") or ["Unavailable"])[:2])

    diff_score = conv_details.get("reasoning_differentiation_score")
    diff_str = f", Reasoning Diff={_fmt_pct(diff_score)}" if diff_score is not None else ""
    cons_note = " (Legitimate Consensus)" if conv_details.get("legitimate_consensus_detected") else (
        " (WARNING: Archetype Collapse)" if conv_details.get("archetype_collapse_warning") else ""
    )

    verdict_qa_rows = [
        [_ph("Verdict Question (Q1-Q8)"), _ph("Reasoned Assessment (Traceable to Simulation Layers)")],
        [_p("1. What happens first? (Immediate)"), _p(final_verdict.get("most_likely_immediate_outcome") or "Unavailable")],
        [_p("2. Secondary effects plausible?"), _p(final_verdict.get("most_plausible_secondary_cascade") or "Unavailable")],
        [_p("3. Biggest downside risk?"), _p(final_verdict.get("most_dangerous_downside_risk") or "Unavailable")],
        [_p("4. Stabilizing / positive outcome?"), _p(final_verdict.get("most_plausible_stabilizing_outcome") or "Unavailable")],
        [_p("5. Key triggers & turning points"), _p(trig_summary_str)],
        [_p("6. Evidence & reasoning gaps"), _p(gap_summary_str)],
        [_p("7. Priority monitoring variables"), _p(top_mon_str)],
        [_p("8. Simulated Metrics & Epistemic Note"), _p(
            f"Active Sim. Adoption A(t)={_fmt_pct(last_macro.get('active_adoption'))}, "
            f"Cumul. Sim. Diffusion N(t)={_fmt_pct(last_macro.get('cumulative_adoption'))}, "
            f"Sim. Media Attention={_fmt_pct(last_macro.get('attention_indicator'))} "
            f"(Ratio {_fmt_num(last_macro.get('attention_vs_adoption_ratio'), 2)}x), "
            f"Model Agreement={agr_str}, Convergence={conv_str}{diff_str}{cons_note}. "
            "Heuristic weights sum to 1.00 (Non-Empirical)."
        )],
    ]
    story.append(_styled_table(verdict_qa_rows, [155.0, CONTENT_WIDTH - 155.0], long_tbl=True))
    story.append(Spacer(1, 8))

    lim_notes = []
    if is_mock:
        lim_notes.append("Executed with Deterministic Mock providers for offline validation rather than live neural LLM inference.")
    if sim_status != "completed":
        lim_notes.append(f"Execution status was {status_label}: {succ_cnt}/{exp_cnt} slots succeeded ({sim_view.get('status_details') or 'see Section 15 diagnostics'}).")
    audit_last = last_fb.get("claim_verification_audit") or {}
    if int(audit_last.get("unsupported", 0)) > 0:
        lim_notes.append(f"Final cycle recorded {audit_last.get('unsupported', 0)} unsupported hypothesis slot(s) out of {audit_last.get('total_interpretations', 0)}.")
    lim_notes.append("Heuristic weights and Bass curves represent bounded simulation estimates rather than empirically calibrated statistical forecasts.")
    story.append(_callout_box("Executive Limitations & Scope Disclosure", "<br/>".join(f"• {_esc(x)}" for x in lim_notes), border_col=AMBER if sim_status != "completed" else SLATE_MID))

    # ── SECTION 2: INITIAL EVENT & EVIDENCE BASELINE ──────────────────────────
    story.append(Paragraph("2. Initial Event &amp; Evidence Baseline", st_h1))
    cfg_rows = [
        [_ph("Experiment Parameter"), _ph("Recorded Value"), _ph("Methodological Role")],
        [_p("Model A / Model B"), _p(f"{prov_a}/{model_a}  vs  {prov_b}/{model_b}"), _p("Dual sequential inference engines (8GB VRAM safe)")],
        [_p("Iterations / Population / Churn"), _p(f"{len(iterations)} rounds | M={_fmt_num(cfg.get('population_ceiling', 1.0), 2)} | c={_fmt_num(cfg.get('churn_rate', 0.08), 3)}"), _p("Bounded recursive horizon [3..5] with behavioral abandonment")],
        [_p("External Weight (alpha)"), _p(_fmt_num(cfg.get("external_weight", 0.4), 2)), _p("Spread = alpha*External + (1-alpha)*Internal")],
        [_p("Duration / GPU Hardware"), _p(f"{diag.get('inference_duration_sec', 'N/A')} s | {gpu_info.get('name') or ('Mock/CPU' if is_mock else 'Unreported')}"), _p(f"VRAM: {gpu_info.get('vram_total_mb', 'N/A')} MB")],
        [_p("Source URL"), _p(source_url or "None supplied"), _p("Injected verbatim into Micro interpretation prompts")],
    ]
    story.append(_styled_table(cfg_rows, [145.0, 175.0, CONTENT_WIDTH - 320.0]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2.1 Strict Epistemic Classification Breakdown (6 Tiers)", st_h2))
    ep_rows = [
        [_ph("Epistemic Category"), _ph("Status Boundary"), _ph("Extracted Items from Signal & Simulation")],
        [_p("1. Source-Confirmed Facts"), _p("Verified premise"), _p_bullets(epistemic_bd.get("source_confirmed_facts"), "None extracted")],
        [_p("2. Attributed Claims"), _p("Reported / Deduced"), _p_bullets(epistemic_bd.get("attributed_claims"), "None recorded", max_items=3)],
        [_p("3. Unverified Allegations"), _p("Speculative"), _p_bullets(epistemic_bd.get("unverified_allegations"), "None isolated", max_items=3)],
        [_p("4. Model Interpretations"), _p("Archetype framing"), _p_bullets(epistemic_bd.get("model_interpretations"), "Unavailable", max_items=2)],
        [_p("5. Simulated Behavioral Outcomes"), _p("Synthetic model"), _p_bullets(epistemic_bd.get("simulated_behavioral_outcomes"), "Unavailable", max_items=2)],
        [_p("6. Hypothetical Future Consequences"), _p("Forecast branch"), _p_bullets(epistemic_bd.get("hypothetical_future_consequences"), "Unavailable", max_items=3)],
    ]
    story.append(_styled_table(ep_rows, [118.0, 78.0, CONTENT_WIDTH - 196.0], long_tbl=True))

    # ── SECTION 3: FIVE BEHAVIOURAL INTERPRETATIONS (MICRO LAYER) ─────────────
    story.append(PageBreak())
    story.append(Paragraph("3. Five Behavioural Interpretations (Micro Layer)", st_h1))
    story.append(
        Paragraph(
            "In each round, five behavioral archetypes generate both an evidence-grounded stance assessment and explicit causal hypotheses "
            "(mechanism, expected stakeholder actions, behavioural changes, emergent consequences, and invalidation criteria) across Model A and Model B.",
            st_body,
        )
    )

    agr_rows = [
        [_ph("Agreement Metric"), _ph("Measured Value"), _ph("Interpretation")],
        [_p("Composite Model Agreement"), _p(_fmt_pct(agr_details.get("score")) if agr_details.get("score") is not None else "N/A"), _p(agr_details.get("reason") or agr_details.get("formula") or "Weighted stance + causal reasoning agreement")],
        [_p("Matched Valid Archetype Pairs"), _p(f"{agr_details.get('valid_pairs', 0)} / {agr_details.get('total_archetypes', 5)}"), _p("Requires both Model A and Model B to succeed")],
        [_p("Stance Agreement / Pearson r"), _p(f"{_fmt_pct(agr_details.get('stance_agreement')) if agr_details.get('stance_agreement') is not None else 'N/A'} | r={_fmt_num(agr_details.get('stance_correlation'), 3, True) if agr_details.get('stance_correlation') is not None else 'N/A'}"), _p("Cross-model directional consistency")],
        [_p("Tag & Claim Semantic Similarity"), _p(_fmt_pct(agr_details.get("tag_semantic_similarity")) if agr_details.get("tag_semantic_similarity") is not None else "N/A"), _p("Jaccard overlap of narrative tags & claims")],
        [
            _p("Causal Reasoning Agreement"),
            _p(
                f"Causal {_fmt_pct(agr_details.get('causal_semantic_similarity'))} | Combined {_fmt_pct(agr_details.get('reasoning_agreement'))}"
                if agr_details.get("causal_semantic_similarity") is not None
                else "N/A"
            ),
            _p(
                "WARNING: Identical stances mask divergent causal mechanisms"
                if agr_details.get("stance_without_reasoning_warning")
                else "Overlap across causal mechanisms, actions, consequences & triggers"
            ),
        ],
        [
            _p("Archetype Differentiation Audit"),
            _p(
                f"Diff {_fmt_pct(conv_details.get('reasoning_differentiation_score'))} | Stance Std {_fmt_num(conv_details.get('archetype_stance_std_final'), 3)}"
                if conv_details.get("reasoning_differentiation_score") is not None
                else f"Conv {_fmt_pct(conv_score)}"
            ),
            _p(
                "WARNING: Archetype collapse detected (low stance std AND high reasoning overlap)"
                if conv_details.get("archetype_collapse_warning")
                else "Legitimate directional consensus with distinct archetype reasoning"
                if conv_details.get("legitimate_consensus_detected")
                else "Each archetype targets a distinct analytical question & stakeholder lens"
            ),
        ],
    ]
    story.append(_styled_table(agr_rows, [155.0, 130.0, CONTENT_WIDTH - 285.0]))
    story.append(Spacer(1, 8))

    story.append(
        KeepTogether([
            _chart_micro_stance(last_iter, cfg, diag_models),
            Spacer(1, 3),
            Paragraph(
                f"Figure 1: Final Iteration (Round {last_iter.get('iteration', '?')}) dual-model stance alignment across five analytical archetypes.",
                st_caption,
            ),
        ])
    )

    story.append(Paragraph(f"3.1 Final Cycle (Round {last_iter.get('iteration')}) Behavioural &amp; Causal Hypotheses", st_h2))
    micro_hdr = [_ph("Archetype"), _ph("Slot / Model"), _ph("Status & Metrics"), _ph("Analytical Lens, Narrative & Claims"), _ph("Causal Mechanism, Balances & Invalidation")]
    micro_rows: List[List[Any]] = [micro_hdr]

    last_micros = last_iter.get("micro_interpretations") or []
    for m in last_micros:
        slot = str(m.get("model_slot") or "unknown").upper().replace("_", " ")
        valid = _is_valid_micro(m)
        arch_disp = str(m.get("archetype_name") or m.get("archetype") or "")
        if valid:
            sev_line = f" Sev:{_fmt_num(m.get('risk_severity'), 2)}" if m.get("risk_severity") is not None else ""
            st_metrics = (
                f"{m.get('evidence_status', 'supported').upper()}\n"
                f"S:{_fmt_num(m.get('stance'), 2, True)} V:{_fmt_num(m.get('virality'), 2)}\n"
                f"C:{_fmt_num(m.get('credibility'), 2)} U:{_fmt_num(m.get('uncertainty'), 2)}{sev_line}"
            )
            claims_list = m.get("claims") or []
            aq = m.get("analytical_question") or ""
            sf = m.get("stakeholder_focus") or ""
            lens_bits = []
            if aq:
                lens_bits.append(f"Q: {aq}")
            if sf:
                lens_bits.append(f"Focus: {sf}")
            narr_txt = (
                (("\n".join(lens_bits) + "\n") if lens_bits else "")
                + f"{m.get('primary_narrative', '')}"
                + (f"\nClaims: {'; '.join(str(c) for c in claims_list[:2])}" if claims_list else "")
            )
            mech = m.get("causal_mechanism") or "See narrative"
            conseq = "; ".join((m.get("emergent_consequences") or [])[:2])
            supp_c = m.get("strongest_supporting_consideration") or ""
            opp_c = m.get("strongest_opposing_consideration") or ""
            inv = "; ".join((m.get("invalidation_conditions") or [])[:1])
            causal_txt = (
                f"[{m.get('time_horizon', 'short_term_weeks')}] Mech: {mech}"
                + (f"\nEffects: {conseq}" if conseq else "")
                + (f"\nSupporting: {supp_c}" if supp_c else "")
                + (f"\nOpposing: {opp_c}" if opp_c else "")
                + (f"\nFalsified if: {inv}" if inv else "")
            )
        else:
            st_metrics = f"FAILED\n({m.get('error_category') or m.get('execution_status')})"
            narr_txt = f"[ERROR] {m.get('error_message') or 'No valid JSON returned; not substituted.'}"
            causal_txt = "N/A (slot failed)"

        micro_rows.append([
            _p(arch_disp),
            _p(f"{slot}\n{m.get('model_name', '')}"),
            _p(st_metrics),
            _p(narr_txt),
            _p(causal_txt),
        ])

    story.append(_styled_table(micro_rows, [68.0, 68.0, 68.0, 154.0, CONTENT_WIDTH - 358.0], long_tbl=True))

    if failed_models:
        story.append(Spacer(1, 6))
        fail_lines = [
            f"• {f.get('model_slot')} ({f.get('model_name')}): {f.get('failure_count')} failed slots — categories: {', '.join(f.get('error_categories') or [])}"
            for f in failed_models
        ]
        story.append(_callout_box("Failed / Incomplete Model Slot Notice", "<br/>".join(_esc(x) for x in fail_lines), border_col=ROSE, bg_col=ROSE_BG))

    # ── SECTION 4: STAKEHOLDER REACTION ANALYSIS ──────────────────────────────
    story.append(Paragraph("4. Stakeholder Reaction Analysis", st_h1))
    if stakeholder_reactions:
        sr_rows = [[_ph("Stakeholder Cohort"), _ph("Meso Activation"), _ph("Expected Reaction & Behavioural Shift"), _ph("Likely Near-Term Actions"), _ph("Direction")]]
        for sr in stakeholder_reactions:
            act_v = sr.get("activation_level")
            sr_rows.append([
                _p(sr.get("stakeholder") or ""),
                _p(_fmt_pct(act_v) if act_v is not None else "N/A"),
                _p(f"{sr.get('expected_reaction', '')}\nShift: {sr.get('behavioural_shift', '')}"),
                _p("; ".join(sr.get("likely_actions") or [])),
                _p(f"{str(sr.get('impact_direction', 'mixed')).upper()}\n({sr.get('evidence_basis', 'inferred')})"),
            ])
        story.append(_styled_table(sr_rows, [92.0, 56.0, 165.0, 140.0, CONTENT_WIDTH - 453.0], long_tbl=True))
    else:
        story.append(Paragraph("Stakeholder reaction analysis unavailable (0 valid model interpretations).", st_body))

    # ── SECTION 5: SOCIAL INFLUENCE NETWORK (MESO LAYER) ───────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Social Influence Network (Meso Layer)", st_h1))
    story.append(
        Paragraph(
            f"Aggregate network pressures in the final round: <b>External Influence = {_fmt_pct(last_meso.get('external_influence_score'))}</b>, "
            f"<b>Internal Contagion = {_fmt_pct(last_meso.get('internal_contagion_score'))}</b>, and "
            f"<b>Effective Spread Potential = {_fmt_pct(last_meso.get('effective_spread_potential'))}</b>. "
            f"Activation semantics: <code>{_esc(last_meso.get('activation_semantics', 'instantaneous_per_iteration'))}</code>.",
            st_body,
        )
    )

    story.append(
        _callout_box(
            "Mathematical Relationship: Node Activation vs Raw Inflow vs Aggregate Scores",
            "1. <b>Normalized Activation [0.0, 1.0]:</b> Each node computes instantaneous activation normalized by capacity and clamped strictly to [0, 1] so repeated cycles never accumulate unbounded values.<br/>"
            "2. <b>Raw Inbound Inflow:</b> Preserved independently as sum(w_ji * T_j) before capacity normalization.<br/>"
            "3. <b>Effective Spread Potential:</b> Convex blend alpha*External + (1 - alpha)*Internal preventing double-counting.",
        )
    )
    story.append(Spacer(1, 8))

    if last_meso.get("nodes"):
        story.append(
            KeepTogether([
                _chart_meso_network(last_meso, int(last_iter.get("iteration", len(iterations)))),
                Spacer(1, 3),
                Paragraph(
                    f"Figure 2: Weighted 8-node Meso influence network and normalized [0, 1] activation vs raw inflow (Round {last_iter.get('iteration', len(iterations))}).",
                    st_caption,
                ),
            ])
        )
        story.append(Spacer(1, 4))

    node_rows = [[_ph("Node Label"), _ph("Type"), _ph("Norm. Activation"), _ph("Raw Influence"), _ph("Ext / Int Contrib"), _ph("Calculation Basis")]]
    for n in (last_meso.get("nodes") or []):
        node_rows.append([
            _p(n.get("label", n.get("id"))),
            _p(str(n.get("node_type", "")).replace("_", " ")),
            _p(_fmt_pct(n.get("current_activation"))),
            _p(_fmt_num(n.get("raw_influence", n.get("current_activation")), 3)),
            _p(f"{_fmt_num(n.get('external_contribution'), 3)} / {_fmt_num(n.get('internal_contribution'), 3)}"),
            _p(n.get("calculation_basis") or "Instantaneous weighted propagation"),
        ])
    story.append(_styled_table(node_rows, [92.0, 68.0, 64.0, 58.0, 74.0, CONTENT_WIDTH - 356.0], long_tbl=True))

    story.append(Spacer(1, 6))
    edge_rows = [[_ph("Source Node"), _ph("Target Node"), _ph("Weight"), _ph("Channel Type")]]
    for e in (last_meso.get("edges") or []):
        edge_rows.append([
            _p(e.get("source")),
            _p(e.get("target")),
            _p(_fmt_num(e.get("weight"), 2)),
            _p(str(e.get("channel_type", "")).replace("_", " ")),
        ])
    story.append(Paragraph("5.1 Directed Influence Edge Weights", st_h2))
    story.append(_styled_table(edge_rows, [145.0, 145.0, 65.0, CONTENT_WIDTH - 355.0]))

    st_mappings = last_meso.get("event_stakeholder_mapping") or []
    if st_mappings:
        story.append(Spacer(1, 8))
        story.append(Paragraph("5.2 Event-Specific Stakeholder Channel Mapping", st_h2))
        map_rows = [[_ph("Stakeholder Cohort"), _ph("Mapped Channel"), _ph("Relevance"), _ph("Node Act."), _ph("Causal Role")]]
        for sm in st_mappings:
            rel_val = sm.get("relevance_weight")
            act_val = sm.get("meso_node_activation")
            map_rows.append([
                _p(sm.get("stakeholder") or ""),
                _p(str(sm.get("meso_node_id") or "").replace("_", " ")),
                _p(_fmt_pct(rel_val) if rel_val is not None else "N/A"),
                _p(_fmt_pct(act_val) if act_val is not None else "N/A"),
                _p(sm.get("causal_role") or "Event-grounded stakeholder channel"),
            ])
        story.append(_styled_table(map_rows, [118.0, 96.0, 50.0, 50.0, CONTENT_WIDTH - 314.0], long_tbl=True))

    # ── SECTION 6 & SECTION 7: FIRST-ORDER & SECOND-ORDER EFFECTS ─────────────
    story.append(PageBreak())
    story.append(Paragraph("6. First-Order Effects (Immediate Behavioural &amp; Operational Consequences)", st_h1))
    first_list = order_effects.get("first_order") or ["No verified 1st-order consequences recorded."]
    for idx_f, item_f in enumerate(first_list, start=1):
        story.append(Paragraph(f"{idx_f}. {_esc(item_f)}", st_body))

    story.append(Paragraph("7. Second-Order Effects (Network Spillover &amp; Adaptive Hedging)", st_h1))
    second_list = order_effects.get("second_order") or ["No 2nd-order consequences recorded."]
    for idx_s, item_s in enumerate(second_list, start=1):
        story.append(Paragraph(f"{idx_s}. {_esc(item_s)}", st_body))

    # ── SECTION 8: THIRD-ORDER & SYSTEMIC EFFECTS (MACRO + FEEDBACK) ──────────
    story.append(Paragraph("8. Third-Order &amp; Systemic Effects (Macro Layer, Feedback Layer &amp; Cross-Layer Synthesis)", st_h1))
    third_list = order_effects.get("third_order_and_systemic") or []
    for idx_t, item_t in enumerate(third_list, start=1):
        story.append(Paragraph(f"• {_esc(item_t)}", st_body))

    story.append(
        KeepTogether([
            Paragraph("8.1 Macro Layer: Modified Bass Diffusion &amp; Attention Separation", st_h2),
            Paragraph(
                "Meso propagation scores parameterize the discrete Modified Bass diffusion equation with behavioral churn <i>c</i>: "
                "<i>dN/dt = (p + q * (N/M)) * (M - N)</i> and <i>dA/dt = dN/dt - c * A(t)</i>, guaranteeing "
                "<i>0 <= A(t) <= N(t) <= M</i>.",
                st_body,
            ),
            _chart_macro_trajectory(iterations, traj),
            Spacer(1, 3),
            Paragraph(
                "Figure 3: Modified Bass simulated diffusion trajectory N(t), active adoption A(t), attention index proxy, and velocity.",
                st_caption,
            ),
        ])
    )
    story.append(Spacer(1, 4))

    macro_rows = [[_ph("Round"), _ph("p (Ext)"), _ph("q (Int)"), _ph("M / c"), _ph("Cumul. Sim. N(t)"), _ph("Active Sim. A(t)"), _ph("Sim. Vel. dN/dt"), _ph("Sim. Attention"), _ph("Attn/Adopt")]]
    for it in iterations:
        mc = it.get("macro_state") or {}
        macro_rows.append([
            _p(str(it.get("iteration"))),
            _p(_fmt_num(mc.get("p_external"), 4)),
            _p(_fmt_num(mc.get("q_internal"), 4)),
            _p(f"{_fmt_num(mc.get('M_potential'), 2)} / {_fmt_num(mc.get('c_churn'), 2)}"),
            _p(_fmt_pct(mc.get("cumulative_adoption"))),
            _p(_fmt_pct(mc.get("active_adoption"))),
            _p(f"{_fmt_num((mc.get('adoption_velocity') or 0) * 100, 2, True)}%"),
            _p(_fmt_pct(mc.get("attention_indicator"))),
            _p(f"{_fmt_num(mc.get('attention_vs_adoption_ratio'), 2)}x"),
        ])
    story.append(_styled_table(macro_rows, [40.0, 54.0, 54.0, 64.0, 64.0, 62.0, 62.0, 62.0, CONTENT_WIDTH - 462.0]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("8.2 Feedback Layer: Iteration-by-Iteration Systemic Evolution", st_h2))
    story.append(
        KeepTogether([
            _chart_feedback_evolution(iterations),
            Spacer(1, 3),
            Paragraph(
                "Figure 4: Iteration-by-iteration claim verification audit (Supported / Inferred / Unsupported / Failed) and polarization evolution.",
                st_caption,
            ),
        ])
    )
    story.append(Spacer(1, 4))

    fb_rows = [[_ph("Round"), _ph("Systemic Posture & Escalation Proxy"), _ph("Claim Audit (S/I/U/F)"), _ph("What Changed & Why (Simulated State Deltas)")]]
    for it in iterations:
        fb = it.get("feedback_state") or {}
        ca = fb.get("claim_verification_audit") or {}
        sys_st = fb.get("systemic_state") or {}
        ca_str = f"{ca.get('supported', 0)} / {ca.get('inferred', 0)} / {ca.get('unsupported', 0)} / {ca.get('failed_interpretations', 0)}"
        posture_str = (
            f"{sys_st.get('institutional_posture', 'Monitored')}\n"
            f"Esc Proxy: {_fmt_pct(sys_st.get('escalation_risk')) if sys_st.get('escalation_risk') is not None else 'N/A'}\n"
            f"Pol: {_fmt_num(fb.get('polarization_index'), 3)}"
        )
        wc_clean = "; ".join(str(x).strip().rstrip(".") for x in (fb.get("what_changed") or [])[:2] if str(x).strip())
        delta_txt = (
            f"{wc_clean or str(fb.get('iteration_change_summary') or '').strip()}\n"
            f"Why: {fb.get('why_changed') or 'Feedback propagation cycle.'}"
        )
        fb_rows.append([
            _p(str(it.get("iteration"))),
            _p(posture_str),
            _p(ca_str),
            _p(delta_txt),
        ])
    story.append(_styled_table(fb_rows, [38.0, 130.0, 76.0, CONTENT_WIDTH - 244.0], long_tbl=True))
    story.append(Spacer(1, 6))

    first_meso = first_iter.get("meso_state") or {}
    first_macro = first_iter.get("macro_state") or {}
    story.append(
        Paragraph(
            f"<b>Cross-Layer Coupling Audit (Simulated Bass Proxies):</b> Initial Round 1 pressures (External={_fmt_pct(first_meso.get('external_influence_score'))}, "
            f"Internal={_fmt_pct(first_meso.get('internal_contagion_score'))}) parameterized Bass coefficients <i>p</i> "
            f"({_fmt_num(first_macro.get('p_external'), 4)} -> {_fmt_num(last_macro.get('p_external'), 4)}) and <i>q</i> "
            f"({_fmt_num(first_macro.get('q_internal'), 4)} -> {_fmt_num(last_macro.get('q_internal'), 4)}), shifting simulated active adoption proxy "
            f"from {_fmt_pct(first_macro.get('active_adoption'))} to {_fmt_pct(last_macro.get('active_adoption'))} (NOT empirical population adoption telemetry) "
            f"while feeding verified anchors and simulated systemic states forward into subsequent rounds.",
            st_body,
        )
    )

    # ── SECTION 9: CAUSAL BUTTERFLY-EFFECT DIAGRAMS ───────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("9. Causal Butterfly-Effect Diagrams", st_h1))
    story.append(
        Paragraph(
            "Competing multi-order pathways map how the initial trigger branches into 1st-order actions, 2nd-order spillovers, "
            "3rd-order structural shifts, and feedback loops. Links resting on unsupported claims or >=3 stacked assumptions "
            "are automatically flagged as weak causal links.",
            st_body,
        )
    )
    story.append(
        KeepTogether([
            _chart_butterfly_causal_diagram(causal_pathways, event_title),
            Spacer(1, 3),
            Paragraph(
                "Figure 5: Multi-order causal Butterfly-Effect branching pathways from Initial Event through 1st-, 2nd-, 3rd-order, and Systemic feedback states.",
                st_caption,
            ),
        ])
    )
    story.append(Spacer(1, 4))

    if causal_pathways:
        pw_rows = [[_ph("Pathway / Order"), _ph("Cause -> Effect & Actor Response"), _ph("Mechanism, Status & Invalidation"), _ph("Conf / Missing & Weak-Link Disclosure")]]
        for pw in causal_pathways:
            for st in (pw.get("steps") or []):
                flags: List[str] = []
                if st.get("is_missing_link"):
                    flags.append(f"MISSING LINK: {st.get('missing_link_reason') or 'Intermediate mechanism absent.'}")
                if st.get("is_weak_link"):
                    flags.append(f"WEAK LINK: {st.get('weak_link_reason')}")
                wk_flag = "\n".join(flags) if flags else "Grounded / Bounded"
                actor = st.get("affected_actor") or ""
                resp = st.get("behavioral_or_institutional_response") or ""
                inv_c = st.get("invalidation_condition") or ""
                ce_txt = (
                    f"Cause: {st.get('cause', '')}\nEffect: {st.get('effect', '')}"
                    + (f"\nActor: {actor}" if actor else "")
                    + (f"\nResponse: {resp}" if resp else "")
                )
                mech_txt = (
                    f"Mech: {st.get('mechanism', '')}\nStatus: {str(st.get('evidence_status', '')).upper()}"
                    + (f"\nInvalidated if: {inv_c}" if inv_c else "")
                )
                pw_rows.append([
                    _p(f"{pw.get('name', '')}\n[{st.get('order', '').replace('_', ' ')} | {st.get('time_horizon', '')}]"),
                    _p(ce_txt),
                    _p(mech_txt),
                    _p(f"Conf: {_fmt_pct(st.get('confidence'))} | Unc: {_fmt_num(st.get('uncertainty'), 2)}\n{wk_flag}"),
                ])
        story.append(_styled_table(pw_rows, [105.0, 155.0, 150.0, CONTENT_WIDTH - 410.0], long_tbl=True))
    else:
        story.append(Paragraph("No causal pathway steps available due to failed model execution.", st_body))

    # ── SECTION 10: FOUR FUTURE SCENARIOS ─────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("10. Four Future Scenarios (Baseline, Positive, Negative, Extreme)", st_h1))
    story.append(
        _callout_box(
            "Calibration Warning (Heuristic Weights vs Empirical Probabilities)",
            _esc((scenarios[0].get("calibration_warning") if scenarios else None) or final_verdict.get("calibration_warning") or ""),
            border_col=AMBER,
            bg_col=AMBER_BG,
        )
    )
    story.append(Spacer(1, 8))

    scen_rows = [[_ph("Scenario & Weight"), _ph("Trajectory Summary & Horizon Effects"), _ph("Triggers & Affected Stakeholders"), _ph("Model Hypotheses vs Counter-Evidence")]]
    for sc in scenarios:
        w = sc.get("conditional_weight")
        w_str = f"{_fmt_pct(w)} weight\n{sc.get('probability_band', 'N/A')}" if w is not None else "N/A (Failed Run)"
        effects_txt = (
            f"{sc.get('summary', '')}\n"
            f"Short: {_join_clean((sc.get('short_term_consequences') or [])[:2])}\n"
            f"Medium: {_join_clean((sc.get('medium_term_consequences') or [])[:2])}\n"
            f"Long: {_join_clean((sc.get('long_term_consequences') or [])[:2])}"
        )
        trig_txt = (
            f"Triggers: {_join_clean((sc.get('trigger_conditions') or [])[:3])}\n"
            f"Stakeholders: {', '.join((sc.get('affected_stakeholders') or [])[:4])}"
        )
        ev_txt = (
            f"For (Hypotheses & Sim Telemetry): {_join_clean((sc.get('supporting_evidence') or [])[:3])}\n"
            f"Opposing / Counter-Evidence: {_join_clean((sc.get('counter_evidence') or [])[:2])}"
        )
        scen_rows.append([
            _p(f"{str(sc.get('scenario_type', '')).upper()}\n{sc.get('title', '')}\n({w_str})"),
            _p(effects_txt),
            _p(trig_txt),
            _p(ev_txt),
        ])
    story.append(_styled_table(scen_rows, [105.0, 165.0, 128.0, CONTENT_WIDTH - 398.0], long_tbl=True))

    # ── SECTION 11: TIME-HORIZON FORECAST ─────────────────────────────────────
    story.append(Paragraph("11. Time-Horizon Forecast &amp; Reversal Conditions", st_h1))
    th_rows = [
        [_ph("Horizon Window"), _ph("Hypothesized Trajectory & Consequences")],
        [_p("Immediate (0 - 7 Days)"), _p_bullets(time_horizon_fc.get("immediate_days"), "Unavailable")],
        [_p("Short-Term (1 - 6 Weeks)"), _p_bullets(time_horizon_fc.get("short_term_weeks"), "Unavailable")],
        [_p("Medium-Term (2 - 6 Months)"), _p_bullets(time_horizon_fc.get("medium_term_months"), "Unavailable")],
        [_p("Long-Term (6 - 24 Months)"), _p_bullets(time_horizon_fc.get("long_term_quarters"), "Unavailable")],
        [_p("Low-Prob / High-Impact Tail Risks"), _p_bullets(time_horizon_fc.get("unexpected_tail_risks"), "None recorded")],
        [_p("Prevention & Reversal Conditions"), _p_bullets(order_effects.get("prevention_and_reversal_conditions"), "Rapid verified de-escalation")],
    ]
    story.append(_styled_table(th_rows, [145.0, CONTENT_WIDTH - 145.0], long_tbl=True))

    # ── SECTION 12: STAKEHOLDER IMPACT ASSESSMENT ─────────────────────────────
    story.append(Paragraph("12. Stakeholder Impact Assessment", st_h1))
    if stakeholder_reactions:
        imp_rows = [[_ph("Affected Stakeholder"), _ph("Primary Exposure Channel"), _ph("Potential Upside / Resilience Lever"), _ph("Downside Vulnerability")]]
        for sr in stakeholder_reactions:
            actions = sr.get("likely_actions") or []
            upside_txt = sr.get("resilience_lever") or (
                actions[0]
                if (actions and str(sr.get("impact_direction", "")).lower() == "stabilizing")
                else "Constructive verification and modular contingency adaptation"
            )
            downside_txt = sr.get("downside_vulnerability") or sr.get("behavioural_shift") or "Heightened uncertainty and coordination friction"
            imp_rows.append([
                _p(sr.get("stakeholder") or ""),
                _p(f"{sr.get('meso_node_id', 'network')} (Activation: {_fmt_pct(sr.get('activation_level')) if sr.get('activation_level') is not None else 'N/A'})"),
                _p(upside_txt),
                _p(downside_txt),
            ])
        story.append(_styled_table(imp_rows, [115.0, 120.0, 145.0, CONTENT_WIDTH - 380.0]))
    else:
        story.append(Paragraph("Stakeholder impact assessment unavailable due to insufficient valid model runs.", st_body))

    # ── SECTION 13: RISK & UNCERTAINTY ANALYSIS ───────────────────────────────
    story.append(Paragraph("13. Risk &amp; Uncertainty Analysis (Methodological Boundaries)", st_h1))
    weak_links_count = sum(int(pw.get("weak_link_count", 0)) for pw in causal_pathways)
    lims = [
        ("Correlation vs Causation Guardrail", str(final_verdict.get("correlation_vs_causation_note") or "Distinguishes signal correlation from causal mechanism.")),
        ("Weak & Speculative Link Audit", f"Automated audit flagged {weak_links_count} weak/unsupported causal link(s) across {len(causal_pathways)} competing pathways due to unverified premises or >=3 stacked assumptions."),
        ("Synthetic vs Empirical Separation", "Micro metrics (stance, virality, emotion, LLM credibility) are synthetic language-model estimates; Macro adoption curves derive from deterministic Bass ODEs, NOT observed population telemetry."),
        ("Missing / Failed Execution Impact", f"Valid slots: {succ_cnt}/{exp_cnt}. Failed slots contribute zero weight and are never silently replaced by mock fallbacks."),
        ("Information Gaps Identified", _join_clean(final_verdict.get("evidence_and_information_gaps") or ["No critical structural gaps flagged."])),
    ]
    trace_refs_sec13 = final_verdict.get("traceability_refs") or {}
    rel_audit_sec13 = trace_refs_sec13.get("relevance_audit") or {}
    if trace_refs_sec13.get("event_domain"):
        off_dom_cnt = rel_audit_sec13.get("rejected_off_domain_count", rel_audit_sec13.get("rejected_off_domain", 0))
        withheld_cnt = rel_audit_sec13.get("withheld_numeric_count", rel_audit_sec13.get("withheld_numeric", 0))
        prem_cnt = rel_audit_sec13.get("filtered_premise_actions_count", rel_audit_sec13.get("filtered_premise_actions", 0))
        hor_cnt = rel_audit_sec13.get("filtered_horizon_mismatches_count", rel_audit_sec13.get("filtered_horizon_mismatches", 0))
        reas_cnt = rel_audit_sec13.get("filtered_reassuring_risks_count", rel_audit_sec13.get("filtered_reassuring_risks", 0))
        lims.append((
            "Domain Grounding & Causal Audit",
            f"Primary domain: {trace_refs_sec13.get('event_domain')} | "
            f"Off-domain items filtered: {off_dom_cnt} | "
            f"Ungrounded numeric figures reframed: {withheld_cnt} | "
            f"Premise/horizon/rationalization filters: {prem_cnt}/{hor_cnt}/{reas_cnt}.",
        ))
    lim_rows = [[_ph("Risk / Epistemic Dimension"), _ph("Audit Findings & Disclosure")]] + [[_p(k), _p(v)] for k, v in lims]
    story.append(_styled_table(lim_rows, [155.0, CONTENT_WIDTH - 155.0], long_tbl=True))

    # ── SECTION 14: KEY INDICATORS TO MONITOR ─────────────────────────────────
    story.append(Paragraph("14. Key Indicators &amp; Turning Points to Monitor Next", st_h1))
    ind_rows = [[_ph("Indicator & Measurable Variable"), _ph("Scenario"), _ph("Supports vs Contradicts Movement"), _ph("Threshold & Analyst Action")]]
    trace_refs = final_verdict.get("traceability_refs") or {}
    monitoring_matrix = final_verdict.get("monitoring_matrix") or trace_refs.get("monitoring_matrix") or []
    triggers = final_verdict.get("key_triggers_and_turning_points") or []
    if monitoring_matrix:
        for mm in monitoring_matrix:
            ind_label = str(mm.get("indicator") or "")
            meas_var = str(mm.get("measurable_variable") or "")
            col1_txt = ind_label if (not meas_var or meas_var == ind_label) else f"{ind_label}\nVar: {meas_var}"
            supp_txt = mm.get("supports_scenario_movement") or ""
            contr_txt = mm.get("contradicts_scenario_movement") or ""
            move_txt = (
                (f"Supports: {supp_txt}\n" if supp_txt else "")
                + (f"Contradicts: {contr_txt}" if contr_txt else "Watch trigger deviation vs baseline")
            )
            trig_th = mm.get("trigger_threshold") or ""
            act_txt = mm.get("analyst_action") or "Monitor empirical telemetry against scenario assumptions"
            col4_txt = (f"Trigger: {trig_th}\n" if trig_th and trig_th not in act_txt else "") + act_txt
            ind_rows.append([
                _p(col1_txt),
                _p(str(mm.get("scenario_type") or "all").upper()),
                _p(move_txt),
                _p(col4_txt),
            ])
    else:
        # Backwards-compatible accurate 1-to-1 reconstruction for legacy persisted sessions
        matched_rows: List[Tuple[str, str, str]] = []
        seen_ind: set[str] = set()
        for sc in scenarios:
            stype = str(sc.get("scenario_type") or "all").upper()
            sc_trigs = sc.get("trigger_conditions") or []
            trig_hint = sc_trigs[0] if sc_trigs else "Compare observed reality against scenario assumptions"
            for ind in (sc.get("indicators_to_monitor") or []):
                if ind.lower() not in seen_ind:
                    seen_ind.add(ind.lower())
                    matched_rows.append((ind, stype, f"Evaluate whether telemetry matches {stype} trigger: {trig_hint}"))
        if not matched_rows:
            matched_rows = [("Track primary empirical updates vs social amplification hype.", "ALL", "Compare observed outcomes against baseline scenario assumptions")]
        for ind, stype, act in matched_rows[:8]:
            ind_rows.append([_p(ind), _p(stype), _p("Compare empirical trend vs baseline expectation"), _p(act)])
    story.append(_styled_table(ind_rows, [185.0, 56.0, 142.0, CONTENT_WIDTH - 383.0], long_tbl=True))

    # ── SECTION 15: FINAL VERDICT & TECHNICAL APPENDIX ────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("15. Final Verdict &amp; Technical Appendix", st_h1))
    summary_method_str = str(summary_meta.get("method") or "deterministic_grounded_extractive")
    if summary_meta.get("fallback_used"):
        summary_method_str += f" (fallback: {summary_meta.get('fallback_reason')})"
    signoff_rows = [
        [_ph("Audit / Verdict Field"), _ph("Recorded Technical State")],
        [_p("Verdict Headline"), _p(final_verdict.get("headline") or "Unavailable")],
        [_p("Leading Branch & Weight"), _p(f"{lead_type_lbl} ({lead_w_str} heuristic index, non-empirical)")],
        [_p("Confidence & Uncertainty"), _p(f"{conf_str} — {conf_band_str} | Uncertainty Index: {_fmt_num(final_verdict.get('uncertainty'), 2)}")],
        [_p("Executive Summary Provenance"), _p(f"{summary_method_str} ({len(forecast_summary_txt.split())} words)")],
        [_p("Key Turning Points"), _p(_join_clean(triggers[:4]) or "Unavailable")],
    ]
    story.append(_styled_table(signoff_rows, [155.0, CONTENT_WIDTH - 155.0]))
    story.append(Spacer(1, 8))

    # Appendix A: Full multi-iteration outputs
    story.append(Paragraph("Appendix A: Complete Per-Iteration Micro Outputs", st_h2))
    app_rows = [[_ph("Iter"), _ph("Archetype"), _ph("Slot / Model"), _ph("Status"), _ph("S / V / C / E"), _ph("Narrative Summary")]]
    for it in iterations:
        it_n = it.get("iteration")
        for m in (it.get("micro_interpretations") or []):
            valid = _is_valid_micro(m)
            app_rows.append([
                _p(str(it_n)),
                _p(m.get("archetype_name") or m.get("archetype")),
                _p(f"{m.get('model_slot') or '?'} ({m.get('model_name')})"),
                _p(f"{m.get('execution_status', 'success')} / {m.get('evidence_status', '-')}"),
                _p(
                    f"{_fmt_num(m.get('stance'), 2, True)}/{_fmt_num(m.get('virality'), 2)}/{_fmt_num(m.get('credibility'), 2)}/{_fmt_num(m.get('emotional_charge'), 2)}"
                    if valid else "FAILED"
                ),
                _p(m.get("primary_narrative") if valid else f"[FAILED: {m.get('error_category')}] {m.get('error_message', '')}"),
            ])
    story.append(_styled_table(app_rows, [30.0, 72.0, 76.0, 62.0, 94.0, CONTENT_WIDTH - 334.0], long_tbl=True))

    # Appendix B: Model Diagnostics
    if model_diagnostics:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Appendix B: Model Execution Diagnostics Telemetry", st_h2))
        diag_rows = [[_ph("Iter"), _ph("Slot"), _ph("Archetype"), _ph("Provider / Model"), _ph("Status"), _ph("Retries"), _ph("Time (s)"), _ph("Finish / Len")]]
        for d in model_diagnostics:
            diag_rows.append([
                _p(str(d.get("iteration", ""))),
                _p(str(d.get("model_slot", ""))),
                _p(str(d.get("archetype", ""))),
                _p(f"{d.get('provider')}/{d.get('model_name')}"),
                _p(str(d.get("execution_status", ""))),
                _p(str(d.get("retry_count", 0))),
                _p(str(d.get("duration_sec", 0.0))),
                _p(f"{d.get('finish_reason') or '-'} / {d.get('response_length') or 0}"),
            ])
        story.append(_styled_table(diag_rows, [28.0, 48.0, 82.0, 96.0, 66.0, 46.0, 48.0, CONTENT_WIDTH - 414.0], long_tbl=True))

    # Appendix C: Formulas
    story.append(Spacer(1, 8))
    story.append(Paragraph("Appendix C: Mathematical Model Specification", st_h2))
    formulas = [
        "1. Evidence Credibility & Transmission Weight: C_eff = clip(C_llm * penalty(evidence) * (1 - 0.35*uncertainty), 0, 1); T_i = (0.20 + 0.80*|stance|) * (0.40*virality + 0.35*C_eff + 0.25*emotional_charge) * ground_factor",
        "2. Meso Instantaneous Node Activation: A_node(t) = clip(sum_j(w_j->node * T_j) / Z_node, 0.0, 1.0); Raw Inflow = sum_j(w_j->node * T_j) preserved independently",
        "3. Effective Spread Potential: Spread = alpha * External_Influence + (1 - alpha) * Internal_Contagion",
        "4. Modified Bass Diffusion: dN/dt = (p + q * (N/M)) * (M - N); N(t+1) = min(M, N(t) + dN/dt); A(t+1) = clip(A(t) + dN/dt - c * A(t), 0, N(t+1))",
        "5. Scenario Brier Score (Post-Event Tracking): Brier = sum((w_k - y_k)^2) across {Baseline, Positive, Negative, Extreme}",
    ]
    for fm in formulas:
        story.append(Paragraph(_esc(fm), st_formula))

    # Appendix D: Source References
    story.append(Paragraph("Appendix D: Source References &amp; File Traceability", st_h2))
    story.append(
        Paragraph(
            f"• <b>Simulation ID:</b> {_esc(sim_id)}<br/>"
            f"• <b>Source URL:</b> {_esc(source_url or 'None provided in configuration')}<br/>"
            f"• <b>Source SHA-256 Digest:</b> {_esc(source_sha or 'In-memory payload')}<br/>"
            f"• <b>Report File Name:</b> {_esc(download_filename)}",
            st_body,
        )
    )

    # Atomic PDF Build via BaseDocTemplate (First Page standalone frame -> Later Pages headered frame)
    tmp_pdf_path = f"{pdf_path}.tmp"
    frame_first = Frame(
        MARGIN_X,
        PAGE1_BOTTOM_MARGIN,
        CONTENT_WIDTH,
        PAGE_HEIGHT - PAGE1_TOP_MARGIN - PAGE1_BOTTOM_MARGIN,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="first_frame",
    )
    frame_later = Frame(
        MARGIN_X,
        MARGIN_BOTTOM,
        CONTENT_WIDTH,
        PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="later_frame",
    )
    doc = BaseDocTemplate(
        tmp_pdf_path,
        pagesize=letter,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title=f"Butterfly Algorithm Report — {event_title}",
        author="The Butterfly Algorithm v4.2.0",
    )
    doc.addPageTemplates([
        PageTemplate(id="First", frames=[frame_first]),
        PageTemplate(id="Later", frames=[frame_later]),
    ])

    def _canvas_factory(*c_args, **c_kwargs):
        return _NumberedResearchCanvas(
            *c_args,
            header_title=event_title,
            sim_id=sim_id,
            status_label=status_label,
            generated_ts=generated_ts,
            **c_kwargs,
        )

    doc.build(story, canvasmaker=_canvas_factory)
    os.replace(tmp_pdf_path, pdf_path)

    file_size = os.path.getsize(pdf_path)
    meta_record = {
        "simulation_id": sim_id,
        "event_title": event_title,
        "filename": download_filename,
        "generated_at": generated_iso,
        "source_sha256": source_sha,
        "file_size_bytes": file_size,
        "status": sim_status,
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta_record, mf, indent=2)

    return {
        "simulation_id": sim_id,
        "event_title": event_title,
        "report_exists": True,
        "cached": False,
        "pdf_path": pdf_path,
        "filename": download_filename,
        "file_size_bytes": file_size,
        "generated_at": generated_iso,
        "download_url": f"/api/simulations/{sim_id}/report/download",
    }
