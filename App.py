# ══════════════════════════════════════════════════════════════════
# app.py — Mshauri wa Malipo ya Kigeni | Tanzania  (v3 – Premium UI)
# ══════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import warnings
from datetime import timedelta, date
warnings.filterwarnings('ignore')

import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mshauri wa Malipo ya Kigeni",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="auto"
)

# ─────────────────────────────────────────────────────────────────
# PREMIUM CSS — 3D cards, animated gradients, glassmorphism
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ══ GLOBAL RESET ══ */
html, body, .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background: #f0f2ff !important;
}

/* Animated mesh background */
.stApp {
    background:
        radial-gradient(ellipse at 20% 20%, rgba(99,102,241,0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 10%, rgba(139,92,246,0.08) 0%, transparent 45%),
        radial-gradient(ellipse at 60% 80%, rgba(59,130,246,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 10% 80%, rgba(16,185,129,0.06) 0%, transparent 45%),
        #eef0fb !important;
    background-attachment: fixed !important;
}

/* ══ KEYFRAMES ══ */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes floatUp {
    0%   { opacity:0; transform: translateY(18px); }
    100% { opacity:1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0.3), 0 20px 60px rgba(99,102,241,0.15), 0 8px 24px rgba(0,0,0,0.10); }
    50%       { box-shadow: 0 0 0 6px rgba(99,102,241,0.08), 0 20px 60px rgba(99,102,241,0.22), 0 8px 24px rgba(0,0,0,0.12); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
@keyframes fadeSlideIn {
    0%   { opacity:0; transform: translateY(10px) scale(0.98); }
    100% { opacity:1; transform: translateY(0)    scale(1);    }
}

/* ══ TOP HERO BANNER ══ */
.hero-banner {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 35%, #2563eb 70%, #0ea5e9 100%);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite, floatUp 0.6s ease both;
    border-radius: 24px;
    padding: 2.2rem 2.6rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 4px 6px rgba(79,70,229,0.12),
        0 20px 60px rgba(79,70,229,0.25),
        0 40px 100px rgba(79,70,229,0.12),
        inset 0 1px 0 rgba(255,255,255,0.25);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #fff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.8px;
    position: relative; z-index: 2;
}
.hero-sub {
    font-size: 0.92rem;
    color: rgba(255,255,255,0.82);
    margin: 0;
    position: relative; z-index: 2;
}
.hero-rate-pill {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.30);
    border-radius: 16px;
    padding: 0.9rem 1.4rem;
    position: relative; z-index: 2;
    display: inline-block;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.2), 0 4px 16px rgba(0,0,0,0.15);
}
.hero-rate-pill .hrp-label {
    font-size: 0.68rem;
    color: rgba(255,255,255,0.75);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}
.hero-rate-pill .hrp-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
    color: #fff;
    line-height: 1.15;
}
.hero-rate-pill .hrp-change {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.88);
}

/* ══ 3D GLASS METRIC CARDS ══ */
.glass-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.75) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.80);
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlideIn 0.5s ease both;
    transition: transform 0.25s cubic-bezier(.34,1.56,.64,1),
                box-shadow 0.25s ease;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.9) inset,
        0 -1px 0 rgba(0,0,0,0.04) inset,
        2px 4px 8px rgba(0,0,0,0.06),
        0 12px 32px rgba(79,70,229,0.08),
        0 24px 48px rgba(0,0,0,0.06);
}
.glass-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.9) inset,
        0 -1px 0 rgba(0,0,0,0.04) inset,
        2px 4px 8px rgba(0,0,0,0.08),
        0 20px 48px rgba(79,70,229,0.15),
        0 32px 64px rgba(0,0,0,0.10);
}
/* Gradient accent line on top */
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 20px 20px 0 0;
}
.glass-card.g-indigo::before  { background: linear-gradient(90deg, #4f46e5, #7c3aed); }
.glass-card.g-violet::before  { background: linear-gradient(90deg, #7c3aed, #ec4899); }
.glass-card.g-emerald::before { background: linear-gradient(90deg, #059669, #0ea5e9); }
.glass-card.g-amber::before   { background: linear-gradient(90deg, #f59e0b, #ef4444); }
.glass-card.g-rose::before    { background: linear-gradient(90deg, #f43f5e, #f97316); }
.glass-card.g-sky::before     { background: linear-gradient(90deg, #0ea5e9, #6366f1); }

.gc-label {
    font-size: 0.70rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6b7280;
    margin-bottom: 0.4rem;
}
.gc-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #111827;
    line-height: 1.2;
}
.gc-note {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 0.25rem;
}
.gc-icon {
    position: absolute;
    top: 1rem; right: 1.2rem;
    font-size: 1.6rem;
    opacity: 0.15;
}

/* ══ 3D RECOMMENDATION HERO CARDS ══ */
.rec-3d {
    border-radius: 22px;
    padding: 2rem 2.2rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlideIn 0.6s ease both, pulseGlow 4s ease-in-out infinite;
    transition: transform 0.3s cubic-bezier(.34,1.56,.64,1);
}
.rec-3d:hover { transform: translateY(-5px) scale(1.008); }

.rec-3d::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    opacity: 0.18;
    background: rgba(255,255,255,0.6);
}

/* LIPA SASA — green gradient */
.rec-go {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 30%, #6ee7b7 60%, #34d399 100%);
    border: 1.5px solid rgba(16,185,129,0.40);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.8) inset,
        0 20px 60px rgba(16,185,129,0.25),
        0 8px 24px rgba(16,185,129,0.15),
        0 2px 4px rgba(0,0,0,0.06);
    color: #064e3b;
}

/* SUBIRI — amber gradient */
.rec-wait {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 30%, #fcd34d 60%, #fbbf24 100%);
    border: 1.5px solid rgba(245,158,11,0.40);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.8) inset,
        0 20px 60px rgba(245,158,11,0.22),
        0 8px 24px rgba(245,158,11,0.15),
        0 2px 4px rgba(0,0,0,0.06);
    color: #78350f;
}

/* ANGALIA — blue gradient */
.rec-watch {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 30%, #93c5fd 60%, #60a5fa 100%);
    border: 1.5px solid rgba(59,130,246,0.35);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.8) inset,
        0 20px 60px rgba(59,130,246,0.22),
        0 8px 24px rgba(59,130,246,0.15),
        0 2px 4px rgba(0,0,0,0.06);
    color: #1e3a8a;
}

.rec-3d-action {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 0.7rem;
    letter-spacing: -0.5px;
    position: relative; z-index: 2;
}
.rec-3d-headline {
    font-size: 0.95rem;
    line-height: 1.65;
    margin-bottom: 0.6rem;
    position: relative; z-index: 2;
    font-weight: 500;
}
.rec-3d-purpose {
    font-size: 0.88rem;
    font-style: italic;
    opacity: 0.82;
    position: relative; z-index: 2;
}

/* ══ GRADIENT PAYMENT BOXES ══ */
.pay-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 1rem 0;
}
.pay-3d {
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s cubic-bezier(.34,1.56,.64,1), box-shadow 0.25s;
    animation: fadeSlideIn 0.5s ease both;
}
.pay-3d:hover {
    transform: translateY(-3px) scale(1.02);
}
.pay-today {
    background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 50%, #a5b4fc 100%);
    border: 1px solid rgba(99,102,241,0.30);
    box-shadow: 0 8px 32px rgba(99,102,241,0.18), 0 2px 8px rgba(0,0,0,0.06),
                inset 0 1px 0 rgba(255,255,255,0.7);
    color: #1e1b4b;
}
.pay-future-okoa {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 50%, #6ee7b7 100%);
    border: 1px solid rgba(16,185,129,0.30);
    box-shadow: 0 8px 32px rgba(16,185,129,0.18), 0 2px 8px rgba(0,0,0,0.06),
                inset 0 1px 0 rgba(255,255,255,0.7);
    color: #064e3b;
}
.pay-future-hasara {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 50%, #fca5a5 100%);
    border: 1px solid rgba(239,68,68,0.30);
    box-shadow: 0 8px 32px rgba(239,68,68,0.15), 0 2px 8px rgba(0,0,0,0.06),
                inset 0 1px 0 rgba(255,255,255,0.7);
    color: #7f1d1d;
}
.pay-future-stable {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 50%, #93c5fd 100%);
    border: 1px solid rgba(59,130,246,0.30);
    box-shadow: 0 8px 32px rgba(59,130,246,0.15), 0 2px 8px rgba(0,0,0,0.06),
                inset 0 1px 0 rgba(255,255,255,0.7);
    color: #1e3a8a;
}
.pay-3d .pb-label { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.70; margin-bottom: 0.4rem; }
.pay-3d .pb-val   { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; line-height: 1.2; }
.pay-3d .pb-note  { font-size: 0.73rem; opacity: 0.70; margin-top: 0.25rem; }

/* ══ SAVINGS BANNER ══ */
.savings-banner {
    border-radius: 16px;
    padding: 1.1rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 0.5rem;
    animation: fadeSlideIn 0.7s ease both;
}
.sb-okoa   {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid rgba(16,185,129,0.35);
    box-shadow: 0 4px 20px rgba(16,185,129,0.12);
    color: #064e3b;
}
.sb-hasara {
    background: linear-gradient(135deg, #fff1f2, #fee2e2);
    border: 1px solid rgba(239,68,68,0.30);
    box-shadow: 0 4px 20px rgba(239,68,68,0.10);
    color: #7f1d1d;
}
.sb-stable {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid rgba(59,130,246,0.28);
    box-shadow: 0 4px 20px rgba(59,130,246,0.10);
    color: #1e3a8a;
}
.sb-label  { font-size: 0.82rem; font-weight: 500; }
.sb-sub    { font-size: 0.73rem; opacity: 0.72; margin-top: 2px; }
.sb-amount { font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 600; }

/* ══ INSIGHT CARDS (best day) ══ */
.insight-3d {
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlideIn 0.6s ease both;
    transition: transform 0.25s cubic-bezier(.34,1.56,.64,1);
}
.insight-3d:hover { transform: translateY(-4px) scale(1.02); }
.insight-3d::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 100px; height: 100px;
    border-radius: 50%;
    background: rgba(255,255,255,0.25);
}
.ins-purple {
    background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 50%, #c4b5fd 100%);
    border: 1px solid rgba(139,92,246,0.28);
    box-shadow: 0 8px 32px rgba(139,92,246,0.18), inset 0 1px 0 rgba(255,255,255,0.6);
    color: #2e1065;
}
.ins-indigo {
    background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 50%, #a5b4fc 100%);
    border: 1px solid rgba(99,102,241,0.28);
    box-shadow: 0 8px 32px rgba(99,102,241,0.18), inset 0 1px 0 rgba(255,255,255,0.6);
    color: #1e1b4b;
}
.ins-teal {
    background: linear-gradient(135deg, #ccfbf1 0%, #99f6e4 50%, #5eead4 100%);
    border: 1px solid rgba(20,184,166,0.28);
    box-shadow: 0 8px 32px rgba(20,184,166,0.18), inset 0 1px 0 rgba(255,255,255,0.6);
    color: #042f2e;
}
.ins-label { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.72; margin-bottom: 0.4rem; }
.ins-value { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 700; }
.ins-note  { font-size: 0.76rem; opacity: 0.72; margin-top: 0.25rem; }

/* ══ DATA WARNING ══ */
.data-warn-3d {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1px solid rgba(245,158,11,0.40);
    border-left: 4px solid #f59e0b;
    border-radius: 14px;
    padding: 0.8rem 1.1rem;
    font-size: 0.83rem;
    color: #78350f;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 16px rgba(245,158,11,0.12);
}

/* ══ SECTION EYEBROW ══ */
.eyebrow {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #6366f1;
    margin-bottom: 0.55rem;
}

/* ══ MODEL CARDS ══ */
.model-3d {
    background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(240,245,255,0.85));
    border: 1px solid rgba(255,255,255,0.80);
    border-radius: 18px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    box-shadow: 0 4px 20px rgba(99,102,241,0.08), 0 1px 0 rgba(255,255,255,0.9) inset,
                0 12px 32px rgba(0,0,0,0.06);
    transition: transform 0.22s cubic-bezier(.34,1.56,.64,1), box-shadow 0.22s;
    animation: fadeSlideIn 0.5s ease both;
}
.model-3d:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15), 0 1px 0 rgba(255,255,255,0.9) inset;
}
.model-3d .mn  { font-weight: 600; font-size: 0.95rem; color: #1e1b4b; }
.model-3d .md  { font-size: 0.78rem; color: #6b7280; margin-top: 3px; }
.model-3d .mr  { text-align: right; }
.model-3d .mel { font-size: 0.68rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.8px; }
.model-3d .mev { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.9rem; color: #4f46e5; }

/* ══ EXPLAINER ══ */
.explainer-3d {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8edff 100%);
    border: 1px solid rgba(99,102,241,0.20);
    border-left: 4px solid #6366f1;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    font-size: 0.87rem;
    line-height: 1.75;
    color: #2d2b5e;
    box-shadow: 0 4px 16px rgba(99,102,241,0.08);
}

/* ══ CURRENCY COMPARE ══ */
.cur-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(245,247,255,0.85));
    border: 1px solid rgba(255,255,255,0.78);
    border-radius: 16px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 16px rgba(99,102,241,0.07), 0 1px 0 rgba(255,255,255,0.9) inset;
    transition: transform 0.22s cubic-bezier(.34,1.56,.64,1);
}
.cur-card:hover { transform: translateY(-2px); }
.cur-card.best-cur {
    border-left: 3px solid #10b981;
    background: linear-gradient(145deg, rgba(209,250,229,0.85), rgba(167,243,208,0.60));
    box-shadow: 0 8px 28px rgba(16,185,129,0.15), 0 1px 0 rgba(255,255,255,0.9) inset;
}
.cur-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1rem;
    margin-bottom: 1rem;
}
.cur-row { display: flex; justify-content: space-between; padding: 0.35rem 0; font-size: 0.87rem; border-bottom: 1px solid rgba(0,0,0,0.04); }
.cur-row:last-child { border-bottom: none; }
.cur-lbl { color: #6b7280; }
.cur-val { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #111827; }

/* ══ URGENCY CHIP ══ */
.urg-chip {
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 30px;
    padding: 0.38rem 1rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.2px;
    margin-bottom: 1rem;
}
.urg-high   { background: linear-gradient(135deg,#fee2e2,#fecaca); color:#7f1d1d; border:1px solid rgba(239,68,68,.30); box-shadow:0 4px 14px rgba(239,68,68,.14); }
.urg-mid    { background: linear-gradient(135deg,#fef3c7,#fde68a); color:#78350f; border:1px solid rgba(245,158,11,.30); box-shadow:0 4px 14px rgba(245,158,11,.14); }
.urg-low    { background: linear-gradient(135deg,#d1fae5,#a7f3d0); color:#064e3b; border:1px solid rgba(16,185,129,.30); box-shadow:0 4px 14px rgba(16,185,129,.14); }

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #f8f7ff 0%, #f0f2ff 100%) !important;
    padding-top: 1rem;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f7ff 0%, #f0f2ff 100%) !important;
}
.sb-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #6366f1;
    margin: 1.1rem 0 0.45rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(99,102,241,0.20);
}
.preset-lbl {
    font-size: 0.70rem;
    font-weight: 600;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.4rem;
}

/* ══ TABS ══ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.70);
    backdrop-filter: blur(10px);
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
    border: 1px solid rgba(99,102,241,0.15);
    box-shadow: 0 4px 16px rgba(99,102,241,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.5rem 1.2rem;
    color: #6b7280;
    border: none;
    background: transparent;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
}

/* ══ PLOTLY CHART WRAPPER ══ */
.chart-wrap {
    background: rgba(255,255,255,0.88);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.80);
    padding: 1rem;
    box-shadow: 0 8px 32px rgba(99,102,241,0.10), 0 1px 0 rgba(255,255,255,0.9) inset;
    margin-bottom: 1rem;
}

/* ══ SHIMMER LOADER (optional accent) ══ */
.shimmer-line {
    height: 3px;
    border-radius: 3px;
    background: linear-gradient(90deg, #e0e7ff 25%, #c7d2fe 50%, #e0e7ff 75%);
    background-size: 200% 100%;
    animation: shimmer 1.8s infinite;
    margin-bottom: 1rem;
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

PURPOSES = {
    "💳 Netflix / Spotify / huduma za mtandaoni": "streaming",
    "📦 Bidhaa / mizigo kutoka nje (China, UAE…)": "bidhaa",
    "✈️  Tiketi / safari nje ya nchi":             "safari",
    "📤 Kutuma pesa kwa familia / rafiki":         "kutuma",
    "🎓 Ada ya masomo nje ya nchi":                "masomo",
    "💊 Dawa / huduma za afya kutoka nje":         "afya",
    "🔄 Malipo mengine ya kigeni":                 "nyingine",
}

CURRENCIES = {
    "USD 🇺🇸 — Dola ya Marekani": ("USD", "usd"),
    "EUR 🇪🇺 — Euro ya Ulaya":    ("EUR", "eur"),
    "CNY 🇨🇳 — Yuan ya China":    ("CNY", "cny"),
}

TIMEFRAMES = {
    "Wiki 1 (Siku 7)":    7,
    "Wiki 2 (Siku 14)":  14,
    "Mwezi 1 (Siku 30)": 30,
}

# ─────────────────────────────────────────────────────────────────
# SIMPLE TRANSLATION / I18N (Chinese, English, Kiswahili)
TRANSLATIONS = {
    'en': {
        'sidebar_title': '⚙️ Tell me more',
        'purpose_label': 'What are you paying for?',
        'currency_label': 'Payment currency',
        'timeframe_label': 'Check horizon',
        'refresh_button': '🔄 Refresh data & models',
        'payment_title': '🧮 Payment amount',
        'download_csv': '⤓ Download forecast CSV',
        'tab_advice': '💬 My advice',
        'tab_trend': '📈 Price trend',
        'tab_analysis': '🔬 Deep analysis',
        'compare_title': '🌍 Current currency comparison',
        'header_title': '💱 Foreign Payment Advisor',
        'header_sub': 'Helps you decide when to pay in USD, EUR, or CNY',
    },
    'sw': {
        'sidebar_title': '⚙️ Niambie Zaidi',
        'purpose_label': 'Unataka kulipa nini?',
        'currency_label': 'Sarafu ya malipo',
        'timeframe_label': 'Angalia mwenendo wa muda gani?',
        'refresh_button': '🔄 Rudisha Data & Modeli',
        'payment_title': '🧮 Kiasi cha Malipo',
        'download_csv': '⤓ Pakua utabiri kama CSV',
        'tab_advice': '💬 Ushauri Wangu',
        'tab_trend': '📈 Mwenendo wa Bei',
        'tab_analysis': '🔬 Uchambuzi wa Kina',
        'compare_title': '🌍 Kulinganisha Sarafu za Sasa',
        'header_title': '💱 Mshauri wa Malipo ya Kigeni',
        'header_sub': 'Inakusaidia kujua wakati mzuri wa kulipa kwa USD, EUR, au CNY — bila kuhitaji ujuzi wa fedha',
    },
    'zh': {
        'sidebar_title': '⚙️ 告诉我更多',
        'purpose_label': '你要支付什么？',
        'currency_label': '支付货币',
        'timeframe_label': '查看时间范围',
        'refresh_button': '🔄 刷新数据和模型',
        'payment_title': '🧮 支付金额',
        'download_csv': '⤓ 下载预测 CSV',
        'tab_advice': '💬 我的建议',
        'tab_trend': '📈 价格趋势',
        'tab_analysis': '🔬 深度分析',
        'compare_title': '🌍 当前货币比较',
        'header_title': '💱 外汇支付顾问',
        'header_sub': '帮助您决定何时以 USD、EUR 或 CNY 支付',
    }
}

def translate(key):
    lang = globals().get('LANG', 'sw')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

PRESETS = {
    "streaming": [("Netflix 15",  15.0),  ("Spotify 10",  10.0),  ("YouTube 14",  14.0)],
    "bidhaa":    [("Ndogo 100",  100.0),  ("Kati 500",   500.0),  ("Kubwa 2000", 2000.0)],
    "safari":    [("Ndege 300",  300.0),  ("Kati 800",   800.0),  ("Luxury 2k", 2000.0)],
    "kutuma":    [("Kidogo 50",   50.0),  ("Kati 200",   200.0),  ("Kubwa 500",  500.0)],
    "masomo":    [("Sem 1500",  1500.0),  ("Mwaka 3k", 3000.0),  ("Masters 6k",6000.0)],
    "afya":      [("Dawa 30",    30.0),   ("Tiba 200",   200.0),  ("Upasuaji 2k",2000.0)],
    "nyingine":  [("Kidogo 50",  50.0),   ("Kati 500",   500.0),  ("Kubwa 2k", 2000.0)],
}


# ─────────────────────────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    dfs = {}
    for cur, name in [('usd','USD'), ('eur','EUR'), ('cny','CNY')]:
        path = os.path.join(DATA_DIR, f'{cur}_tzs_clean.csv')
        df   = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        df   = df.sort_values('date').reset_index(drop=True)
        dfs[name] = df
    return dfs


@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    load_errors = {}   # NEW: collect what actually failed, per currency/model
    for cur in ['usd', 'eur', 'cny']:
        models[cur] = {'arima': None, 'prophet': None,
                       'xgboost': None, 'lstm': None, 'scaler': None}
        load_errors[cur] = {}
        try:
            models[cur]['arima'] = joblib.load(
                os.path.join(MODELS_DIR, f'arima_{cur}_model.pkl'))
        except Exception as e:
            load_errors[cur]['arima'] = str(e)
        try:
            models[cur]['arima'] = joblib.load(
                os.path.join(MODELS_DIR, f'arima_{cur}_model.pkl'))
        except Exception as e:
            load_errors[cur]['arima'] = str(e)
        try:
            models[cur]['prophet'] = joblib.load(
                os.path.join(MODELS_DIR, f'prophet_{cur}_model.pkl'))
        except Exception as e:
            load_errors[cur]['prophet'] = str(e)
        try:
            models[cur]['xgboost'] = joblib.load(
                os.path.join(MODELS_DIR, f'xgboost_{cur}_model_direct.pkl'))
        except Exception as e:
            load_errors[cur]['xgboost'] = str(e)
        try:
            import tensorflow as tf
            models[cur]['lstm'] = tf.keras.models.load_model(
                os.path.join(MODELS_DIR, f'lstm_{cur}_model.h5'), compile=False)
            models[cur]['scaler'] = joblib.load(
                os.path.join(MODELS_DIR, f'scaler_{cur}.pkl'))
        except Exception as e:
            load_errors[cur]['lstm'] = str(e)

        # Print to server-side logs (visible in Render "Logs" tab) so failures
        # are no longer silent.
        for name, err in load_errors[cur].items():
            print(f"[MODEL LOAD FAILED] currency={cur} model={name} error={err}")

    models['_load_errors'] = load_errors   # stashed here so UI code can show it
    return models


# ─────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────
def pred_arima(model, steps):
    try:
        return model.forecast(steps=steps).values
    except Exception:
        return None

def pred_prophet(model, steps):
    try:
        future   = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        return forecast['yhat'].tail(steps).values
    except Exception:
        return None

def _xgboost_features(df):
    """Must exactly match the feature engineering used in retrain_models.py"""
    vals = df['mean'].values
    i = len(vals) - 1
    l1  = vals[i]
    l3  = vals[i-2]  if i >= 2  else l1
    l7  = vals[i-6]  if i >= 6  else l1
    l14 = vals[i-13] if i >= 13 else l1
    l30 = vals[i-29] if i >= 29 else l1
    ma7  = vals[max(0, i-6):i+1].mean()
    ma30 = vals[max(0, i-29):i+1].mean()
    std7 = vals[max(0, i-6):i+1].std()
    d = df['date'].iloc[i]
    return np.array([[l1, l3, l7, l14, l30, ma7, ma30, std7,
                      d.month, d.weekday(), (d.month - 1) // 3 + 1]])

def pred_xgboost(model, df, steps):
    # NEW: this model now predicts all `steps` days DIRECTLY in one call from
    # real historical data — it no longer feeds its own guesses back in as
    # if they were real, which is what previously let errors compound into
    # impossible values (e.g. a negative exchange rate).
    try:
        feat = _xgboost_features(df)
        pred = model.predict(feat)[0]   # shape: (7,) — one shot, all days at once
        return np.array(pred[:steps])
    except Exception:
        return None

def pred_lstm(model, scaler, df, steps, seq=60):
    # NEW: this model now predicts all `steps` days DIRECTLY from the last
    # `seq` real days — same reasoning as XGBoost above, no recursive
    # feedback loop.
    try:
        if len(df) < seq:
            return None
        scaled = scaler.transform(df['mean'].values[-seq:].reshape(-1, 1)).flatten()
        X = scaled.reshape(1, seq, 1)
        pred_scaled = model.predict(X, verbose=0)[0]
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return pred[:steps]
    except Exception:
        return None

def get_ensemble(models_dict, df, cur_key, steps):
    # NEW: always try every model that actually loaded, instead of only
    # falling back to ARIMA/Prophet when BOTH XGBoost and LSTM fail.
    # This is what was silently changing predictions between environments —
    # if e.g. only LSTM failed to load, the ensemble used to quietly drop to
    # "XGBoost only" with no ARIMA/Prophet included, and no warning shown.
    forecasts = {}

    p = pred_xgboost(models_dict[cur_key]['xgboost'], df, steps)
    if p is not None: forecasts['XGBoost'] = p

    if models_dict[cur_key]['lstm'] and models_dict[cur_key]['scaler']:
        p = pred_lstm(models_dict[cur_key]['lstm'], models_dict[cur_key]['scaler'], df, steps)
        if p is not None: forecasts['LSTM'] = p

    p = pred_arima(models_dict[cur_key]['arima'], steps)
    if p is not None: forecasts['ARIMA'] = p

    # Prophet excluded from the ensemble average: backtesting showed it
    # consistently has the highest error of all 4 models across every
    # currency (4-15% MAPE vs <3% for the others) — not a bug, just a
    # genuine accuracy finding for this particular data. Still loaded so
    # its number is visible for comparison in the debug table.
    # p = pred_prophet(models_dict[cur_key]['prophet'], steps)
    # if p is not None: forecasts['Prophet'] = p

    if not forecasts:
        return None, {}
    ensemble = np.mean(list(forecasts.values()), axis=0)
    return ensemble, forecasts

def get_confidence_band(all_forecasts, ensemble):
    if len(all_forecasts) < 2:
        spread = ensemble * 0.005
        return ensemble - spread, ensemble + spread
    arr = np.array(list(all_forecasts.values()))
    std = np.std(arr, axis=0)
    return ensemble - std, ensemble + std


# ─────────────────────────────────────────────────────────────────
# BEST-DAY ANALYSIS
# ─────────────────────────────────────────────────────────────────
def best_day_analysis(df):
    temp = df.copy()
    temp['weekday']       = temp['date'].dt.day_name()
    temp['week_of_month'] = (temp['date'].dt.day - 1) // 7 + 1
    wd  = temp.groupby('weekday')['mean'].mean()
    wom = temp.groupby('week_of_month')['mean'].mean()
    best_wd  = wd.idxmin()
    best_wom = int(wom.idxmin())
    day_sw   = {'Monday':'Jumatatu','Tuesday':'Jumanne','Wednesday':'Jumatano',
                'Thursday':'Alhamisi','Friday':'Ijumaa',
                'Saturday':'Jumamosi','Sunday':'Jumapili'}
    wom_lbl  = {1:"Wiki ya 1",2:"Wiki ya 2",3:"Wiki ya 3",4:"Wiki ya 4"}
    return {
        'best_weekday': day_sw.get(best_wd, best_wd),
        'best_wom':     wom_lbl.get(best_wom, f"Wiki {best_wom}"),
        'avg_saving_wd': float(wd.max() - wd.min()),
    }


# ─────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────
def get_advice(current_rate, ensemble, payment_amount, cur_name, purpose_key, steps):
    avg_future = float(np.mean(ensemble))
    change_pct = (avg_future - current_rate) / current_rate * 100
    tzs_now    = payment_amount * current_rate
    tzs_later  = payment_amount * avg_future
    tzs_save   = tzs_now - tzs_later

    if change_pct > 1.0:
        action = "LIPA SASA"; rec_style = "rec-go"; icon = "⚡"
        headline = (f"Bei ya {cur_name} inatarajiwa KUPANDA kwa {change_pct:.1f}% "
                    f"katika siku {steps} zijazo. Ukilipa sasa utalipa TZS chache zaidi.")
        saving_direction = "hasara"; saving_amount = abs(tzs_save)
    elif change_pct < -1.0:
        action = "SUBIRI"; rec_style = "rec-wait"; icon = "⏳"
        headline = (f"Bei ya {cur_name} inatarajiwa KUSHUKA kwa {abs(change_pct):.1f}% "
                    f"katika siku {steps} zijazo. Ukisubiri utalipa TZS chache zaidi!")
        saving_direction = "okoa"; saving_amount = abs(tzs_save)
    else:
        action = "ANGALIA"; rec_style = "rec-watch"; icon = "👀"
        headline = (f"Bei ya {cur_name} iko STABLE sasa hivi "
                    f"(mabadiliko ya {change_pct:.1f}% tu).")
        saving_direction = "stable"; saving_amount = 0.0

    if abs(change_pct) >= 2.0:
        urg_level = "Juu";   urg_class = "urg-high"; urg_note = "Mabadiliko ya bei ni makubwa — chukua hatua haraka."
    elif abs(change_pct) >= 1.0:
        urg_level = "Kati";  urg_class = "urg-mid";  urg_note = "Mabadiliko ni wastani — angalia kabla ya kulipa."
    else:
        urg_level = "Ndogo"; urg_class = "urg-low";  urg_note = "Mabadiliko ni madogo — hatari ni ndogo kwa sasa."

    purpose_msgs = {
        "streaming": {"LIPA SASA":f"Lipia subscription sasa — bei itapanda na utapoteza zaidi ukisubiri. 🎬","SUBIRI":f"Ukisubiri siku {steps}, subscription itakugharimu TZS chache zaidi. 🎬","ANGALIA":f"Bei iko stable — unaweza kulipa wakati wowote. 🎬"},
        "bidhaa":    {"LIPA SASA":f"Lipia invoice ya {cur_name} {payment_amount:,.0f} sasa kabla bei haijapanda. 📦","SUBIRI":f"Subiri — bei itashuka na utaokoa. 📦","ANGALIA":f"Bei iko stable. Unaweza kulipa wakati wowote. 📦"},
        "safari":    {"LIPA SASA":f"Nunua {cur_name} kwa safari sasa — bei itapanda. ✈️","SUBIRI":f"Subiri — bei itashuka na utapata thamani nzuri. ✈️","ANGALIA":f"Bei iko stable. Unaweza kubadilisha pesa ya safari wakati wowote. ✈️"},
        "kutuma":    {"LIPA SASA":f"Tuma pesa sasa! Bei itapanda — mpokeaji atapata kidogo zaidi. 📤","SUBIRI":f"Subiri siku {steps} — bei itashuka. 📤","ANGALIA":f"Bei iko stable. Unaweza kutuma pesa wakati wowote. 📤"},
        "masomo":    {"LIPA SASA":f"Lipia ada sasa — bei itapanda na ada itakuwa ghali zaidi. 🎓","SUBIRI":f"Kama tarehe inaruhusu, subiri — bei itashuka. 🎓","ANGALIA":f"Bei iko stable — panga malipo kulingana na tarehe yako. 🎓"},
        "afya":      {"LIPA SASA":f"Lipia sasa — bei itapanda. Afya yako ni muhimu! 💊","SUBIRI":f"Kama hali inaruhusu, subiri — bei itashuka. 💊","ANGALIA":f"Bei iko stable. Unaweza kulipa wakati wowote. 💊"},
        "nyingine":  {"LIPA SASA":f"Fanya malipo sasa — bei itapanda.","SUBIRI":f"Subiri — bei itashuka.","ANGALIA":f"Bei iko stable. Unaweza kufanya malipo wakati wowote."},
    }
    purpose_text = purpose_msgs.get(purpose_key, purpose_msgs["nyingine"])[action]

    return {
        "action": action, "rec_style": rec_style, "icon": icon,
        "headline": headline, "purpose_text": purpose_text,
        "change_pct": change_pct,
        "tzs_now": tzs_now, "tzs_later": tzs_later, "tzs_save": tzs_save,
        "saving_direction": saving_direction, "saving_amount": saving_amount,
        "avg_future": avg_future,
        "urgency_level": urg_level, "urgency_class": urg_class, "urgency_note": urg_note,
    }


# ─────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────
all_data = {}; all_models = {}
with st.spinner("Inapakia mfumo..."):
    try:
        all_data   = load_data()
        all_models = load_models()
    except Exception as e:
        st.error(f"❌ Hitilafu ya kupakia: {e}"); st.stop()

if not all_data:
    st.error("❌ Data haikupatikana. Angalia folder ya 'data/'"); st.stop()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 0.8rem 0">
        <span style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
              background:linear-gradient(135deg,#4f46e5,#7c3aed);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            💱 Mshauri
        </span>
    </div>""", unsafe_allow_html=True)
    # Language selector
    lang_choice = st.selectbox("Language / Lugha / 语言", ["Kiswahili (sw)", "English (en)", "中文 (zh)"], index=0)
    if lang_choice.endswith("(sw)"):
        LANG = 'sw'
    elif lang_choice.endswith("(en)"):
        LANG = 'en'
    else:
        LANG = 'zh'

    st.markdown(f'<div class="sb-title">{translate("sidebar_title")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">🎯 Lengo la Malipo</div>', unsafe_allow_html=True)
    purpose_label = st.selectbox(translate('purpose_label'), list(PURPOSES.keys()), label_visibility="collapsed")
    purpose_key   = PURPOSES[purpose_label]

    st.markdown('<div class="sb-title">💱 Sarafu &amp; Muda</div>', unsafe_allow_html=True)
    currency_label = st.selectbox(translate('currency_label'), list(CURRENCIES.keys()), label_visibility="collapsed")
    cur_display, cur_key = CURRENCIES[currency_label]
    timeframe_label = st.selectbox(translate('timeframe_label'), list(TIMEFRAMES.keys()))
    steps = TIMEFRAMES[timeframe_label]

    st.markdown("**— au —** chagua tarehe maalum:")
    use_date = st.checkbox("Weka tarehe yangu mwenyewe")
    if use_date:
        target_date = st.date_input(
            "Tarehe ya kulipa",
            value=date.today() + timedelta(days=7),
            min_value=date.today() + timedelta(days=1),
            max_value=date.today() + timedelta(days=90),
        )
        steps = max(1, (target_date - date.today()).days)
        st.caption(f"📅 Siku {steps} hadi {target_date.strftime('%d/%m/%Y')}")

    st.markdown(f'<div class="sb-title">{translate("payment_title")}</div>', unsafe_allow_html=True)
    presets = PRESETS.get(purpose_key, PRESETS["nyingine"])
    st.markdown('<p class="preset-lbl">Chaguo la haraka:</p>', unsafe_allow_html=True)
    pc = st.columns(3)
    preset_value = None
    for i, (lbl, val) in enumerate(presets):
        with pc[i]:
            if st.button(lbl, key=f"pre_{i}", use_container_width=True):
                preset_value = val

    if 'payment_amount' not in st.session_state:
        st.session_state['payment_amount'] = 15.0 if purpose_key == "streaming" else 1000.0
    if preset_value is not None:
        st.session_state['payment_amount'] = preset_value

    payment_amount = st.number_input(
        f"Kiasi ({cur_display})", min_value=0.01, max_value=10_000_000.0,
        value=float(st.session_state['payment_amount']), step=1.0,
    )
    st.session_state['payment_amount'] = payment_amount

    st.markdown("---")
    if st.button(translate('refresh_button'), use_container_width=True):
        load_data.clear(); load_models.clear(); st.rerun()

    st.markdown("""<p style='font-size:0.73rem;color:#9ca3af;line-height:1.6;margin-top:0.8rem;'>
    ⚠️ Ushauri kulingana na data ya kihistoria. Thibitisha na
    <a href="https://www.bot.go.tz" target="_blank" style="color:#6366f1">BOT</a>
    au benki yako.</p>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# FETCH DATA & RUN FORECAST
# ─────────────────────────────────────────────────────────────────
df           = all_data[cur_display]
current_rate = float(df['mean'].iloc[-1])
prev_rate    = float(df['mean'].iloc[-2])
rate_delta   = current_rate - prev_rate
rate_delta_p = rate_delta / prev_rate * 100
last_date    = df['date'].iloc[-1]
days_old     = (pd.Timestamp.now() - last_date).days

with st.spinner(f"Inachambua {cur_display}/TZS..."):
    ensemble, all_forecasts = get_ensemble(all_models, df, cur_key, steps)

if ensemble is None:
    st.error("❌ Modeli hazikuweza kutoa utabiri."); st.stop()

# NEW: visible diagnostics — shows exactly which models contributed to this
# forecast, and surfaces any model-loading errors captured in load_models().
# This is what used to be completely invisible (silently swallowed
# exceptions), and is the reason predictions could differ between your
# local machine and Render with no error shown anywhere.
with st.expander("🔧 Debug: model status (tap to expand)", expanded=False):
    st.write(f"Models used in this forecast: {list(all_forecasts.keys())}")
    cur_errors = all_models.get('_load_errors', {}).get(cur_key, {})
    if cur_errors:
        st.warning(f"Models that FAILED to load for {cur_display}:")
        for model_name, err in cur_errors.items():
            st.code(f"{model_name}: {err}")
    else:
        st.success("All models loaded successfully for this currency.")

    # NEW: show each model's raw last-day prediction side by side, so you can
    # see if one model is wildly disagreeing with the others (a red flag)
    # instead of that disagreement being hidden inside an averaged ensemble.
    if all_forecasts:
        st.write("Each model's predicted rate for the final day of this forecast:")
        compare_rows = []
        for name, vals in all_forecasts.items():
            pct_change = (vals[-1] - current_rate) / current_rate * 100
            compare_rows.append({
                "Model": name,
                f"Predicted {cur_display}/TZS (day {steps})": round(float(vals[-1]), 2),
                "% change vs today": round(float(pct_change), 2)
            })
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
        st.caption("If these numbers disagree wildly with each other, treat the "
                   "ensemble average with caution — it means the models "
                   "aren't actually agreeing, they're just being averaged.")

lower_band, upper_band = get_confidence_band(all_forecasts, ensemble)
advice = get_advice(current_rate, ensemble, payment_amount, cur_display, purpose_key, steps)
bda    = best_day_analysis(df)

future_dates = [df['date'].iloc[-1] + timedelta(days=i+1) for i in range(steps)]

# ─────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────
arrow  = "▲" if rate_delta >= 0 else "▼"
acolor_css = "color:#ef4444" if rate_delta >= 0 else "color:#10b981"

col_hdr, col_pill = st.columns([2, 1])
with col_hdr:
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">{translate('header_title')}</div>
        <p class="hero-sub">{translate('header_sub')}</p>
    </div>
    """, unsafe_allow_html=True)

with col_pill:
    st.markdown(f"""
    <div style="height:100%;display:flex;flex-direction:column;justify-content:center;gap:10px;padding-top:0.2rem">
        <div class="glass-card g-indigo" style="margin-bottom:0">
            <div class="gc-label">Bei ya Sasa · {cur_display}/TZS</div>
            <div class="gc-value">TZS {current_rate:,.2f}</div>
            <div class="gc-note" style="{acolor_css};font-weight:600">{arrow} {abs(rate_delta):.2f} ({abs(rate_delta_p):.2f}%) tangu jana</div>
            <div class="gc-icon">💱</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Data warning
if days_old > 2:
    st.markdown(f"""
    <div class="data-warn-3d">
        📅 Bei ni ya <b>{last_date.strftime('%d/%m/%Y')}</b> (siku {days_old} zilizopita).
        Kwa bei halisi tembelea
        <a href="https://www.bot.go.tz" target="_blank" style="color:#92400e;font-weight:600">www.bot.go.tz</a>.
    </div>""", unsafe_allow_html=True)

# Shimmer divider
st.markdown('<div class="shimmer-line"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([translate('tab_advice'), translate('tab_trend'), translate('tab_analysis')])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — USHAURI
# ══════════════════════════════════════════════════════════════════
with tab1:

    # ── Row 1: 4 glass metric chips ─────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="glass-card g-indigo">
            <div class="gc-label">Bei ya Wastani</div>
            <div class="gc-value">TZS {current_rate:,.0f}</div>
            <div class="gc-note" style="{acolor_css};font-weight:600">{arrow} {abs(rate_delta_p):.2f}% leo</div>
            <div class="gc-icon">📊</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="glass-card g-emerald">
            <div class="gc-label">Bei ya Kununua</div>
            <div class="gc-value">TZS {df['buying'].iloc[-1]:,.0f}</div>
            <div class="gc-note">Benki inanunua {cur_display}</div>
            <div class="gc-icon">🏦</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="glass-card g-violet">
            <div class="gc-label">Bei ya Kuuza</div>
            <div class="gc-value">TZS {df['selling'].iloc[-1]:,.0f}</div>
            <div class="gc-note">Benki inakuuzia {cur_display}</div>
            <div class="gc-icon">💰</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        urg_color_map = {"urg-high":"g-rose","urg-mid":"g-amber","urg-low":"g-emerald"}
        st.markdown(f"""
        <div class="glass-card {urg_color_map.get(advice['urgency_class'],'g-sky')}">
            <div class="gc-label">Hatari ya Dharura</div>
            <div class="gc-value">{advice['urgency_level']}</div>
            <div class="gc-note">{advice['urgency_note']}</div>
            <div class="gc-icon">⚠️</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Recommendation + Payment ─────────────────────────
    col_rec, col_pay = st.columns([1.1, 1], gap="large")

    with col_rec:
        st.markdown('<div class="eyebrow">Ushauri wa Leo</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rec-3d {advice['rec_style']}">
            <div class="rec-3d-action">{advice['icon']} {advice['action']}</div>
            <div class="rec-3d-headline">{advice['headline']}</div>
            <div class="rec-3d-purpose">{advice['purpose_text']}</div>
        </div>
        <span class="urg-chip {advice['urgency_class']}">
            ● Hatari: {advice['urgency_level']} — {advice['urgency_note']}
        </span>
        """, unsafe_allow_html=True)

    with col_pay:
        st.markdown('<div class="eyebrow">Hesabu ya Fedha Yako</div>', unsafe_allow_html=True)

        future_class = (
            "pay-future-okoa"   if advice['saving_direction'] == 'okoa'
            else "pay-future-hasara" if advice['saving_direction'] == 'hasara'
            else "pay-future-stable"
        )
        st.markdown(f"""
        <div class="pay-grid">
            <div class="pay-3d pay-today">
                <div class="pb-label">Ukilipa LEO</div>
                <div class="pb-val">TZS {advice['tzs_now']:,.0f}</div>
                <div class="pb-note">{cur_display} {payment_amount:,.2f} × {current_rate:,.2f}</div>
            </div>
            <div class="pay-3d {future_class}">
                <div class="pb-label">Baada ya Siku {steps}</div>
                <div class="pb-val">TZS {advice['tzs_later']:,.0f}</div>
                <div class="pb-note">Tabiri: TZS {advice['avg_future']:,.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if advice['saving_direction'] == 'okoa':
            st.markdown(f"""
            <div class="savings-banner sb-okoa">
                <div><div class="sb-label">💡 Ukifuata ushauri, utaokoa:</div>
                     <div class="sb-sub">Tofauti kati ya leo na baada ya siku {steps}</div></div>
                <div class="sb-amount">TZS {advice['saving_amount']:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        elif advice['saving_direction'] == 'hasara':
            st.markdown(f"""
            <div class="savings-banner sb-hasara">
                <div><div class="sb-label">⚠️ Ukisubiri badala ya kulipa sasa, utapoteza:</div>
                     <div class="sb-sub">Bei itapanda — lipa sasa kulinda fedha yako</div></div>
                <div class="sb-amount">TZS {advice['saving_amount']:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="savings-banner sb-stable">
                <div><div class="sb-label">📊 Tofauti ya kulipa leo vs siku {steps}:</div>
                     <div class="sb-sub">Bei iko stable — hakuna haraka</div></div>
                <div class="sb-amount">TZS {abs(advice['tzs_now']-advice['tzs_later']):,.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Best-day insight cards ───────────────────────────
    st.markdown('<div class="eyebrow">📅 Siku Bora ya Kulipa — Uchambuzi wa Kihistoria</div>', unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown(f"""
        <div class="insight-3d ins-purple">
            <div class="ins-label">Siku Bora ya Wiki</div>
            <div class="ins-value">🗓 {bda['best_weekday']}</div>
            <div class="ins-note">Bei ya wastani iko chini zaidi siku hii</div>
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown(f"""
        <div class="insight-3d ins-indigo">
            <div class="ins-label">Wiki Bora ya Mwezi</div>
            <div class="ins-value">📆 {bda['best_wom']}</div>
            <div class="ins-note">Bei iko chini zaidi katika wiki hii ya mwezi</div>
        </div>""", unsafe_allow_html=True)
    with ic3:
        st.markdown(f"""
        <div class="insight-3d ins-teal">
            <div class="ins-label">Akiba ya Wastani kwa Wiki</div>
            <div class="ins-value">💰 TZS {bda['avg_saving_wd']:,.0f}</div>
            <div class="ins-note">Tofauti ya bei kati ya siku bora na mbaya zaidi</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — MWENENDO WA BEI
# ══════════════════════════════════════════════════════════════════
with tab2:

    hist = df.tail(120)

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(upper_band) + list(lower_band[::-1]),
        fill='toself',
        fillcolor='rgba(99,102,241,0.10)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True, name='Upeo wa Utabiri (±)',
        hoverinfo='skip'
    ))

    # Historical
    fig.add_trace(go.Scatter(
        x=hist['date'], y=hist['mean'],
        name='Bei ya Kihistoria',
        line=dict(color='#4f46e5', width=2.5),
        mode='lines',
        hovertemplate='%{x|%d/%m/%Y}<br>TZS %{y:,.2f}<extra></extra>'
    ))

    # MA_30
    if 'MA_30' in hist.columns:
        fig.add_trace(go.Scatter(
            x=hist['date'], y=hist['MA_30'],
            name='Wastani wa Siku 30',
            line=dict(color='#f59e0b', width=1.8, dash='dot'),
            mode='lines',
            hovertemplate='MA30: TZS %{y:,.2f}<extra></extra>'
        ))

    # Ensemble forecast
    fig.add_trace(go.Scatter(
        x=future_dates, y=ensemble,
        name=f'Utabiri (Siku {steps})',
        line=dict(color='#10b981', width=3),
        mode='lines+markers',
        marker=dict(size=6, color='#10b981',
                    line=dict(color='white', width=2)),
        hovertemplate='%{x|%d/%m/%Y}<br>Tabiri: TZS %{y:,.2f}<extra></extra>'
    ))

    fig.add_vrect(
        x0=df['date'].iloc[-1], x1=future_dates[-1],
        fillcolor="rgba(99,102,241,0.04)", layer="below", line_width=0,
        annotation_text="Eneo la utabiri", annotation_position="top left",
        annotation_font_size=11, annotation_font_color="#6366f1"
    )

    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='rgba(99,102,241,0.20)', borderwidth=1),
        xaxis=dict(title='Tarehe', showgrid=True,
                   gridcolor='rgba(99,102,241,0.08)', zeroline=False,
                   tickfont=dict(size=11)),
        yaxis=dict(title=f'TZS kwa 1 {cur_display}', showgrid=True,
                   gridcolor='rgba(99,102,241,0.08)', zeroline=False,
                   tickfont=dict(size=11)),
        hovermode='x unified',
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color='#374151'),
    )

    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Bei Inayotarajiwa — Kila Siku</div>', unsafe_allow_html=True)

    rows = []
    for i, (d, v) in enumerate(zip(future_dates, ensemble)):
        change  = v - current_rate
        chg_p   = change / current_rate * 100
        lo, hi  = lower_band[i], upper_band[i]
        rows.append({
            "Tarehe":                          d.strftime('%d/%m/%Y (%a)'),
            f"Bei ({cur_display}/TZS)":        f"TZS {v:,.2f}",
            "Upeo wa Chini":                   f"TZS {lo:,.2f}",
            "Upeo wa Juu":                     f"TZS {hi:,.2f}",
            "Mabadiliko":                      f"{'▲' if change>=0 else '▼'} {abs(change):.2f} ({chg_p:+.2f}%)",
            f"Malipo ({payment_amount:.0f} {cur_display})": f"TZS {payment_amount*v:,.0f}",
        })

    forecast_df = pd.DataFrame(rows)
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(translate('download_csv'), csv,
                       file_name=f"{cur_key}_forecast_{steps}days.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — UCHAMBUZI WA KINA
# ══════════════════════════════════════════════════════════════════
with tab3:

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="eyebrow">🤖 Modeli Zilizotumika</div>', unsafe_allow_html=True)

        model_info = {
            'XGBoost': {'mae':{'usd':3.63,'eur':10.79,'cny':1.49},
                        'mape':{'usd':0.14,'eur':0.36,'cny':0.40},
                        'desc':'Inajifunza patterns kutoka vipengele vingi — mwelekeo wa wiki na mwezi'},
            'LSTM':    {'mae':{'usd':5.33,'eur':22.69,'cny':1.12},
                        'mape':{'usd':0.21,'eur':0.76,'cny':0.30},
                        'desc':'Ina "kumbukumbu" ya muda mrefu — inajifunza mifuatano ya bei'},
        }
        for mname, mdata in model_info.items():
            mae  = mdata['mae'].get(cur_key, 0)
            mape = mdata['mape'].get(cur_key, 0)
            st.markdown(f"""
            <div class="model-3d">
                <div class="mc-left">
                    <div class="mn">{mname}</div>
                    <div class="md">{mdata['desc']}</div>
                </div>
                <div class="mr">
                    <div class="mel">Kosa la wastani</div>
                    <div class="mev">TZS {mae:.2f} ({mape:.2f}%)</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">ℹ️ Jinsi Inavyofanya Kazi</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer-3d">
            Mfumo unatumia modeli 2 bora (<strong>XGBoost + LSTM</strong>) kuchambua
            data ya miaka 3. Kila modeli inatabiri bei ya siku zijazo — wastani wake
            unatoa ushauri. Upeo wa utabiri (eneo la bluu kwenye grafu) unaonyesha
            pengo la kutokuwa na uhakika. Ushauri ni kwa mtu anayetaka
            <strong>KULIPA</strong> kwa sarafu ya kigeni.
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown(f'<div class="eyebrow">{translate("compare_title")}</div>', unsafe_allow_html=True)

        comparison_rows = []
        for label, (display, key) in CURRENCIES.items():
            od   = all_data[display]
            cv   = float(od['mean'].iloc[-1])
            pv   = float(od['mean'].iloc[-2])
            chg  = cv - pv
            chgp = chg / pv * 100
            comparison_rows.append({"display":display,"raw":cv,
                "fmt":f"{cv:,.2f}","chg":f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f} ({chgp:+.2f}%)"})

        best = min(comparison_rows, key=lambda x: x['raw'])
        cards_html = '<div class="cur-cards">'
        for row in comparison_rows:
            is_best = row['display'] == best['display']
            badge_html = ''
            if is_best:
                badge_html = '<div style="font-size:0.72rem;font-weight:700;color:#059669;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.5rem">✅ Nafuu Zaidi</div>'

            card_html = '<div class="cur-card' + (' best-cur' if is_best else '') + '">'
            card_html += badge_html
            card_html += '<div class="cur-row"><span class="cur-lbl">Sarafu</span><span class="cur-val">' + row['display'] + '</span></div>'
            card_html += '<div class="cur-row"><span class="cur-lbl">Bei (TZS)</span><span class="cur-val">' + row['fmt'] + '</span></div>'
            card_html += '<div class="cur-row"><span class="cur-lbl">Mabadiliko</span><span class="cur-val">' + row['chg'] + '</span></div>'
            card_html += '</div>'

            cards_html += card_html
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        if len(all_forecasts) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">📊 Ulinganisho wa Modeli</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            cmap = {'XGBoost':'#4f46e5','LSTM':'#10b981','ARIMA':'#ef4444','Prophet':'#8b5cf6'}
            for mname, preds in all_forecasts.items():
                fig2.add_trace(go.Scatter(
                    x=future_dates, y=preds, name=mname,
                    line=dict(color=cmap.get(mname,'#888'), width=2),
                    mode='lines',
                    hovertemplate=f'{mname}: TZS %{{y:,.2f}}<extra></extra>'
                ))
            fig2.add_trace(go.Scatter(
                x=future_dates, y=ensemble, name='Ensemble',
                line=dict(color='#f59e0b', width=3, dash='dash'),
                hovertemplate='Ensemble: TZS %{y:,.2f}<extra></extra>'
            ))
            fig2.update_layout(
                height=270,
                margin=dict(l=0,r=0,t=20,b=0),
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            bgcolor='rgba(255,255,255,0.85)'),
                xaxis=dict(showgrid=True, gridcolor='rgba(99,102,241,0.08)', zeroline=False),
                yaxis=dict(title='TZS', showgrid=True, gridcolor='rgba(99,102,241,0.08)', zeroline=False),
                plot_bgcolor='rgba(255,255,255,0.95)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans, sans-serif', color='#374151'),
            )
            st.plotly_chart(fig2, use_container_width=True)