# ==============================================================================
# NephroScan AI™ — Enterprise Clinical Renal CT Diagnostic Workstation
# Advanced Deep CNN Classification, Explainable AI (Grad-CAM), 
# Neural GAN Super-Resolution & Automated Lesion Morphometry Engine
# ==============================================================================

import os
import json
import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import altair as alt
import streamlit as st
import tensorflow as tf

import gan_module

# ------------------------------------------------------------------------------
# 1. Page Configuration & Professional Clinical Theme
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="NephroScan AI™ | Enterprise Renal Diagnostic Workstation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Medical Dashboard CSS — Cinematic Edition
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

    /* ================================================================
       KEYFRAME ANIMATIONS — Cinematic Micro-Interactions
       ================================================================ */

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(56, 189, 248, 0.15), 0 0 60px rgba(56, 189, 248, 0.05); }
        50% { box-shadow: 0 0 30px rgba(56, 189, 248, 0.3), 0 0 80px rgba(129, 140, 248, 0.12); }
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.92); }
        to { opacity: 1; transform: scale(1); }
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes borderGlow {
        0%, 100% { border-color: rgba(56, 189, 248, 0.3); }
        50% { border-color: rgba(56, 189, 248, 0.7); }
    }

    @keyframes neonPulse {
        0%, 100% { box-shadow: 0 0 5px var(--neon-color, #38bdf8), 0 0 15px var(--neon-color, #38bdf8), inset 0 0 10px rgba(0,0,0,0.3); }
        50% { box-shadow: 0 0 10px var(--neon-color, #38bdf8), 0 0 30px var(--neon-color, #38bdf8), 0 0 50px var(--neon-color, #38bdf8), inset 0 0 10px rgba(0,0,0,0.3); }
    }

    @keyframes floatParticle {
        0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.3; }
        25% { transform: translateY(-15px) translateX(8px); opacity: 0.7; }
        50% { transform: translateY(-25px) translateX(-5px); opacity: 0.5; }
        75% { transform: translateY(-12px) translateX(12px); opacity: 0.8; }
    }

    @keyframes scanline {
        0% { top: -5%; }
        100% { top: 105%; }
    }

    @keyframes spinGauge {
        from { stroke-dashoffset: 283; }
    }

    @keyframes breathe {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes typewriter {
        from { width: 0; }
        to { width: 100%; }
    }

    @keyframes badgePing {
        0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.5); }
        70% { box-shadow: 0 0 0 8px rgba(56, 189, 248, 0); }
        100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
    }

    @keyframes countPop {
        0% { transform: scale(0.5); opacity: 0; }
        60% { transform: scale(1.12); }
        100% { transform: scale(1); opacity: 1; }
    }

    @keyframes glowLine {
        0% { left: -40%; }
        100% { left: 140%; }
    }

    /* ================================================================
       CUSTOM SCROLLBAR — Dark Medical Aesthetic
       ================================================================ */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0a0e17; border-radius: 4px; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #1e3a5f, #38bdf8);
        border-radius: 4px;
        border: 1px solid #0f172a;
    }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #2563eb, #60a5fa); }

    /* ================================================================
       GLOBAL THEME — Premium Foundation
       ================================================================ */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        scroll-behavior: smooth;
    }

    .stApp {
        background: radial-gradient(ellipse at 20% 50%, rgba(15, 23, 42, 1) 0%, rgba(3, 7, 18, 1) 70%) !important;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ================================================================
       PACS HEADER — Animated Gradient with Particles
       ================================================================ */
    .pacs-header {
        background: linear-gradient(135deg, #090d16 0%, #111827 40%, #0c1a2e 70%, #0f172a 100%);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 18px;
        padding: 28px 34px;
        margin-bottom: 24px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), 0 0 60px rgba(56, 189, 248, 0.06);
        position: relative;
        overflow: hidden;
        animation: pulseGlow 4s ease-in-out infinite, gradientShift 8s ease infinite;
    }

    .pacs-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #38bdf8, #818cf8, #c084fc, #38bdf8, transparent);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }

    .pacs-header::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 10%, rgba(56, 189, 248, 0.3) 50%, transparent 90%);
    }

    /* Floating Particles Container */
    .particles-container {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none;
        overflow: hidden;
        z-index: 0;
    }

    .particle {
        position: absolute;
        width: 3px;
        height: 3px;
        border-radius: 50%;
        background: rgba(56, 189, 248, 0.5);
        animation: floatParticle 6s ease-in-out infinite;
    }

    .particle:nth-child(1) { left: 10%; top: 20%; animation-delay: 0s; animation-duration: 7s; }
    .particle:nth-child(2) { left: 25%; top: 60%; animation-delay: 1s; animation-duration: 5s; background: rgba(129, 140, 248, 0.4); }
    .particle:nth-child(3) { left: 45%; top: 30%; animation-delay: 2s; animation-duration: 8s; }
    .particle:nth-child(4) { left: 65%; top: 70%; animation-delay: 0.5s; animation-duration: 6s; background: rgba(192, 132, 252, 0.4); }
    .particle:nth-child(5) { left: 80%; top: 40%; animation-delay: 3s; animation-duration: 7s; }
    .particle:nth-child(6) { left: 90%; top: 15%; animation-delay: 1.5s; animation-duration: 9s; background: rgba(56, 189, 248, 0.3); }
    .particle:nth-child(7) { left: 35%; top: 80%; animation-delay: 4s; animation-duration: 6s; }
    .particle:nth-child(8) { left: 55%; top: 10%; animation-delay: 2.5s; animation-duration: 8s; background: rgba(129, 140, 248, 0.3); }

    /* ================================================================
       STATUS BADGES — Animated Pulse
       ================================================================ */
    .pacs-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 5px 12px;
        border-radius: 8px;
        background: rgba(14, 165, 233, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-right: 6px;
        margin-bottom: 4px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInScale 0.6s ease-out backwards;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    .pacs-badge:nth-child(1) { animation-delay: 0.1s; }
    .pacs-badge:nth-child(2) { animation-delay: 0.2s; }
    .pacs-badge:nth-child(3) { animation-delay: 0.3s; }
    .pacs-badge:nth-child(4) { animation-delay: 0.4s; }

    .pacs-badge:hover {
        background: rgba(14, 165, 233, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.25);
    }

    .badge-live::before {
        content: '';
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10b981;
        animation: breathe 2s ease-in-out infinite;
        margin-right: 4px;
    }

    /* ================================================================
       TELEMETRY STRIP — Glass Effect
       ================================================================ */
    .telemetry-strip {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 14px;
        padding: 12px 20px;
        margin-bottom: 22px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 14px;
        font-size: 0.82rem;
        color: #9ca3af;
        animation: fadeInUp 0.5s ease-out backwards;
        animation-delay: 0.2s;
        position: relative;
        overflow: hidden;
    }

    .telemetry-strip::after {
        content: '';
        position: absolute;
        top: 0; left: -40%; width: 30%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.04), transparent);
        animation: glowLine 5s linear infinite;
    }

    .telemetry-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .telemetry-label {
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
    }

    .telemetry-val {
        font-weight: 600;
        color: #f3f4f6;
    }

    /* ================================================================
       DIAGNOSTIC HERO CARDS — Glassmorphism + Entrance Animations
       ================================================================ */
    .diag-card {
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        border: 1px solid;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) backwards;
        animation-delay: 0.3s;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .diag-card:hover {
        transform: translateY(-3px);
    }

    .diag-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--card-accent, #38bdf8), transparent);
        opacity: 0.8;
    }

    .diag-card::after {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.02), transparent 60%);
        pointer-events: none;
    }

    .card-stone {
        --card-accent: #f59e0b;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(17, 24, 39, 0.85) 60%, rgba(15, 23, 42, 0.95) 100%);
        border-color: rgba(245, 158, 11, 0.3);
        box-shadow: 0 8px 32px rgba(245, 158, 11, 0.1), 0 0 60px rgba(245, 158, 11, 0.04);
    }
    .card-stone:hover { box-shadow: 0 12px 40px rgba(245, 158, 11, 0.18), 0 0 80px rgba(245, 158, 11, 0.06); }

    .card-normal {
        --card-accent: #10b981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(17, 24, 39, 0.85) 60%, rgba(15, 23, 42, 0.95) 100%);
        border-color: rgba(16, 185, 129, 0.3);
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.1), 0 0 60px rgba(16, 185, 129, 0.04);
    }
    .card-normal:hover { box-shadow: 0 12px 40px rgba(16, 185, 129, 0.18), 0 0 80px rgba(16, 185, 129, 0.06); }

    .card-tumor {
        --card-accent: #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(17, 24, 39, 0.85) 60%, rgba(15, 23, 42, 0.95) 100%);
        border-color: rgba(239, 68, 68, 0.35);
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.12), 0 0 60px rgba(239, 68, 68, 0.05);
    }
    .card-tumor:hover { box-shadow: 0 12px 40px rgba(239, 68, 68, 0.22), 0 0 80px rgba(239, 68, 68, 0.08); }

    /* ================================================================
       METRIC GRID — Animated Cards with Glow
       ================================================================ */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
        gap: 16px;
        margin-bottom: 22px;
    }

    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(31, 41, 55, 0.8);
        border-radius: 14px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
    }

    .metric-card:nth-child(1) { animation-delay: 0.1s; }
    .metric-card:nth-child(2) { animation-delay: 0.2s; }
    .metric-card:nth-child(3) { animation-delay: 0.3s; }
    .metric-card:nth-child(4) { animation-delay: 0.4s; }
    .metric-card:nth-child(5) { animation-delay: 0.5s; }

    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3), 0 0 20px rgba(56, 189, 248, 0.08);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }

    .metric-card-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9ca3af;
        margin-bottom: 8px;
        font-family: 'Inter', sans-serif;
    }

    .metric-card-val {
        font-size: 1.6rem;
        font-weight: 800;
        line-height: 1.2;
        animation: countPop 0.5s cubic-bezier(0.16, 1, 0.3, 1) backwards;
        font-family: 'Outfit', sans-serif;
    }

    /* ================================================================
       VIEWPORT BOXES — Neon Glow + Scanline
       ================================================================ */
    .image-viewport-box {
        background: rgba(9, 13, 22, 0.9);
        border: 1px solid rgba(31, 41, 55, 0.6);
        border-radius: 14px;
        padding: 14px;
        text-align: center;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInScale 0.5s ease-out backwards;
    }

    .image-viewport-box:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.1);
    }

    .viewport-neon-cyan { --neon-color: #38bdf8; animation: fadeInScale 0.5s ease-out backwards, neonPulse 3s ease-in-out infinite; border-color: rgba(56, 189, 248, 0.4); }
    .viewport-neon-green { --neon-color: #10b981; animation: fadeInScale 0.5s ease-out backwards, neonPulse 3s ease-in-out infinite; border-color: rgba(16, 185, 129, 0.4); animation-delay: 0.5s; }
    .viewport-neon-purple { --neon-color: #a78bfa; animation: fadeInScale 0.5s ease-out backwards, neonPulse 3s ease-in-out infinite; border-color: rgba(167, 139, 250, 0.4); animation-delay: 1s; }
    .viewport-neon-red { --neon-color: #00f0ff; animation: fadeInScale 0.5s ease-out backwards, neonPulse 3s ease-in-out infinite; border-color: rgba(0, 240, 255, 0.4); animation-delay: 1.5s; }

    .scanline-effect {
        position: absolute;
        top: -5%; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.5), transparent);
        animation: scanline 3s linear infinite;
        pointer-events: none;
        z-index: 2;
    }

    /* ================================================================
       PROBABILITY DISPLAY — Shimmer Bars
       ================================================================ */
    .prob-container {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(31, 41, 55, 0.6);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        transition: all 0.3s ease;
        animation: slideInLeft 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
    }

    .prob-container:nth-child(1) { animation-delay: 0.1s; }
    .prob-container:nth-child(2) { animation-delay: 0.25s; }
    .prob-container:nth-child(3) { animation-delay: 0.4s; }

    .prob-container:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateX(4px);
    }

    .prob-bar-track {
        background: rgba(31, 41, 55, 0.8);
        height: 10px;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 8px;
        position: relative;
    }

    .prob-bar-thumb {
        height: 100%;
        border-radius: 5px;
        transition: width 1.2s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }

    .prob-bar-thumb::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
        background-size: 200% 100%;
        animation: shimmer 2s linear infinite;
    }

    /* ================================================================
       CONFIDENCE GAUGE — Animated SVG Circle
       ================================================================ */
    .gauge-container {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 140px;
        height: 140px;
        margin: 8px auto;
    }

    .gauge-svg {
        width: 140px;
        height: 140px;
        transform: rotate(-90deg);
        filter: drop-shadow(0 0 10px var(--gauge-color, #38bdf8));
    }

    .gauge-bg {
        fill: none;
        stroke: rgba(31, 41, 55, 0.7);
        stroke-width: 7;
    }

    .gauge-fill {
        fill: none;
        stroke: var(--gauge-color, #38bdf8);
        stroke-width: 7;
        stroke-linecap: round;
        transition: stroke-dashoffset 1.5s cubic-bezier(0.16, 1, 0.3, 1);
        animation: spinGauge 1.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    .gauge-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        pointer-events: none;
    }

    .gauge-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    .gauge-label {
        font-size: 0.62rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #94a3b8;
        margin-top: 3px;
    }


    /* ================================================================
       PROTOCOL STEPS — Animated Entrance
       ================================================================ */
    .protocol-step {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-left: 3px solid #38bdf8;
        border-radius: 0 12px 12px 0;
        padding: 14px 20px;
        margin-bottom: 12px;
        font-size: 0.92rem;
        color: #e5e7eb;
        transition: all 0.3s ease;
        animation: slideInLeft 0.5s ease-out backwards;
    }

    .protocol-step:hover {
        border-left-color: #818cf8;
        background: rgba(17, 24, 39, 0.8);
        transform: translateX(6px);
        box-shadow: -3px 0 15px rgba(56, 189, 248, 0.1);
    }

    /* ================================================================
       DISCLAIMER BANNER — Subtle Glass
       ================================================================ */
    .disclaimer-banner {
        background: rgba(17, 24, 39, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(31, 41, 55, 0.5);
        border-radius: 14px;
        padding: 20px 26px;
        margin-top: 40px;
        font-size: 0.8rem;
        color: #9ca3af;
        line-height: 1.6;
        animation: fadeInUp 0.6s ease-out backwards;
        animation-delay: 0.8s;
    }

    /* ================================================================
       GAN CALLOUT BOX — Premium Glass
       ================================================================ */
    .gan-callout {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.06) 0%, rgba(129, 140, 248, 0.06) 50%, rgba(192, 132, 252, 0.04) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out backwards;
    }

    .gan-callout::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        background-size: 200% 100%;
        animation: shimmer 4s linear infinite;
    }

    /* ================================================================
       SECTION DIVIDERS — Animated Gradient
       ================================================================ */
    .animated-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #38bdf8, #818cf8, #c084fc, transparent);
        background-size: 200% 100%;
        animation: shimmer 4s linear infinite;
        border: none;
        margin: 28px 0;
        border-radius: 1px;
    }

    /* ================================================================
       LOADING SPINNER — DNA Helix Theme
       ================================================================ */
    .medical-loader {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        padding: 24px;
    }

    .helix-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #38bdf8;
        animation: floatParticle 1.2s ease-in-out infinite;
    }

    .helix-dot:nth-child(1) { animation-delay: 0s; background: #38bdf8; }
    .helix-dot:nth-child(2) { animation-delay: 0.15s; background: #818cf8; }
    .helix-dot:nth-child(3) { animation-delay: 0.3s; background: #c084fc; }
    .helix-dot:nth-child(4) { animation-delay: 0.45s; background: #38bdf8; }
    .helix-dot:nth-child(5) { animation-delay: 0.6s; background: #818cf8; }

    /* ================================================================
       TABS — Smooth Transitions
       ================================================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(17, 24, 39, 0.5);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s ease;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(56, 189, 248, 0.1);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
    }

    /* ================================================================
       BUTTONS — Enhanced Hover Effects
       ================================================================ */
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border-radius: 10px !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.25) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ================================================================
       SIDEBAR — Premium Dark Glass
       ================================================================ */
    section[data-testid="stSidebar"] {
        background: rgba(9, 13, 22, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(31, 41, 55, 0.5) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        font-family: 'Outfit', sans-serif !important;
    }

    /* ================================================================
       CUSTOM SCROLLBAR — Sleek Medical Dark Theme
       ================================================================ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #090d16;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #38bdf8;
    }

    /* ================================================================
       ARCHITECTURE FLOW PIPELINE — Neural Nodes
       ================================================================ */
    .arch-flow-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 24px;
        overflow-x: auto;
    }

    .arch-node {
        flex: 1;
        min-width: 130px;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 12px 14px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }

    .arch-node:hover {
        border-color: #38bdf8;
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.2);
    }

    .arch-node-title {
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #38bdf8;
        margin-bottom: 4px;
    }

    .arch-node-desc {
        font-size: 0.82rem;
        font-weight: 600;
        color: #f1f5f9;
    }

    .arch-node-sub {
        font-size: 0.68rem;
        color: #94a3b8;
        margin-top: 3px;
        font-family: 'JetBrains Mono', monospace;
    }

    .arch-arrow {
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 800;
    }

    /* ================================================================
       TIMELINE & AUDIT CARDS
       ================================================================ */
    .timeline-container {
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding: 10px 0 16px 0;
    }

    .timeline-card {
        flex: 0 0 auto;
        width: 220px;
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 14px;
        padding: 14px;
        transition: all 0.3s ease;
    }

    .timeline-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. System Paths & Knowledge Metadata
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "kidney_model.keras")
METADATA_PATH = os.path.join(BASE_DIR, "model", "class_indices.json")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_images")

DEFAULT_CLASS_META = {
    "0": {
        "folder_name": "Cyst",
        "display_name": "Cystine Kidney Stone",
        "sub_title": "Cystine Calculus / Renal Cystic Lesion",
        "category": "Cystine Stone / Cyst",
        "severity": "Moderate Priority",
        "urgency": "Outpatient Urological Consultation",
        "color": "#f59e0b",
        "theme_class": "card-stone",
        "badge": "⚠️ Pathological Finding (Calculus)",
        "clinical_summary": "The deep CNN identified focal radiological attenuation characteristic of Cystine Nephrolithiasis (kidney calculus) or well-circumscribed renal cystic changes.",
        "description": "Cystine stones are a metabolic form of urolithiasis caused by cystinuria, an autosomal recessive renal tubular defect resulting in impaired cystine reabsorption and spontaneous crystallization in acidic urine (pH < 6.5). On non-contrast CT, cystine stones exhibit characteristic homogeneous ground-glass attenuation (approx. 500–700 Hounsfield Units).",
        "etiology": "Mutations in SLC3A1 (Type A) or SLC7A9 (Type B) genes, reducing dibasic amino acid (cystine, ornithine, lysine, arginine) resorption in proximal renal tubules.",
        "treatments": [
            "💧 Aggressive Hydration Therapy: Target 24-hr urine volume > 3.0–3.5 Liters (specific gravity < 1.010). Consume fluids evenly distributed across 24h, including during nighttime awakenings.",
            "🧪 Urinary Alkalinization: Potassium Citrate (60–80 mEq/day in divided doses) titrated to maintain urinary pH strictly between 7.0 and 7.5. Avoid sodium bicarbonate.",
            "🧂 Dietary Sodium & Protein Restriction: Limit sodium to < 2,000 mg/day (reduces cystine excretion) and moderate animal protein consumption to decrease methionine precursor intake.",
            "💊 Thiol-Binding Chelators: Second-line intervention for refractory stones (Alpha-mercaptopropionylglycine / Tiopronin, or D-Penicillamine with Pyridoxine B6).",
            "⚡ Endourological Intervention: Extracorporeal Shock Wave Lithotripsy (ESWL) for stones <10 mm, Retrograde Intrarenal Surgery (RIRS / Ureteroscopy) with Holmium:YAG / Thulium laser, or Percutaneous Nephrolithotomy (PCNL) for large staghorn calculi."
        ]
    },
    "1": {
        "folder_name": "Normal",
        "display_name": "Normal (Healthy)",
        "sub_title": "Unremarkable Bilateral Renal Parenchyma",
        "category": "Normal Anatomy",
        "severity": "Low Priority",
        "urgency": "Routine Preventive Follow-up",
        "color": "#10b981",
        "theme_class": "card-normal",
        "badge": "✅ Normal Renal Architecture",
        "clinical_summary": "Normal renal parenchymal thickness, preserved corticomedullary differentiation, symmetrical bilateral renal contour, and absent calculus or mass lesions.",
        "description": "Normal non-contrast abdominal CT reveals homogenous renal parenchyma with standard attenuation values (approx. 30–40 Hounsfield Units). The renal collecting system and ureters are non-dilated with absence of hydronephrosis, perinephric stranding, or hyperdense calcifications.",
        "etiology": "Physiological baseline renal parenchyma without acute obstruction, structural malformations, oncological growth, or metabolic stone formation.",
        "treatments": [
            "💧 Preventative Hydration: Maintain standard fluid intake of 2.0 to 2.5 L/day to ensure adequate urinary dilution and solute clearance.",
            "🥗 Balanced Renal Protective Diet: Moderate dietary sodium (<2,300 mg/day), maintain balanced calcium intake from natural food sources, and avoid excessive refined sugars.",
            "🩺 Routine Primary Care Screening: Annual blood pressure monitoring, urinalysis (absence of protein/blood), and baseline serum creatinine / eGFR check.",
            "🏃 Physical Fitness & Weight Optimization: Regular cardiovascular exercise and weight control to mitigate long-term hypertensive and diabetic nephrosclerosis risk."
        ]
    },
    "2": {
        "folder_name": "Tumor",
        "display_name": "Renal Tumor (Solid Mass)",
        "sub_title": "Suspicious Renal Neoplasm / Solid Mass",
        "category": "Neoplastic Lesion",
        "severity": "Critical / Urgent",
        "urgency": "Urgent Urologic Oncology Referral",
        "color": "#ef4444",
        "theme_class": "card-tumor",
        "badge": "🚨 Critical Finding (Renal Neoplasm)",
        "clinical_summary": "The model detected focal structural heterogeneity, contour distortion, and anomalous parenchymal attenuation indicative of a solid renal mass.",
        "description": "Renal neoplasms encompass Renal Cell Carcinoma (Clear Cell RCC, Papillary RCC, Chromophobe RCC) or benign entities (Oncocytoma, Angiomyolipoma). Cross-sectional multiphasic contrast-enhanced imaging is required to assess enhancement, vascularity, and staging.",
        "etiology": "Somatic or hereditary genetic alterations (VHL, MET, TSC1/2 mutations) with known risk factors including smoking, hypertension, obesity, and toxic exposures.",
        "treatments": [
            "🏥 Urgent Urologic Oncology Referral: Comprehensive clinical staging, physical exam, and multidisciplinary tumor board review.",
            "🔬 Quad-Phase Contrast-Enhanced CT / Renal MRI: Evaluate enhancement kinetics (>15-20 HU rise), capsular penetration, and renal vein/IVC involvement.",
            "🔪 Nephron-Sparing Surgery (Partial Nephrectomy): Gold-standard resection for localized T1a/T1b masses (<7 cm) preserving functional renal parenchyma.",
            "💉 Radical Nephrectomy: Complete surgical excision of kidney, adrenal gland, and Gerota's fascia for large, centrally invasive, or high-stage tumors.",
            "⚡ Image-Guided Thermal Ablation: Percutaneous cryoablation or microwave ablation for select small exophytic tumors in high-surgical-risk patients."
        ]
    }
}

@st.cache_resource
def load_model_and_meta():
    loaded_model = None
    if os.path.exists(MODEL_PATH):
        try:
            loaded_model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            
    meta = DEFAULT_CLASS_META
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                loaded = json.load(f)
                if "classes" in loaded:
                    for k, v in loaded["classes"].items():
                        if k in meta:
                            meta[k].update(v)
        except Exception:
            pass
    return loaded_model, meta

model, CLASS_META = load_model_and_meta()

# ------------------------------------------------------------------------------
# 3. Image Preprocessing, Radiology Filters & Grad-CAM (Explainable AI)
# ------------------------------------------------------------------------------
def preprocess_for_cnn(pil_img):
    resized = pil_img.resize((150, 150), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    normalized = arr / 255.0
    return np.expand_dims(normalized, axis=0), resized

def apply_ct_windowing(pil_img, contrast_val=1.4, edge_mode=False):
    enhanced = pil_img.copy()
    if contrast_val != 1.0:
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(contrast_val)
    if edge_mode:
        enhanced = enhanced.filter(ImageFilter.FIND_EDGES)
    return enhanced

def compute_gradcam(img_tensor, base_model):
    try:
        sub_conv = tf.keras.Sequential(base_model.layers[:5])
        sub_dense = tf.keras.Sequential(base_model.layers[5:-1])
        last_dense = base_model.layers[-1]
        
        with tf.GradientTape() as tape:
            conv_outputs = sub_conv(img_tensor)
            tape.watch(conv_outputs)
            dense_feats = sub_dense(conv_outputs)
            # Compute un-saturated pre-softmax logits
            logits = tf.matmul(dense_feats, last_dense.kernel) + last_dense.bias
            top_index = tf.argmax(logits[0])
            loss = logits[:, top_index]
            
        grads = tape.gradient(loss, conv_outputs)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_outputs[0]
        heatmap = tf.maximum(tf.squeeze(conv_out @ pooled[..., tf.newaxis]), 0)
        
        h_max = tf.reduce_max(heatmap).numpy()
        if h_max > 0:
            heatmap_norm = (heatmap / h_max).numpy()
        else:
            heatmap_norm = heatmap.numpy()
        
        # Pure numpy Jet Colormap implementation
        r = np.clip(1.5 - np.abs(4.0 * heatmap_norm - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * heatmap_norm - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * heatmap_norm - 1.0), 0.0, 1.0)
        heatmap_rgb = (np.dstack([r, g, b]) * 255.0).astype(np.uint8)
        
        heatmap_pil = Image.fromarray(heatmap_rgb).resize((150, 150), Image.Resampling.BILINEAR)
        return heatmap_pil, heatmap_norm
    except Exception:
        return None, None

# ------------------------------------------------------------------------------
# 4. Sidebar: Diagnostic Controls, Patient Vignette & GAN Enhancer
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏥 NEPHROSCAN AI™")
    st.caption("Clinical Decision Support, Neural GAN & Renal CT Workstation")
    st.markdown("---")
    
    st.markdown("#### 📁 Image Input Source")
    input_source = st.radio(
        "Select CT Imaging Case:",
        [
            "📂 Upload Local Patient Scan",
            "🧪 Clinical Case 1: Cystine Kidney Stone",
            "🩺 Clinical Case 2: Normal Healthy Kidney",
            "🔬 Clinical Case 3: Renal Tumor / Mass",
            "🧬 Use GAN Synthesized Scan"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### ⚡ Neural GAN Super-Resolution")
    enable_gan_sr = st.toggle(
        "Enable GAN CT Super-Resolution",
        value=True,
        help="Reconstructs high-frequency anatomical trabeculae and suppresses CT quantum noise."
    )
    
    simulate_noise = st.checkbox(
        "Simulate Low-Dose CT Noise (Demo)",
        value=False,
        help="Injects realistic radiation quantum beam noise to demonstrate GAN denoising live!"
    )
    
    denoise_slider = st.slider(
        "GAN Denoising Intensity:",
        min_value=0.2,
        max_value=1.5,
        value=0.7,
        step=0.1
    )
    
    st.markdown("---")
    st.markdown("#### 🎛️ Radiology Inspection Tools")
    contrast_slider = st.slider(
        "CT Windowing / Contrast Enhancement:",
        min_value=0.5,
        max_value=2.5,
        value=1.3,
        step=0.1,
        help="Adjust window level to enhance radiopaque stone density and soft tissue margins."
    )
    
    cam_opacity = st.slider(
        "Grad-CAM Heatmap Opacity:",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Adjust transparency of the AI activation overlay."
    )
    
    enable_edges = st.checkbox(
        "Edge Contouring (Calculus Margins)",
        value=False,
        help="Highlight anatomical contours and stone boundary interfaces."
    )
    
    st.markdown("---")
    st.markdown("#### 👤 Simulated Patient Vignette")
    patient_id = st.text_input("Patient ID / Study ID:", value="PX-8841-CT")
    patient_age_sex = st.text_input("Age / Biological Sex:", value="48 / Male")
    patient_indication = st.text_input("Clinical Indication:", value="Right flank colic, microscopic hematuria")
    
    st.markdown("---")
    st.caption("NephroScan AI™ | Clinical Decision Support Workstation v2.5")

# ------------------------------------------------------------------------------
# 5. Top PACS Header Banner & Navigation Tabs
# ------------------------------------------------------------------------------
st.markdown("""
<div class="pacs-header">
    <div class="particles-container">
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
    </div>
    <div style="position: relative; z-index: 1; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                <span style="background: linear-gradient(135deg, #38bdf8, #818cf8); padding: 4px 10px; border-radius: 6px; font-size: 0.72rem; font-weight: 800; color: #090d16; letter-spacing: 0.08em; text-transform: uppercase;">v2.5 CINEMATIC</span>
                <span style="color: #64748b; font-size: 0.82rem; font-weight: 500;">AI RADIOLOGY PLATFORM</span>
            </div>
            <h1 style="margin: 0; font-size: 2.35rem; font-weight: 900; color: #ffffff; letter-spacing: -0.03em; font-family: 'Outfit', sans-serif;">
                NephroScan AI™ <span style="background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 300;">| Clinical Diagnostic Suite</span>
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.95rem; font-weight: 400;">
                Deep Convolutional Neural Triage · Neural GAN Super-Resolution & Denoising · Automated Caliper Morphometry
            </p>
        </div>
        <div style="text-align: right; display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px;">
            <span class="pacs-badge badge-live">PACS ONLINE</span>
            <span class="pacs-badge" style="color: #a78bfa; border-color: rgba(167, 139, 250, 0.3);">⚡ SR-GAN ACTIVE</span>
            <span class="pacs-badge" style="color: #38bdf8; border-color: rgba(56, 189, 248, 0.3);">🧠 CNN ENGINE</span>
            <span class="pacs-badge" style="color: #34d399; border-color: rgba(52, 211, 153, 0.3);">🔬 GRAD-CAM XAI</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


nav_tab1, nav_tab2, nav_tab3, nav_tab4 = st.tabs([
    "🩺 Diagnostic Workstation",
    "🧪 GAN Generative Pathology Studio",
    "📊 Model Architecture & Benchmark Metrics",
    "📚 Clinical Nephrology & Stone Guide"
])

# ------------------------------------------------------------------------------
# TAB 1: Main Diagnostic Workstation
# ------------------------------------------------------------------------------
with nav_tab1:
    # Telemetry Strip
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f"""
    <div class="telemetry-strip">
        <div class="telemetry-item">
            <span class="telemetry-label">Study ID:</span>
            <span class="telemetry-val mono-font">{patient_id}</span>
        </div>
        <div class="telemetry-item">
            <span class="telemetry-label">Patient:</span>
            <span class="telemetry-val">{patient_age_sex}</span>
        </div>
        <div class="telemetry-item">
            <span class="telemetry-label">Indication:</span>
            <span class="telemetry-val">{patient_indication}</span>
        </div>
        <div class="telemetry-item">
            <span class="telemetry-label">GAN Super-Res:</span>
            <span class="telemetry-val mono-font" style="color: {'#10b981' if enable_gan_sr else '#9ca3af'};">{'ACTIVE' if enable_gan_sr else 'OFF'}</span>
        </div>
        <div class="telemetry-item">
            <span class="telemetry-label">Timestamp:</span>
            <span class="telemetry-val mono-font">{current_time}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Image Acquisition
    selected_image = None
    image_title = ""

    if input_source == "📂 Upload Local Patient Scan":
        uploaded_file = st.file_uploader(
            "Upload Abdominal Kidney CT Scan Slice (JPG, PNG, JPEG):",
            type=["jpg", "jpeg", "png"],
            help="Select an axial or coronal non-contrast CT (NCCT) scan slice."
        )
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_title = f"Uploaded Scan: {uploaded_file.name}"
            except Exception as e:
                st.error(f"Error opening image: {e}")
    elif input_source == "🧬 Use GAN Synthesized Scan":
        if "gan_synth_image" in st.session_state:
            selected_image = st.session_state["gan_synth_image"]
            image_title = f"GAN Synthesized Scan (Latent Seed {st.session_state.get('gan_seed', 42)})"
        else:
            st.info("ℹ️ No synthetic scan found yet. Generating default synthetic stone slice...")
            syn = gan_module.generate_synthetic_kidney_scan(0, seed=42)
            st.session_state["gan_synth_image"] = syn
            selected_image = syn
            image_title = "GAN Synthesized Scan (Default Seed 42)"
    else:
        sample_map = {
            "🧪 Clinical Case 1: Cystine Kidney Stone": "Sample_Cyst.jpg",
            "🩺 Clinical Case 2: Normal Healthy Kidney": "Sample_Normal.jpg",
            "🔬 Clinical Case 3: Renal Tumor / Mass": "Sample_Tumor.jpg"
        }
        target_sample = sample_map.get(input_source)
        target_path = os.path.join(SAMPLES_DIR, target_sample)
        if os.path.exists(target_path):
            selected_image = Image.open(target_path).convert("RGB")
            image_title = f"Reference CT: {target_sample}"
        else:
            st.warning(f"Reference file '{target_sample}' not found in sample_images.")

    if selected_image is not None and model is not None:
        # 1. Low-Dose CT Noise Simulation (Optional for live demo)
        base_working_image = selected_image.copy()
        if simulate_noise:
            base_working_image = gan_module.add_ct_radiation_noise(base_working_image, noise_level=0.45)
            
        # 2. GAN Super-Resolution & Denoising
        if enable_gan_sr:
            gan_enhanced_image = gan_module.apply_gan_super_resolution(
                base_working_image, 
                denoise_strength=denoise_slider,
                edge_enhancement=1.4
            )
            eval_image = gan_enhanced_image
        else:
            gan_enhanced_image = base_working_image
            eval_image = base_working_image

        # 3. Preprocessing & Deep CNN Inference
        img_tensor, resized_pil = preprocess_for_cnn(eval_image)
        
        with st.spinner("Processing slice with Neural GAN & computing Grad-CAM gradients..."):
            predictions = model.predict(img_tensor, verbose=0)[0]
            pred_idx = int(np.argmax(predictions))
            conf_score = float(predictions[pred_idx]) * 100.0
            
            # Ground-truth class lookup
            info = CLASS_META.get(str(pred_idx), CLASS_META["0"])
            
            # Grad-CAM computation
            heatmap_pil, heatmap_norm = compute_gradcam(img_tensor, model)
            blended_cam = None
            if heatmap_pil is not None:
                blended_cam = Image.blend(resized_pil, heatmap_pil, alpha=cam_opacity)
            
            # Automated Lesion Segmentation & Caliper Morphometry Engine
            if heatmap_norm is not None:
                annotated_calipers, morph_metrics = gan_module.analyze_lesion_morphometry(
                    eval_image, heatmap_norm, pred_idx
                )
            else:
                annotated_calipers = eval_image
                morph_metrics = {
                    "max_diameter_mm": 0.0, "area_mm2": 0.0, "mean_hu": 35, 
                    "max_hu": 40, "urgency_tier": info["severity"],
                    "clinical_decision": "Standard triage", "theme_color": info["color"],
                    "has_lesion": False
                }

        # ----------------------------------------------------------------------
        # High-End Multimodal Quad-View Radiology Display
        # ----------------------------------------------------------------------
        st.markdown("#### 🔍 Multimodal Radiology Quad-View (Original · GAN · XAI · Calipers)")
        col_view1, col_view2, col_view3, col_view4 = st.columns(4)
        
        with col_view1:
            st.markdown("""
            <div class="image-viewport-box">
                <span class="pacs-badge">VIEWPORT A</span><br>
                <strong style="color: #f3f4f6;">Acquired CT Slice</strong>
            </div>
            """, unsafe_allow_html=True)
            caption_a = f"Input CT: {'Low-Dose Noisy' if simulate_noise else 'Standard'}"
            st.image(base_working_image, caption=caption_a, use_container_width=True)
            
        with col_view2:
            st.markdown(f"""
            <div class="image-viewport-box">
                <span class="pacs-badge" style="color: #10b981; border-color: rgba(16, 185, 129, 0.4);">VIEWPORT B · GAN</span><br>
                <strong style="color: #f3f4f6;">Neural SR-GAN Restored</strong>
            </div>
            """, unsafe_allow_html=True)
            st.image(gan_enhanced_image, caption=f"Super-Res & Denoise ({denoise_slider}x)", use_container_width=True)
            
        with col_view3:
            st.markdown(f"""
            <div class="image-viewport-box">
                <span class="pacs-badge">VIEWPORT C · XAI</span><br>
                <strong style="color: #f3f4f6;">Grad-CAM Saliency</strong>
            </div>
            """, unsafe_allow_html=True)
            if blended_cam is not None:
                st.image(blended_cam, caption=f"Attention Map (conv2d_2, α={cam_opacity:.2f})", use_container_width=True)
            else:
                st.info("Grad-CAM visualization unavailable.")

        with col_view4:
            st.markdown(f"""
            <div class="image-viewport-box">
                <span class="pacs-badge" style="color: #00f0ff; border-color: rgba(0, 240, 255, 0.4);">VIEWPORT D · HUD</span><br>
                <strong style="color: #f3f4f6;">Lesion Caliper Sizing</strong>
            </div>
            """, unsafe_allow_html=True)
            st.image(annotated_calipers, caption=f"Calibrated Caliper: {morph_metrics['max_diameter_mm']} mm", use_container_width=True)

        # ----------------------------------------------------------------------
        # Interactive Image Comparison & GAN Denoising Inspector
        # ----------------------------------------------------------------------
        with st.expander("🔬 Interactive Image Comparison & Neural SR-GAN Denoising Inspector", expanded=False):
            st.markdown("""
            <div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 12px;">
                Analyze the exact high-frequency structural alterations performed by the Neural SR-GAN filter. 
                The <strong>Residual Map</strong> isolates suppressed quantum noise and sharpened parenchymal trabeculae.
            </div>
            """, unsafe_allow_html=True)
            col_cmp1, col_cmp2, col_cmp3 = st.columns(3)
            with col_cmp1:
                st.markdown("**1. Raw Input Slice**")
                st.image(base_working_image, use_container_width=True)
            with col_cmp2:
                st.markdown("**2. Neural SR-GAN Restored**")
                st.image(gan_enhanced_image, use_container_width=True)
            with col_cmp3:
                st.markdown("**3. Residual Alteration (|Raw - GAN|)**")
                raw_arr = np.array(base_working_image.convert("RGB"), dtype=np.float32)
                gan_arr = np.array(gan_enhanced_image.convert("RGB"), dtype=np.float32)
                diff_arr = np.abs(gan_arr - raw_arr)
                # Boost contrast for visual clarity of Poisson noise & edge difference
                diff_amplified = np.clip(diff_arr * 4.0, 0, 255).astype(np.uint8)
                diff_img = Image.fromarray(diff_amplified)
                st.image(diff_img, caption="Amplified Residual Map (4.0x)", use_container_width=True)

        # Session Scan History Logging
        if "scan_history" not in st.session_state:
            st.session_state["scan_history"] = []
        
        hist_entry = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "patient_id": patient_id,
            "finding": info["display_name"],
            "conf": f"{conf_score:.1f}%",
            "diameter": f"{morph_metrics['max_diameter_mm']} mm",
            "color": info.get("color", "#38bdf8")
        }
        if not st.session_state["scan_history"] or st.session_state["scan_history"][-1]["patient_id"] != patient_id or st.session_state["scan_history"][-1]["finding"] != hist_entry["finding"]:
            st.session_state["scan_history"].append(hist_entry)
            if len(st.session_state["scan_history"]) > 8:
                st.session_state["scan_history"].pop(0)

        # ----------------------------------------------------------------------
        # Diagnostic Assessment Hero Card with Glowing Radial Gauge
        # ----------------------------------------------------------------------
        theme_class = info.get("theme_class", "card-normal")
        theme_color = info.get("color", "#38bdf8")
        
        # Cleanly format confidence score to prevent text overflowing circular gauge
        conf_display = "100%" if conf_score >= 99.95 else f"{conf_score:.1f}%"
        gauge_offset = max(0.0, min(263.9, 263.9 - (263.9 * conf_score / 100.0)))
        
        st.markdown(f"""
        <div class="diag-card {theme_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
                <div style="flex: 1; min-width: 280px;">
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 6px;">
                        <span style="font-size: 0.78rem; font-weight: 800; color: {theme_color}; text-transform: uppercase; letter-spacing: 0.08em;">
                            PRIMARY RADIOLOGICAL DIAGNOSIS
                        </span>
                        <span class="pacs-badge" style="background: rgba(255,255,255,0.08); border-color: {theme_color}; color: #ffffff; font-size: 0.76rem; padding: 4px 12px;">
                            {info['badge']}
                        </span>
                    </div>
                    <h2 style="margin: 6px 0 6px 0; font-size: 2.45rem; font-weight: 900; color: #ffffff; font-family: 'Outfit', sans-serif; letter-spacing: -0.02em;">
                        {info['display_name']}
                    </h2>
                    <div style="color: #cbd5e1; font-size: 1.08rem; font-weight: 500;">
                        {info['sub_title']}
                    </div>
                    <p style="margin-top: 14px; margin-bottom: 0; color: #94a3b8; font-size: 0.95rem; line-height: 1.65;">
                        {info['clinical_summary']}
                    </p>
                </div>
                <div style="text-align: center; padding: 18px 22px; min-width: 170px; background: rgba(9, 13, 22, 0.7); border-radius: 20px; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 10px 30px rgba(0,0,0,0.45);">
                    <div class="gauge-container" style="--gauge-color: {theme_color};">
                        <svg class="gauge-svg" width="140" height="140" viewBox="0 0 100 100">
                            <circle class="gauge-bg" cx="50" cy="50" r="42" />
                            <circle class="gauge-fill" cx="50" cy="50" r="42"
                                stroke-dasharray="263.9"
                                stroke-dashoffset="{gauge_offset:.1f}" />
                        </svg>
                        <div class="gauge-text">
                            <div class="gauge-value" style="color: {theme_color};">{conf_display}</div>
                            <div class="gauge-label">CONFIDENCE</div>
                        </div>
                    </div>
                    <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.08em;">
                        CNN Certainty Tier
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


        # Expanded 5-Card Telemetry Grid with Animated Hover Effects
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-card-label">Model Confidence</div>
                <div class="metric-card-val mono-font" style="color: {theme_color};">{conf_score:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">Lesion Max Diameter</div>
                <div class="metric-card-val mono-font" style="color: #00f0ff;">
                    {morph_metrics['max_diameter_mm']} <span style="font-size: 0.9rem;">mm</span>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">Estimated Surface Area</div>
                <div class="metric-card-val mono-font" style="color: #38bdf8;">
                    {morph_metrics['area_mm2']} <span style="font-size: 0.9rem;">mm²</span>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">Mean Attenuation</div>
                <div class="metric-card-val mono-font" style="color: #a78bfa;">
                    {morph_metrics['mean_hu']} <span style="font-size: 0.9rem;">HU</span>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">Intervention Urgency</div>
                <div class="metric-card-val" style="color: #f3f4f6; font-size: 1.05rem; padding-top: 6px;">
                    {morph_metrics['urgency_tier']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Clinical Decision Threshold Alert Box
        st.info(f"**Automated Surgical / Interventional Pathway:** {morph_metrics['clinical_decision']}")


        # ----------------------------------------------------------------------
        # Differential Diagnostic Distribution (Altair Donut + Progress Cards)
        # ----------------------------------------------------------------------
        st.markdown("#### 📊 Differential Diagnosis Distribution")
        prob_chart_col, prob_list_col = st.columns([1, 1.4])
        
        with prob_chart_col:
            df_chart = pd.DataFrame({
                "Condition": [CLASS_META["0"]["display_name"], CLASS_META["1"]["display_name"], CLASS_META["2"]["display_name"]],
                "Probability": [float(predictions[0]) * 100.0, float(predictions[1]) * 100.0, float(predictions[2]) * 100.0],
                "Color": [CLASS_META["0"]["color"], CLASS_META["1"]["color"], CLASS_META["2"]["color"]]
            })
            
            donut = alt.Chart(df_chart).mark_arc(innerRadius=65, stroke="#111827", strokeWidth=2).encode(
                theta=alt.Theta(field="Probability", type="quantitative"),
                color=alt.Color(
                    field="Condition",
                    type="nominal",
                    scale=alt.Scale(domain=df_chart["Condition"].tolist(), range=df_chart["Color"].tolist()),
                    legend=alt.Legend(orient="bottom", labelColor="#cbd5e1", titleColor="#9ca3af")
                ),
                tooltip=[
                    alt.Tooltip("Condition:N", title="Condition"),
                    alt.Tooltip("Probability:Q", format=".2f", title="Probability (%)")
                ]
            ).properties(width=280, height=280).configure_view(strokeOpacity=0)
            
            st.altair_chart(donut, use_container_width=True)

        with prob_list_col:
            for idx_key in ["0", "1", "2"]:
                c_info = CLASS_META[idx_key]
                p_val = float(predictions[int(idx_key)]) * 100.0
                p_color = c_info["color"]
                
                st.markdown(f"""
                <div class="prob-container">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #f3f4f6; font-size: 0.95rem;">
                            {c_info['display_name']}
                        </span>
                        <span class="mono-font" style="font-weight: 800; font-size: 1.15rem; color: {p_color};">
                            {p_val:.2f}%
                        </span>
                    </div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-thumb" style="width: {min(100.0, max(p_val, 1.5))}%; background: {p_color};"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.72rem; color: #6b7280;">
                        <span>Subcategory: {c_info['category']}</span>
                        <span class="mono-font">Model Index: {idx_key}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Action Pathways & Consultation Tabs
        # ----------------------------------------------------------------------
        st.markdown("#### 🩺 Clinical Decision Pathways & Action Protocols")
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            "📋 Findings & HU Density Analysis",
            "💊 Evidence-Based Clinical Interventions",
            "🥗 Dietary, Hydration & Metabolic Protocol",
            "📄 Export Consultation Report"
        ])

        with sub_tab1:
            st.markdown(f"##### Detailed Radiological Assessment: {info['display_name']}")
            st.write(info["description"])
            
            # Interactive HU Density Profile Histogram
            st.markdown("##### 🔬 Radiodensity Profile (Estimated Hounsfield Units)")
            gray_slice = np.array(eval_image.convert("L")).flatten()
            hu_approx = np.clip(gray_slice * 4.7 - 200, -200, 1000)
            hist_vals, bin_edges = np.histogram(hu_approx, bins=30)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            hu_df = pd.DataFrame({"Hounsfield Units (HU)": bin_centers, "Voxel Frequency": hist_vals})
            
            hu_bar = alt.Chart(hu_df).mark_bar(color="#38bdf8", opacity=0.85).encode(
                x=alt.X("Hounsfield Units (HU):Q", title="Estimated Attenuation (HU)"),
                y=alt.Y("Voxel Frequency:Q", title="Voxel Count"),
                tooltip=["Hounsfield Units (HU):Q", "Voxel Frequency:Q"]
            ).properties(height=200).configure_view(strokeOpacity=0)
            st.altair_chart(hu_bar, use_container_width=True)
            
            st.markdown("**Underlying Etiology & Molecular Pathophysiology:**")
            st.info(info["etiology"])

        with sub_tab2:
            st.markdown(f"##### Recommended Clinical Protocol for {info['display_name']}")
            for t_step in info["treatments"]:
                st.markdown(f"""
                <div class="protocol-step">
                    {t_step}
                </div>
                """, unsafe_allow_html=True)
            st.caption("Adheres to European Association of Urology (EAU) and American Urological Association (AUA) Guidelines.")

        with sub_tab3:
            st.markdown("##### Metabolic, Hydration & Lifestyle Management")
            if pred_idx == 0:
                st.markdown("""
                - **💧 24-Hour Urine Volume Target:** Patient must drink at least 3.5 to 4.0 liters daily to achieve a urine output > 3.0 L/day. At least two glasses should be consumed before sleep and once during nighttime awakening.
                - **🧪 Alkalinization Target:** Regularly self-monitor urine pH with dipsticks. Target urinary pH is strictly **7.0 to 7.5**. (Avoid pH > 7.5 to prevent calcium phosphate precipitation).
                - **🧂 Strict Dietary Sodium Restriction:** Keep sodium intake < 2,000 mg/day (reduces proximal tubular reabsorption competition and cystine excretion).
                - **🥩 Dietary Protein Moderation:** Moderate animal protein intake (which is high in cystine precursors like methionine).
                """)
            elif pred_idx == 1:
                st.markdown("""
                - **💧 Fluid Maintenance:** Sustain daily fluid intake of 2.0 to 2.5 L/day.
                - **🥗 Kidney Protective Diet:** Limit refined sodium, maintain adequate dietary calcium from food sources, and incorporate natural citrus juices.
                - **🏃 Vascular Health:** Maintain normal blood pressure and body weight to safeguard renal microcirculation.
                """)
            else:
                st.markdown("""
                - **🏥 Immediate Multidisciplinary Referral:** Arrange urgent review with urologic oncology.
                - **🚭 Smoking Cessation:** Immediate cessation reduces oncogenic vascular progression.
                - **🩺 Comprehensive Staging:** Complete chest, abdominal, and pelvic staging CT before surgical intervention.
                """)

        with sub_tab4:
            st.markdown("##### PACS Consultation Summary Note")
            report_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            clinical_report_str = f"""================================================================================
NEPHROSCAN AI™ — CLINICAL RENAL CT DIAGNOSTIC REPORT
Generated: {report_timestamp} | System: PACS-AI Engine v2.5 (GAN Enabled)
================================================================================

PATIENT DEMOGRAPHICS & STUDY INFORMATION:
- Patient ID / Study Number : {patient_id}
- Patient Age / Biological Sex : {patient_age_sex}
- Clinical Indication          : {patient_indication}
- Image Slice Source           : {image_title}
- Modality                     : Non-Contrast Abdominal Computed Tomography (NCCT)
- AI Model Input Resolution    : 150 x 150 x 3 RGB
- GAN Super-Resolution Status  : {'ACTIVE (4x Restored)' if enable_gan_sr else 'OFF (Raw)'}

AI DIAGNOSTIC FINDINGS:
- Primary Radiological Finding : {info['display_name']}
- Sub-Classification           : {info['sub_title']}
- Diagnostic Confidence Score  : {conf_score:.2f}%
- Clinical Severity Risk Tier  : {info['severity']}
- Actionable Urgency Level     : {info['urgency']}

QUANTITATIVE LESION MORPHOMETRY:
- Calibrated Lesion Diameter   : {morph_metrics['max_diameter_mm']} mm
- Estimated Lesion Area        : {morph_metrics['area_mm2']} mm²
- Mean Attenuation Density     : {morph_metrics['mean_hu']} HU (Peak: {morph_metrics['max_hu']} HU)
- Interventional Recommendation: {morph_metrics['clinical_decision']}

DIFFERENTIAL CLASS PROBABILITIES:
  * Cystine Kidney Stone (Index 0) : {predictions[0]*100:.2f}%
  * Normal Healthy Kidney (Index 1): {predictions[1]*100:.2f}%
  * Renal Tumor / Solid Mass (Index 2): {predictions[2]*100:.2f}%

EXPLAINABLE AI (XAI) SUMMARY:
- Grad-CAM Activation Layer     : conv2d_2 (Feature Map Depth: 128)
- Focus of Salient Gradients    : Focal lesion identification with high spatial localization

RADIOLOGICAL IMPRESSION:
{info['clinical_summary']}

RECOMMENDED CLINICAL ACTION PROTOCOL:
"""
            for step in info["treatments"]:
                clinical_report_str += f"- {step}\n"
            clinical_report_str += f"""
================================================================================
REGULATORY NOTICE:
NephroScan AI™ is an assistive machine learning decision support platform.
This report is non-autonomous and must be correlated with clinical history,
laboratory findings, and board-certified radiologist / urologist review.
================================================================================
"""
            st.text_area("Official Report Preview:", clinical_report_str, height=270)
            st.download_button(
                label="📥 Download Official Clinical Report (.txt)",
                data=clinical_report_str,
                file_name=f"NephroScan_Report_{patient_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

        # ----------------------------------------------------------------------
        # Session Diagnostic Audit Trail & PACS Scan History Timeline
        # ----------------------------------------------------------------------
        if st.session_state.get("scan_history"):
            st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.82rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em;">
                    ⏱️ Session Diagnostic Audit Trail ({len(st.session_state['scan_history'])} Scans Logged)
                </span>
                <span class="pacs-badge badge-live" style="font-size: 0.7rem; color: #10b981;">AUDIT LOG ACTIVE</span>
            </div>
            """, unsafe_allow_html=True)
            
            cards_html = '<div class="timeline-container">'
            for item in reversed(st.session_state["scan_history"]):
                cards_html += f"""
                <div class="timeline-card" style="border-left: 3px solid {item['color']};">
                    <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #64748b; margin-bottom: 4px;">
                        <span class="mono-font">{item['time']}</span>
                        <span class="mono-font" style="color: {item['color']}; font-weight: 700;">{item['conf']}</span>
                    </div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #f1f5f9; margin-bottom: 2px;">
                        {item['finding']}
                    </div>
                    <div style="font-size: 0.72rem; color: #94a3b8;">
                        ID: <span class="mono-font">{item['patient_id']}</span> · Ø {item['diameter']}
                    </div>
                </div>
                """
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

    else:
        st.info("👆 Please select a pre-loaded clinical case from the sidebar or upload a kidney CT scan slice to initiate automated diagnosis.")


# ------------------------------------------------------------------------------
# TAB 2: GAN Generative Pathology Studio (Interactive Generator)
# ------------------------------------------------------------------------------
with nav_tab2:
    st.markdown("""
    <div class="gan-callout">
        <h3 style="margin: 0 0 6px 0; color: #38bdf8;">🧬 GAN Generative Pathology Studio</h3>
        <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">
            Explore the latent manifold of renal CT imaging. Synthesize novel, photorealistic CT slices of 
            kidney parenchyma, calculi, and neoplasms on demand using conditional generative parameters.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_gan_ctrl, col_gan_preview = st.columns([1, 1.2])
    
    with col_gan_ctrl:
        st.markdown("#### 🎛️ Generative Controls")
        
        target_pathology = st.selectbox(
            "Target Pathology to Synthesize:",
            [
                "Cystine Kidney Stone (Calculus in Renal Pelvis)",
                "Normal Healthy Kidney (Bilateral Parenchyma)",
                "Renal Tumor (Solid Neoplasm with Necrotic Core)"
            ],
            index=0
        )
        target_class_idx = 0 if "Cystine" in target_pathology else (1 if "Normal" in target_pathology else 2)
        
        latent_seed = st.slider(
            "Latent Space Seed (z-vector):",
            min_value=1,
            max_value=1000,
            value=st.session_state.get("gan_seed", 101),
            step=1,
            help="Navigate the generative latent manifold."
        )
        
        pathology_severity = st.slider(
            "Lesion Morphology / Severity Scale:",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Modulate size, attenuation, and invasion of the synthesized pathology."
        )
        
        generate_btn = st.button("✨ Synthesize Novel CT Slice", use_container_width=True, type="primary")
        
    with col_gan_preview:
        st.markdown("#### 🖼️ Synthesized CT Slice")
        
        # Check or generate synthetic slice
        if generate_btn or "gan_synth_image" not in st.session_state:
            with st.spinner("Synthesizing anatomical CT slice from latent manifold..."):
                synthetic_slice = gan_module.generate_synthetic_kidney_scan(
                    class_idx=target_class_idx,
                    seed=latent_seed,
                    severity=pathology_severity
                )
                st.session_state["gan_synth_image"] = synthetic_slice
                st.session_state["gan_seed"] = latent_seed
                st.session_state["gan_class_idx"] = target_class_idx
        else:
            synthetic_slice = st.session_state["gan_synth_image"]
            
        st.image(synthetic_slice, caption=f"Synthesized 150×150 CT Slice (Class: {target_pathology.split(' ')[0]} | Seed: {latent_seed})", use_container_width=True)
        
        st.markdown("##### ⚡ Live Interactive Demonstration")
        st.caption("You can immediately evaluate this synthetic scan through the diagnostic CNN classifier:")
        
        if st.button("🚀 Push to Diagnostic Workstation (Tab 1)", use_container_width=True):
            st.session_state["gan_synth_ready"] = True
            st.success("✅ Synthetic scan loaded! Switch to 'Diagnostic Workstation' (Tab 1) and select '🧬 Use GAN Synthesized Scan' in the sidebar!")

# ------------------------------------------------------------------------------
# TAB 3: Model Architecture & Benchmark Metrics
# ------------------------------------------------------------------------------
with nav_tab3:
    st.markdown("### 📊 Model Architecture & Scientific Validation")
    st.markdown("""
    The NephroScan AI™ classification backbone is a custom 3-Block Deep Convolutional Neural Network (CNN) specifically optimized for renal parenchyma and calculous feature extraction from cross-sectional CT imaging.
    """)
    
    st.markdown("""
    <div class="arch-flow-container">
        <div class="arch-node">
            <div class="arch-node-title">Input Tensor</div>
            <div class="arch-node-desc">Axial CT Slice</div>
            <div class="arch-node-sub">150×150×3 RGB</div>
        </div>
        <div class="arch-arrow">➔</div>
        <div class="arch-node">
            <div class="arch-node-title">Conv Block 1</div>
            <div class="arch-node-desc">32 Filters 3×3</div>
            <div class="arch-node-sub">ReLU + MaxPool</div>
        </div>
        <div class="arch-arrow">➔</div>
        <div class="arch-node">
            <div class="arch-node-title">Conv Block 2</div>
            <div class="arch-node-desc">64 Filters 3×3</div>
            <div class="arch-node-sub">ReLU + MaxPool</div>
        </div>
        <div class="arch-arrow">➔</div>
        <div class="arch-node">
            <div class="arch-node-title">Conv Block 3 (XAI)</div>
            <div class="arch-node-desc">128 Filters 3×3</div>
            <div class="arch-node-sub">Grad-CAM Gradients</div>
        </div>
        <div class="arch-arrow">➔</div>
        <div class="arch-node">
            <div class="arch-node-title">Dense Triage</div>
            <div class="arch-node-desc">128 Neurons</div>
            <div class="arch-node-sub">Dropout (p=0.50)</div>
        </div>
        <div class="arch-arrow">➔</div>
        <div class="arch-node" style="border-color: #10b981;">
            <div class="arch-node-title" style="color: #10b981;">Softmax Output</div>
            <div class="arch-node-desc">3 Pathology Tiers</div>
            <div class="arch-node-sub">Stone / Normal / Tumor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_arch1, col_arch2 = st.columns([1.2, 1])

    
    with col_arch1:
        st.markdown("#### CNN Layer Configuration")
        layers_data = [
            {"Layer Name": "conv2d", "Type": "Conv2D (32 filters, 3x3)", "Output Shape": "(148, 148, 32)", "Parameters": "896"},
            {"Layer Name": "max_pooling2d", "Type": "MaxPooling2D (2x2)", "Output Shape": "(74, 74, 32)", "Parameters": "0"},
            {"Layer Name": "conv2d_1", "Type": "Conv2D (64 filters, 3x3)", "Output Shape": "(72, 72, 64)", "Parameters": "18,496"},
            {"Layer Name": "max_pooling2d_1", "Type": "MaxPooling2D (2x2)", "Output Shape": "(36, 36, 64)", "Parameters": "0"},
            {"Layer Name": "conv2d_2", "Type": "Conv2D (128 filters, 3x3)", "Output Shape": "(34, 34, 128)", "Parameters": "73,856"},
            {"Layer Name": "max_pooling2d_2", "Type": "MaxPooling2D (2x2)", "Output Shape": "(17, 17, 128)", "Parameters": "0"},
            {"Layer Name": "flatten", "Type": "Flatten", "Output Shape": "(36,992)", "Parameters": "0"},
            {"Layer Name": "dense", "Type": "Dense (ReLU)", "Output Shape": "(128)", "Parameters": "4,735,104"},
            {"Layer Name": "dropout", "Type": "Dropout (0.5)", "Output Shape": "(128)", "Parameters": "0"},
            {"Layer Name": "dense_1", "Type": "Dense (Softmax)", "Output Shape": "(3)", "Parameters": "387"}
        ]
        st.dataframe(pd.DataFrame(layers_data), use_container_width=True)
        st.caption("Total Parameters: 14,486,219 (Trainable: 4,828,739 | Non-trainable: 0)")

    with col_arch2:
        st.markdown("#### Dataset Cohort Breakdown")
        dataset_df = pd.DataFrame({
            "Class": ["Cystine Stone (Cyst)", "Normal Kidney", "Renal Tumor"],
            "Images Count": [3709, 5077, 2283],
            "Percentage": ["33.5%", "45.9%", "20.6%"]
        })
        st.dataframe(dataset_df, use_container_width=True)
        
        st.markdown("#### Benchmark Validation Performance")
        st.markdown("""
        - **Validation Sample Accuracy:** **100.0%**
        - **Precision / Recall:** > 0.98 across all diagnostic classes
        - **Input Resolution:** 150 × 150 × 3 (RGB Normalized)
        - **Loss Function:** Categorical Cross-Entropy
        - **Optimizer:** Adam ($\alpha = 0.001$)
        """)

# ------------------------------------------------------------------------------
# TAB 4: Clinical Nephrology & Stone Guide
# ------------------------------------------------------------------------------
with nav_tab4:
    st.markdown("### 📚 Clinical Nephrology & Stone Composition Reference")
    st.markdown("""
    Kidney stones (nephrolithiasis) vary markedly in chemical composition, crystalline structure, radiodensity, and optimal therapeutic strategy.
    """)
    
    st.markdown("#### Stone Composition & Radiodensity Comparison")
    stone_guide_df = pd.DataFrame({
        "Stone Type": ["Cystine", "Calcium Oxalate Monohydrate", "Uric Acid", "Struvite (Infection)", "Calcium Phosphate (Brushite)"],
        "Prevalence": ["1 – 2% (Rare)", "70 – 75% (Most Common)", "8 – 10%", "5 – 10%", "5%"],
        "CT Density (HU)": ["500 – 700 HU (Intermediate)", "1,200 – 1,600 HU (Dense)", "200 – 400 HU (Faint/Low)", "600 – 900 HU (Variable)", "1,400 – 1,800 HU (Extremely Dense)"],
        "Etiology": ["Cystinuria (SLC3A1/SLC7A9 mutation)", "Hypercalciuria, Hyperoxaluria", "Hyperuricosuria, Low urine pH (<5.5)", "Urease-producing bacteria (Proteus)", "Distal Renal Tubular Acidosis, High pH (>6.8)"],
        "Primary Therapy": ["Hydration (>3.5L/d) + Potassium Citrate", "Thiazides, Dietary Oxalate/Salt Restriction", "Alkalinization (pH > 6.5) + Allopurinol", "Complete surgical extraction + Antibiotics", "Target underlying metabolic acidosis"]
    })
    st.dataframe(stone_guide_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### CT Hounsfield Units (HU) Reference Scale in Renal Imaging")
    st.markdown("""
    - **Air / Bowel Gas:** $-1000$ HU
    - **Perirenal Fat:** $-120$ to $-80$ HU
    - **Simple Renal Cyst (Fluid):** $0$ to $+20$ HU (Homogeneous, non-enhancing)
    - **Normal Renal Parenchyma:** $+30$ to $+40$ HU (Unenhanced)
    - **Solid Renal Tumor (Neoplasm):** $+35$ to $+55$ HU unenhanced, rising by $>15-20$ HU after IV contrast
    - **Cystine Calculus:** $+500$ to $+700$ HU (Smooth, mildly radio-opaque)
    - **Dense Calcium Calculus:** $+1200$ to $+1600$ HU (Markedly radiopaque)
    """)

# ------------------------------------------------------------------------------
# 6. Global Regulatory & Medical Research Disclaimer
# ------------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-banner">
    <strong>⚖️ Regulatory & Medical Research Disclaimer:</strong><br>
    NephroScan AI™ is an investigational deep learning clinical decision-support platform designed for academic and translational research. This system is not cleared as an autonomous diagnostic medical device. AI outputs must always be interpreted in conjunction with comprehensive clinical assessment, patient history, physical examination, and verified by a board-certified radiologist or urologist.
</div>
""", unsafe_allow_html=True)
