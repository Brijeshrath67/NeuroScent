"""
Streamlit Web App: NeuroScent-GNN
Fully Real-Time In Silico Olfactory Platform & Electronic Nose Simulator.
Every calculation, graph, metric, and tensor across all 4 tabs is computed dynamically in real time.
Features exact shadcn/ui Dashboard design system with dynamic Light and Dark mode.
"""

import json
import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import altair as alt
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw, rdMolDescriptors
import torch_geometric
from sensor import GasSensorArray
from train_nose import load_nose, predict_nose, train_nose
from smell_model import load_smell_model, predict_molecule_descriptors, predict_mixture_descriptors, train_smell_model
from gnn_model import load_or_train_gnn, predict_gnn_smell, predict_gnn_mixture, mol_to_graph
from sklearn.decomposition import PCA

# Global hardware acceleration device (Apple Silicon M4 10-Core GPU via MPS, with CPU fallback)
gnn_device = "mps" if torch.backends.mps.is_available() else "cpu"

# Page configuration
st.set_page_config(
    page_title="NeuroScent-GNN: Deep Molecular Olfactory Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Theme State
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

is_dark = st.session_state["dark_mode"]

# Dynamic Chart & Theme Colors
if is_dark:
    chart_primary = "#FAFAFA"    # Light bar on dark background
    chart_secondary = "#2EB88A"  # Emerald
    chart_accent = "#E88C30"     # Amber
    chart_line = "#FAFAFA"       # White line

    theme_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }

        /* Sleek Modern Webkit Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #09090B;
        }
        ::-webkit-scrollbar-thumb {
            background: #27272A;
            border-radius: 9999px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #3F3F46;
        }

        /* Pulsating Live Status Indicator */
        @keyframes pulse-dot {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.35; transform: scale(1.2); }
            100% { opacity: 1; transform: scale(1); }
        }
        .pulse-indicator {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #22C55E;
            margin-right: 6px;
            animation: pulse-dot 2s infinite ease-in-out;
            vertical-align: middle;
        }

        /* Dark Canvas & Background */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #09090B !important;
            color: #FAFAFA !important;
        }

        /* Remove massive empty top padding & pull page up */
        .block-container, [data-testid="stAppViewBlockContainer"] {
            padding-top: 1.25rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 98% !important;
        }
        [data-testid="stHeader"] {
            height: 0px !important;
            min-height: 0px !important;
            background: transparent !important;
            display: none !important;
        }
        
        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color: #FAFAFA !important;
        }
        [data-testid="stMarkdownContainer"] > p {
            color: #FAFAFA !important;
        }
        .main-title {
            font-size: 2.35rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.035em !important;
            color: #FAFAFA !important;
            margin-bottom: 0.35rem !important;
            line-height: 1.2 !important;
        }
        .sub-title {
            font-size: 0.875rem !important;
            color: #A1A1AA !important;
            margin-bottom: 1.25rem !important;
        }
        
        /* Dark Mode Badges & Pills */
        span.tech-pill, [data-testid="stMarkdownContainer"] span.tech-pill {
            display: inline-block !important;
            padding: 0.2rem 0.55rem !important;
            margin-right: 0.35rem !important;
            margin-bottom: 0.3rem !important;
            border-radius: 6px !important;
            background-color: #18181B !important;
            color: #FAFAFA !important;
            font-weight: 500 !important;
            font-size: 0.75rem !important;
            border: 1px solid #27272A !important;
        }
        span.tech-pill-accent, [data-testid="stMarkdownContainer"] span.tech-pill-accent {
            display: inline-block !important;
            padding: 0.2rem 0.55rem !important;
            margin-right: 0.35rem !important;
            margin-bottom: 0.3rem !important;
            border-radius: 6px !important;
            background-color: #FAFAFA !important;
            color: #09090B !important;
            font-weight: 600 !important;
            font-size: 0.75rem !important;
            border: 1px solid #FAFAFA !important;
        }
        
        span.badge-high, [data-testid="stMarkdownContainer"] span.badge-high {
            display: inline-block !important;
            padding: 0.25rem 0.65rem !important;
            margin: 0.2rem !important;
            border-radius: 9999px !important;
            background-color: #FAFAFA !important;
            color: #09090B !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            border: 1px solid #FAFAFA !important;
        }
        span.badge-med, [data-testid="stMarkdownContainer"] span.badge-med {
            display: inline-block !important;
            padding: 0.25rem 0.65rem !important;
            margin: 0.2rem !important;
            border-radius: 9999px !important;
            background-color: #27272A !important;
            color: #FAFAFA !important;
            font-weight: 500 !important;
            font-size: 0.8rem !important;
            border: 1px solid #3F3F46 !important;
        }
        span.badge-low, [data-testid="stMarkdownContainer"] span.badge-low {
            display: inline-block !important;
            padding: 0.25rem 0.65rem !important;
            margin: 0.2rem !important;
            border-radius: 9999px !important;
            background-color: #18181B !important;
            color: #A1A1AA !important;
            font-weight: 400 !important;
            font-size: 0.8rem !important;
            border: 1px solid #27272A !important;
        }

        /* Dark Mode Card Containers */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 8px !important;
            border: 1px solid #27272A !important;
            background-color: #09090B !important;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.4) !important;
            padding: 4px !important;
        }

        /* Dark Mode Metric Cards */
        [data-testid="stMetric"] {
            background-color: #09090B !important;
            border: 1px solid #27272A !important;
            border-radius: 8px !important;
            padding: 12px 14px !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.3) !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            color: #A1A1AA !important;
            letter-spacing: -0.01em !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            color: #FAFAFA !important;
        }

        /* Dark Mode Navigation Tabs List */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #18181B !important;
            padding: 4px !important;
            border-radius: 8px !important;
            gap: 4px !important;
            border: 1px solid #27272A !important;
            margin-bottom: 1rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px !important;
            padding: 6px 14px !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: #A1A1AA !important;
            border: none !important;
            background-color: transparent !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #27272A !important;
            color: #FAFAFA !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.3) !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* Primary Button in Dark Mode: Solid White with Pitch Black Text */
        .stButton > button,
        button[data-testid="baseButton-primary"],
        button[kind="primary"] {
            background-color: #FAFAFA !important;
            border-radius: 6px !important;
            border: 1px solid #FAFAFA !important;
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            padding: 0.45rem 1rem !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.2) !important;
            transition: all 0.15s ease-in-out !important;
            color: #09090B !important;
        }
        .stButton > button *,
        button[data-testid="baseButton-primary"] *,
        button[kind="primary"] *,
        .stButton > button p,
        button[data-testid="baseButton-primary"] p,
        .stButton > button span,
        button[data-testid="baseButton-primary"] span {
            color: #09090B !important;
            fill: #09090B !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover,
        button[data-testid="baseButton-primary"]:hover,
        button[kind="primary"]:hover {
            background-color: #E4E4E7 !important;
            border-color: #E4E4E7 !important;
        }
        .stButton > button:hover *,
        button[data-testid="baseButton-primary"]:hover * {
            color: #09090B !important;
        }

        /* Download / Secondary Button in Dark Mode: Dark Zinc with Crisp White Text */
        .stDownloadButton > button,
        button[data-testid="baseButton-secondary"],
        button[kind="secondary"] {
            background-color: #18181B !important;
            border: 1px solid #27272A !important;
            border-radius: 6px !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: #FAFAFA !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.2) !important;
            transition: all 0.15s ease-in-out !important;
        }
        .stDownloadButton > button *,
        button[data-testid="baseButton-secondary"] *,
        .stDownloadButton > button p,
        button[data-testid="baseButton-secondary"] p,
        .stDownloadButton > button span,
        button[data-testid="baseButton-secondary"] span {
            color: #FAFAFA !important;
            fill: #FAFAFA !important;
            font-weight: 500 !important;
        }
        .stDownloadButton > button:hover,
        button[data-testid="baseButton-secondary"]:hover {
            background-color: #27272A !important;
            border-color: #3F3F46 !important;
        }

        /* Inputs, Selects & Expanders in Dark Mode */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        input {
            border-radius: 6px !important;
            border-color: #27272A !important;
            background-color: #18181B !important;
            color: #FAFAFA !important;
            font-size: 0.875rem !important;
        }
        div[data-baseweb="select"] * {
            color: #FAFAFA !important;
        }
        div[data-baseweb="popover"],
        ul[role="listbox"],
        li[role="option"] {
            background-color: #18181B !important;
            color: #FAFAFA !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background-color: #27272A !important;
            color: #FAFAFA !important;
        }

        /* Expanders in Dark Mode */
        [data-testid="stExpander"] {
            border: 1px solid #27272A !important;
            border-radius: 8px !important;
            background-color: #09090B !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.2) !important;
            overflow: hidden !important;
        }
        [data-testid="stExpander"] details {
            background-color: #09090B !important;
        }
        [data-testid="stExpander"] summary {
            background-color: #18181B !important;
            color: #FAFAFA !important;
            border-radius: 6px !important;
            padding: 0.5rem 0.75rem !important;
        }
        [data-testid="stExpander"] summary:hover {
            background-color: #27272A !important;
            color: #FAFAFA !important;
        }
        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {
            color: #FAFAFA !important;
            fill: #FAFAFA !important;
        }
        [data-testid="stExpander"] summary svg {
            fill: #FAFAFA !important;
            stroke: #FAFAFA !important;
        }
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: #09090B !important;
            color: #FAFAFA !important;
            border-top: 1px solid #27272A !important;
            padding: 0.75rem !important;
        }

        /* Radio & Toggle in Dark Mode */
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label p,
        [data-testid="stRadio"] label span {
            color: #FAFAFA !important;
        }
        div[data-testid="stToggle"] label p,
        div[data-testid="stToggle"] label span {
            color: #FAFAFA !important;
            font-weight: 500 !important;
        }

        /* Invert 2D Chemical Structure in Dark Mode */
        [data-testid="stImage"] img {
            border-radius: 6px !important;
            filter: invert(0.92) hue-rotate(180deg) !important;
            background-color: #09090B !important;
        }

        /* Force Vega Lite chart container to match background */
        div[data-testid="stVegaLiteChart"],
        div[data-testid="stVegaLiteChart"] > div,
        div[data-testid="stVegaLiteChart"] svg,
        div[data-testid="stVegaLiteChart"] canvas {
            background-color: transparent !important;
        }

        /* Dark Mode Progress Bar */
        [data-testid="stProgress"] > div > div > div > div {
            background-color: #FAFAFA !important;
            border-radius: 9999px !important;
        }
        [data-testid="stProgress"] > div > div {
            background-color: #27272A !important;
            border-radius: 9999px !important;
        }

        /* Code in Dark Mode */
        code {
            background-color: #18181B !important;
            color: #FAFAFA !important;
            border: 1px solid #27272A !important;
            border-radius: 4px !important;
            padding: 0.15rem 0.35rem !important;
        }

        /* Micro-interactions & Polished Transitions */
        [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"] {
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: #3F3F46 !important;
        }
        .stButton > button:active, .stDownloadButton > button:active {
            transform: scale(0.985) !important;
        }
        div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within {
            border-color: #38BDF8 !important;
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
        }

        /* Deployment Footer in Dark Mode */
        .deployment-footer {
            margin-top: 2.5rem;
            padding: 1.25rem 0.5rem;
            border-top: 1px solid #27272A;
            font-size: 0.82rem;
            color: #71717A;
            font-family: 'Inter', -apple-system, sans-serif;
        }
        .developer-highlight {
            color: #FAFAFA !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(168, 85, 247, 0.15));
            border: 1px solid rgba(56, 189, 248, 0.4);
            padding: 2px 9px;
            border-radius: 5px;
            letter-spacing: -0.01em;
            display: inline-block;
        }

        /* Toast Notifications in Dark Mode - shadcn/ui specification */
        [data-testid="stToast"], .stToast, div[data-testid="stToast"] {
            background-color: #18181B !important;
            color: #FAFAFA !important;
            border: 1px solid #27272A !important;
            border-radius: 8px !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6), 0 8px 10px -6px rgba(0, 0, 0, 0.6) !important;
            padding: 0.75rem 1rem !important;
        }
        [data-testid="stToast"] [data-testid="stMarkdownContainer"],
        [data-testid="stToast"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stToast"] p,
        [data-testid="stToast"] span,
        [data-testid="stToast"] div {
            color: #FAFAFA !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            line-height: 1.4 !important;
        }
        [data-testid="stToast"] button {
            background-color: transparent !important;
            border: none !important;
            color: #A1A1AA !important;
        }
        [data-testid="stToast"] button svg {
            fill: #A1A1AA !important;
            stroke: #A1A1AA !important;
        }
        [data-testid="stToast"] button:hover svg {
            fill: #FAFAFA !important;
            stroke: #FAFAFA !important;
        }

        /* Alerts in Dark Mode */
        [data-testid="stAlert"], .stAlert {
            background-color: #18181B !important;
            border: 1px solid #27272A !important;
            color: #FAFAFA !important;
            border-radius: 8px !important;
        }
        [data-testid="stAlert"] * {
            color: #FAFAFA !important;
        }

        /* Tooltips & Vega Popovers in Dark Mode */
        div[role="tooltip"], .stTooltipContent {
            background-color: #18181B !important;
            border: 1px solid #27272A !important;
            color: #FAFAFA !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) !important;
            border-radius: 6px !important;
        }
        div[role="tooltip"] * {
            color: #FAFAFA !important;
        }
        #vg-tooltip-element {
            background-color: #18181B !important;
            border: 1px solid #27272A !important;
            color: #FAFAFA !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) !important;
            font-family: 'Inter', sans-serif !important;
            border-radius: 6px !important;
        }
        #vg-tooltip-element table tr td {
            color: #FAFAFA !important;
        }
        #vg-tooltip-element table tr td.key {
            color: #A1A1AA !important;
        }
    </style>
    """
else:
    chart_primary = "#18181B"    # Dark bar on light background
    chart_secondary = "#2A9D90"  # Teal
    chart_accent = "#E76E50"     # Terracotta Coral
    chart_line = "#18181B"       # Dark line

    theme_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }

        /* Sleek Modern Webkit Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #FFFFFF;
        }
        ::-webkit-scrollbar-thumb {
            background: #E4E4E7;
            border-radius: 9999px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #D4D4D8;
        }

        /* Pulsating Live Status Indicator */
        @keyframes pulse-dot {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.35; transform: scale(1.2); }
            100% { opacity: 1; transform: scale(1); }
        }
        .pulse-indicator {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #16A34A;
            margin-right: 6px;
            animation: pulse-dot 2s infinite ease-in-out;
            vertical-align: middle;
        }
        
        /* Light Canvas & Typography */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF !important;
            color: #09090B !important;
        }

        /* Remove massive empty top padding & pull page up */
        .block-container, [data-testid="stAppViewBlockContainer"] {
            padding-top: 1.25rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 98% !important;
        }
        [data-testid="stHeader"] {
            height: 0px !important;
            min-height: 0px !important;
            background: transparent !important;
            display: none !important;
        }
        
        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color: #09090B !important;
        }
        [data-testid="stMarkdownContainer"] > p {
            color: #09090B !important;
        }
        .main-title {
            font-size: 2.35rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.035em !important;
            color: #09090B !important;
            margin-bottom: 0.35rem !important;
            line-height: 1.2 !important;
        }
        .sub-title {
            font-size: 0.875rem !important;
            color: #71717A !important;
            margin-bottom: 1.25rem !important;
        }
        
        /* Light Mode Badges & Pills */
        span.tech-pill, [data-testid="stMarkdownContainer"] span.tech-pill {
            display: inline-block !important;
            padding: 0.2rem 0.55rem !important;
            margin-right: 0.35rem !important;
            margin-bottom: 0.3rem !important;
            border-radius: 6px !important;
            background-color: #F4F4F5 !important;
            color: #18181B !important;
            font-weight: 500 !important;
            font-size: 0.75rem !important;
            border: 1px solid #E4E4E7 !important;
        }
        span.tech-pill-accent, [data-testid="stMarkdownContainer"] span.tech-pill-accent {
            display: inline-block !important;
            padding: 0.2rem 0.55rem !important;
            margin-right: 0.35rem !important;
            margin-bottom: 0.3rem !important;
            border-radius: 6px !important;
            background-color: #18181B !important;
            color: #FAFAFA !important;
            font-weight: 600 !important;
            font-size: 0.75rem !important;
            border: 1px solid #18181B !important;
        }
        
        span.badge-high, [data-testid="stMarkdownContainer"] span.badge-high {
            display: inline-block !important;
            padding: 0.25rem 0.65rem !important;
            margin: 0.2rem !important;
            border-radius: 9999px !important;
            background-color: #18181B !important;
            color: #FAFAFA !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            border: 1px solid #18181B !important;
        }
        span.badge-med, [data-testid="stMarkdownContainer"] span.badge-med {
            display: inline-block !important;
            padding: 0.25rem 0.65rem !important;
            margin: 0.2rem !important;
            border-radius: 9999px !important;
            background-color: #F4F4F5 !important;
            color: #18181B !important;
            font-weight: 500 !important;
            font-size: 0.8rem !important;
            border: 1px solid #E4E4E7 !important;
        }
        span.badge-low, [data-testid="stMarkdownContainer"] span.badge-low {
            display: inline-block !important;
            padding: 0.25rem 0.65rem !important;
            margin: 0.2rem !important;
            border-radius: 9999px !important;
            background-color: #FFFFFF !important;
            color: #71717A !important;
            font-weight: 400 !important;
            font-size: 0.8rem !important;
            border: 1px solid #E4E4E7 !important;
        }

        /* Light Mode Card Containers */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 8px !important;
            border: 1px solid #E4E4E7 !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
            padding: 4px !important;
        }

        /* Light Mode KPI Metric Cards */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E4E4E7 !important;
            border-radius: 8px !important;
            padding: 12px 14px !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04) !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            color: #71717A !important;
            letter-spacing: -0.01em !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            color: #09090B !important;
        }

        /* Light Mode Navigation Tabs List */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #F4F4F5 !important;
            padding: 4px !important;
            border-radius: 8px !important;
            gap: 4px !important;
            border: 1px solid #E4E4E7 !important;
            margin-bottom: 1rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px !important;
            padding: 6px 14px !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: #71717A !important;
            border: none !important;
            background-color: transparent !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #09090B !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* Primary Button in Light Mode: Solid Dark with Pure White Text */
        .stButton > button,
        button[data-testid="baseButton-primary"],
        button[kind="primary"] {
            background-color: #18181B !important;
            border-radius: 6px !important;
            border: 1px solid #18181B !important;
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            padding: 0.45rem 1rem !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
            transition: all 0.15s ease-in-out !important;
            color: #FAFAFA !important;
        }
        .stButton > button *,
        button[data-testid="baseButton-primary"] *,
        button[kind="primary"] *,
        .stButton > button p,
        button[data-testid="baseButton-primary"] p,
        .stButton > button span,
        button[data-testid="baseButton-primary"] span {
            color: #FAFAFA !important;
            fill: #FAFAFA !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover,
        button[data-testid="baseButton-primary"]:hover,
        button[kind="primary"]:hover {
            background-color: #27272A !important;
            border-color: #27272A !important;
        }
        .stButton > button:hover *,
        button[data-testid="baseButton-primary"]:hover * {
            color: #FAFAFA !important;
        }

        /* Download / Secondary Button in Light Mode: Crisp White with Dark Text */
        .stDownloadButton > button,
        button[data-testid="baseButton-secondary"],
        button[kind="secondary"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E4E4E7 !important;
            border-radius: 6px !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: #18181B !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04) !important;
            transition: all 0.15s ease-in-out !important;
        }
        .stDownloadButton > button *,
        button[data-testid="baseButton-secondary"] *,
        .stDownloadButton > button p,
        button[data-testid="baseButton-secondary"] p,
        .stDownloadButton > button span,
        button[data-testid="baseButton-secondary"] span {
            color: #18181B !important;
            fill: #18181B !important;
            font-weight: 500 !important;
        }
        .stDownloadButton > button:hover,
        button[data-testid="baseButton-secondary"]:hover {
            background-color: #F4F4F5 !important;
            border-color: #D4D4D8 !important;
        }

        /* Inputs, Selects & Expanders in Light Mode */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        input {
            border-radius: 6px !important;
            border-color: #E4E4E7 !important;
            background-color: #FFFFFF !important;
            color: #09090B !important;
            font-size: 0.875rem !important;
        }
        div[data-baseweb="select"] * {
            color: #09090B !important;
        }
        div[data-baseweb="popover"],
        ul[role="listbox"],
        li[role="option"] {
            background-color: #FFFFFF !important;
            color: #09090B !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background-color: #F4F4F5 !important;
            color: #09090B !important;
        }

        /* Expanders in Light Mode */
        [data-testid="stExpander"] {
            border: 1px solid #E4E4E7 !important;
            border-radius: 8px !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03) !important;
            overflow: hidden !important;
        }
        [data-testid="stExpander"] details {
            background-color: #FFFFFF !important;
        }
        [data-testid="stExpander"] summary {
            background-color: #F4F4F5 !important;
            color: #09090B !important;
            border-radius: 6px !important;
            padding: 0.5rem 0.75rem !important;
        }
        [data-testid="stExpander"] summary:hover {
            background-color: #E4E4E7 !important;
            color: #09090B !important;
        }
        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {
            color: #09090B !important;
            fill: #09090B !important;
        }
        [data-testid="stExpander"] summary svg {
            fill: #09090B !important;
            stroke: #09090B !important;
        }
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: #FFFFFF !important;
            color: #09090B !important;
            border-top: 1px solid #E4E4E7 !important;
            padding: 0.75rem !important;
        }

        /* Radio & Toggle in Light Mode */
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label p,
        [data-testid="stRadio"] label span {
            color: #09090B !important;
        }
        div[data-testid="stToggle"] label p,
        div[data-testid="stToggle"] label span {
            color: #09090B !important;
            font-weight: 500 !important;
        }

        /* 2D Chemical Structure in Light Mode */
        [data-testid="stImage"] img {
            border-radius: 6px !important;
            filter: none !important;
            background-color: #FFFFFF !important;
        }

        /* Force Vega Lite chart container to match background */
        div[data-testid="stVegaLiteChart"],
        div[data-testid="stVegaLiteChart"] > div,
        div[data-testid="stVegaLiteChart"] svg,
        div[data-testid="stVegaLiteChart"] canvas {
            background-color: transparent !important;
        }

        /* Light Mode Progress Bar */
        [data-testid="stProgress"] > div > div > div > div {
            background-color: #18181B !important;
            border-radius: 9999px !important;
        }
        [data-testid="stProgress"] > div > div {
            background-color: #E4E4E7 !important;
            border-radius: 9999px !important;
        }

        /* Code in Light Mode */
        code {
            background-color: #F4F4F5 !important;
            color: #09090B !important;
            border: 1px solid #E4E4E7 !important;
            border-radius: 4px !important;
            padding: 0.15rem 0.35rem !important;
        }

        /* Micro-interactions & Polished Transitions */
        [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"] {
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: #D4D4D8 !important;
        }
        .stButton > button:active, .stDownloadButton > button:active {
            transform: scale(0.985) !important;
        }
        div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within {
            border-color: #0284C7 !important;
            box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.18) !important;
        }

        /* Deployment Footer in Light Mode */
        .deployment-footer {
            margin-top: 2.5rem;
            padding: 1.25rem 0.5rem;
            border-top: 1px solid #E4E4E7;
            font-size: 0.82rem;
            color: #71717A;
            font-family: 'Inter', -apple-system, sans-serif;
        }
        .developer-highlight {
            color: #09090B !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, rgba(2, 132, 199, 0.1), rgba(147, 51, 234, 0.1));
            border: 1px solid rgba(2, 132, 199, 0.35);
            padding: 2px 9px;
            border-radius: 5px;
            letter-spacing: -0.01em;
            display: inline-block;
        }

        /* Toast Notifications in Light Mode - Crisp White shadcn/ui Toast */
        [data-testid="stToast"], .stToast, div[data-testid="stToast"] {
            background-color: #FFFFFF !important;
            color: #09090B !important;
            border: 1px solid #E4E4E7 !important;
            border-radius: 8px !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.12), 0 8px 10px -6px rgba(0, 0, 0, 0.08) !important;
            padding: 0.75rem 1rem !important;
        }
        [data-testid="stToast"] [data-testid="stMarkdownContainer"],
        [data-testid="stToast"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stToast"] p,
        [data-testid="stToast"] span,
        [data-testid="stToast"] div {
            color: #09090B !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            line-height: 1.4 !important;
        }
        [data-testid="stToast"] button {
            background-color: transparent !important;
            border: none !important;
            color: #71717A !important;
        }
        [data-testid="stToast"] button svg {
            fill: #71717A !important;
            stroke: #71717A !important;
        }
        [data-testid="stToast"] button:hover svg {
            fill: #09090B !important;
            stroke: #09090B !important;
        }

        /* Alerts in Light Mode */
        [data-testid="stAlert"], .stAlert {
            background-color: #F4F4F5 !important;
            border: 1px solid #E4E4E7 !important;
            color: #09090B !important;
            border-radius: 8px !important;
        }
        [data-testid="stAlert"] * {
            color: #09090B !important;
        }

        /* Tooltips & Vega Popovers in Light Mode */
        div[role="tooltip"], .stTooltipContent {
            background-color: #FFFFFF !important;
            border: 1px solid #E4E4E7 !important;
            color: #09090B !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
            border-radius: 6px !important;
        }
        div[role="tooltip"] * {
            color: #09090B !important;
        }
        #vg-tooltip-element {
            background-color: #FFFFFF !important;
            border: 1px solid #E4E4E7 !important;
            color: #09090B !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
            font-family: 'Inter', sans-serif !important;
            border-radius: 6px !important;
        }
        #vg-tooltip-element table tr td {
            color: #09090B !important;
        }
        #vg-tooltip-element table tr td.key {
            color: #71717A !important;
        }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

# -------------------------------------------------------------
# ALTAIR CHART HELPERS (Zoomable, Scrollable, Point-Inspection)
# -------------------------------------------------------------
def make_bar_chart(df, x_col, y_col, is_dark_mode, bar_color=None, height=220, tooltips=None, y_title=None):
    bg = "#09090B" if is_dark_mode else "#FFFFFF"
    text_color = "#A1A1AA" if is_dark_mode else "#71717A"
    grid_color = "#27272A" if is_dark_mode else "#E4E4E7"
    if bar_color is None:
        bar_color = "#38BDF8" if is_dark_mode else "#0284C7"
    
    y_axis_kwargs = {
        "labelColor": text_color,
        "titleColor": text_color,
        "gridColor": grid_color,
        "domainColor": grid_color
    }
    if y_title:
        y_axis_kwargs["title"] = y_title

    enc = {
        "x": alt.X(f"{x_col}:N", sort=None, axis=alt.Axis(
            labelColor=text_color, 
            titleColor=text_color, 
            labelAngle=-30, 
            grid=False, 
            domainColor=grid_color
        )),
        "y": alt.Y(f"{y_col}:Q", axis=alt.Axis(**y_axis_kwargs))
    }
    if tooltips:
        enc["tooltip"] = tooltips
    else:
        enc["tooltip"] = [alt.Tooltip(f"{x_col}:N", title=x_col), alt.Tooltip(f"{y_col}:Q", format=".3f", title=y_title or y_col)]

    chart = alt.Chart(df).mark_bar(
        color=bar_color, 
        cornerRadiusTopLeft=4, 
        cornerRadiusTopRight=4
    ).encode(
        **enc
    ).properties(
        height=height, 
        background=bg
    ).configure_view(
        strokeOpacity=0
    ).configure(
        background=bg
    )
    return chart

def make_interactive_latent_chart(df, is_dark_mode, line_color=None, height=230):
    bg = "#09090B" if is_dark_mode else "#FFFFFF"
    text_color = "#A1A1AA" if is_dark_mode else "#71717A"
    grid_color = "#27272A" if is_dark_mode else "#E4E4E7"
    if line_color is None:
        line_color = "#38BDF8" if is_dark_mode else "#0284C7"
    point_color = "#F43F5E"  # Vibrant Rose for active cursor pinpoint
    
    # 1. Base Latent Line
    base_line = alt.Chart(df).mark_line(color=line_color, strokeWidth=2).encode(
        x=alt.X("DimIndex:Q", title="Latent Odor Vector Dimension (D0 → D255)", axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color, domainColor=grid_color)),
        y=alt.Y("Activation:Q", title="Neural Activation Level", axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color, domainColor=grid_color))
    )
    
    # 2. Nearest point selection on cursor hover
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["DimIndex"], empty=False)
    
    # 3. Transparent selector points along the curve
    selectors = alt.Chart(df).mark_point().encode(
        x="DimIndex:Q",
        opacity=alt.value(0)
    ).add_params(nearest)
    
    # 4. Point highlight circle that snaps exactly to the cursor
    points = alt.Chart(df).mark_circle(size=110, color=point_color).encode(
        x="DimIndex:Q",
        y="Activation:Q",
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("Dimension:N", title="Latent Feature"),
            alt.Tooltip("DimIndex:Q", title="Dimension Index"),
            alt.Tooltip("Activation:Q", format=".4f", title="Activation Value"),
            alt.Tooltip("Status:N", title="Feature Significance")
        ]
    )
    
    # 5. Crosshair vertical rule to pinpoint the cursor
    rules = alt.Chart(df).mark_rule(color="#71717A", strokeDash=[3, 3]).encode(
        x="DimIndex:Q",
        opacity=alt.condition(nearest, alt.value(0.65), alt.value(0))
    )
    
    chart = alt.layer(base_line, selectors, points, rules).properties(
        height=height, background=bg
    ).configure_view(strokeOpacity=0).configure(
        background=bg
    ).interactive()
    
    return chart

def make_scatter_chart(df, x_col, y_col, color_col, is_dark_mode, height=440, tooltips=None):
    bg = "#09090B" if is_dark_mode else "#FFFFFF"
    text_color = "#A1A1AA" if is_dark_mode else "#71717A"
    grid_color = "#27272A" if is_dark_mode else "#E4E4E7"
    
    if is_dark_mode:
        palette = [
            "#38BDF8",  # Sky Blue (Active Query)
            "#FB923C",  # Vibrant Orange (Citrus)
            "#F43F5E",  # Rose Red (Fruity)
            "#34D399",  # Emerald Mint (Mint / Cooling)
            "#A78BFA",  # Purple (Roasted / Coffee)
            "#FBBF24",  # Amber Gold (Spicy / Warm)
            "#F472B6",  # Pink (Sweet / Vanilla)
            "#94A3B8",  # Slate (Woody / Smoky)
        ]
        point_stroke = "#FFFFFF"
        stroke_opacity = 0.5
    else:
        # Deep, saturated, dark colors so points never look faded or washed out on light backgrounds
        palette = [
            "#09090B",  # Jet Black (Active Query - stands out boldly)
            "#C2410C",  # Deep Burnt Orange (Citrus)
            "#BE123C",  # Deep Crimson Red (Fruity)
            "#047857",  # Deep Forest Emerald (Mint / Cooling)
            "#581C87",  # Deep Royal Purple (Roasted / Coffee)
            "#B45309",  # Deep Dark Ochre Amber (Spicy / Warm)
            "#9D174D",  # Deep Magenta Wine (Sweet / Vanilla)
            "#3F3F46",  # Deep Charcoal Zinc (Woody / Smoky)
        ]
        point_stroke = "#09090B"
        stroke_opacity = 0.85

    # Conditional sizing: Active Query is extra large (250) vs reference clusters (130)
    if "Sample Type" in df.columns:
        size_enc = alt.condition(alt.datum["Sample Type"] == "Active Molecular Query", alt.value(250), alt.value(130))
    else:
        size_enc = alt.value(130)

    enc = {
        "x": alt.X(f"{x_col}:Q", axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color, domainColor=grid_color)),
        "y": alt.Y(f"{y_col}:Q", axis=alt.Axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color, domainColor=grid_color)),
        "color": alt.Color(
            f"{color_col}:N",
            scale=alt.Scale(range=palette),
            legend=alt.Legend(
                title=color_col,
                labelColor=text_color,
                titleColor=text_color,
                symbolSize=130,
                symbolOpacity=1.0
            )
        ),
        "size": size_enc
    }
    if tooltips:
        enc["tooltip"] = tooltips
    else:
        enc["tooltip"] = [
            alt.Tooltip(f"{color_col}:N", title="Cluster"),
            alt.Tooltip(f"{x_col}:Q", format=".2f", title=x_col),
            alt.Tooltip(f"{y_col}:Q", format=".2f", title=y_col)
        ]

    chart = alt.Chart(df).mark_circle(
        opacity=0.95,
        stroke=point_stroke,
        strokeWidth=1.2,
        strokeOpacity=stroke_opacity
    ).encode(
        **enc
    ).properties(height=height, background=bg).configure_view(strokeOpacity=0).configure(
        background=bg
    ).interactive()
    return chart

def render_styled_table(df, is_dark_mode, max_height=None):
    bg_header = "#18181B" if is_dark_mode else "#F4F4F5"
    text_header = "#FAFAFA" if is_dark_mode else "#18181B"
    bg_row1 = "#09090B" if is_dark_mode else "#FFFFFF"
    bg_row2 = "#121215" if is_dark_mode else "#FAFAFA"
    text_row = "#E4E4E7" if is_dark_mode else "#18181B"
    border_col = "#27272A" if is_dark_mode else "#E4E4E7"
    
    height_style = f"max-height: {max_height}px; overflow-y: auto;" if max_height else ""
    html = f"""<div style='border: 1px solid {border_col}; border-radius: 8px; overflow-x: auto; {height_style} margin: 8px 0;'>
    <table style='width: 100%; border-collapse: collapse; font-size: 0.83rem; font-family: Inter, -apple-system, sans-serif; background-color: {bg_row1};'>
        <thead>
            <tr style='background-color: {bg_header}; border-bottom: 1px solid {border_col}; position: sticky; top: 0; z-index: 1;'>"""
    for col in df.columns:
        html += f"<th style='padding: 9px 12px; text-align: left; font-weight: 600; color: {text_header}; font-size: 0.8rem; letter-spacing: -0.01em;'>{col}</th>"
    html += """</tr>
        </thead>
        <tbody>"""
    for i, (_, row) in enumerate(df.iterrows()):
        row_bg = bg_row2 if i % 2 == 1 else bg_row1
        html += f"<tr style='background-color: {row_bg}; border-bottom: 1px solid {border_col};'>"
        for val in row:
            html += f"<td style='padding: 8px 12px; color: {text_row}; font-size: 0.82rem;'>{val}</td>"
        html += "</tr>"
    html += """</tbody>
    </table>
    </div>"""
    return st.markdown(html, unsafe_allow_html=True)

# -------------------------------------------------------------
# MODEL CACHE & DATASET LOADER
# -------------------------------------------------------------
@st.cache_resource
def get_all_models():
    # 1. Electronic Nose Model
    nose_path = os.path.join(os.path.dirname(__file__), "nose_model.joblib")
    if not os.path.exists(nose_path):
        with st.spinner("Training electronic nose model..."):
            nose_bundle = train_nose(model_output_path=nose_path)
    else:
        nose_bundle = load_nose(nose_path)

    # 2. Baseline Smell Model (ECFP4 Fingerprints)
    smell_path = os.path.join(os.path.dirname(__file__), "smell_model.joblib")
    if not os.path.exists(smell_path):
        with st.spinner("Loading baseline smell model..."):
            smell_bundle = train_smell_model(save_path=smell_path)
    else:
        smell_bundle = load_smell_model(smell_path)

    # 3. Deep Molecular GNN (MPNN / NeuroScent)
    gnn_device = "mps" if torch.backends.mps.is_available() else "cpu"
    with st.spinner("Initializing NeuroScent-GNN Message Passing Engine..."):
        gnn_model, gnn_descriptors, gnn_auc = load_or_train_gnn(device=gnn_device, epochs=10)

    # 4. Samples library
    samples_path = os.path.join(os.path.dirname(__file__), "samples.json")
    with open(samples_path, "r") as f:
        samples = json.load(f)

    # 5. Pre-compute reference cluster embeddings for the 2D Principal Odor Map
    reference_clusters = {
        "Sweet / Vanilla": ["COc1cc(C=O)ccc1O", "CCOc1cc(C=O)ccc1O", "COc1cc(C(=O)C)ccc1O"],
        "Mint / Cooling": ["CC1CCC(C(C1)O)C(C)C", "CC1CCC(C(=O)C1)C(C)C", "CC1(C)C2CCC(O2)(C)CC1"],
        "Citrus": ["CC1=CCC(CC1)C(=C)C", "CC(=CCCC(=CC=O)C)C", "CC1=CCC(=CC1)C(C)C"],
        "Fruity": ["CC(C)CCOC(=O)C", "CCCC(=O)OCCC(C)C", "CCCCOC(=O)C", "CCCC(=O)OCC"],
        "Woody / Smoky": ["COc1ccccc1O", "COc1cc(cc(c1O)OC)", "Oc1ccccc1", "Cc1ccccc1O"],
        "Roasted / Coffee": ["SCC1=CC=CO1", "CCc1cncc(c1C)C", "COc1cc(C=C)ccc1O"],
        "Spicy / Warm": ["O=CC=Cc1ccccc1", "COc1cc(CC=C)ccc1O", "COc1cc(C=CC)ccc1O"]
    }

    ref_data = []
    for cat, smiles_list in reference_clusters.items():
        for s in smiles_list:
            _, emb = predict_gnn_smell(s, gnn_model, gnn_descriptors, device=gnn_device)
            if emb is not None:
                ref_data.append({"category": cat, "smiles": s, "embedding": emb})

    all_embs = np.array([item["embedding"] for item in ref_data])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_embs)

    for item in ref_data:
        coords = pca.transform(item["embedding"].reshape(1, -1))[0]
        item["x"] = float(coords[0])
        item["y"] = float(coords[1])

    # 6. Load Dataset stats live from CSV
    data_path = os.path.join(os.path.dirname(__file__), "curated_GS_LF_merged_4983.csv")
    df_raw = pd.read_csv(data_path)
    desc_cols = [c for c in df_raw.columns if c not in ["nonStereoSMILES", "descriptors"]]
    dataset_stats = {
        "num_compounds": len(df_raw),
        "num_descriptors": len(desc_cols),
        "avg_labels_per_mol": float(df_raw[desc_cols].sum(axis=1).mean()),
        "top_descriptors": df_raw[desc_cols].sum(axis=0).sort_values(ascending=False).head(10).to_dict(),
        "total_labels": int(df_raw[desc_cols].sum().sum())
    }

    return nose_bundle, smell_bundle, gnn_model, gnn_descriptors, gnn_auc, samples, ref_data, pca, dataset_stats

try:
    nose_bundle, smell_bundle, gnn_model, gnn_descriptors, gnn_auc, samples, ref_data, pca, dataset_stats = get_all_models()
except Exception as e:
    st.error(f"Error initializing models: {e}")
    st.stop()

# Header with Dynamic Real-Time Indicators and Theme Toggle
col_title, col_toggle = st.columns([10, 2], vertical_alignment="center")
with col_title:
    st.markdown('<div class="main-title">⚡ NeuroScent-GNN: Deep Molecular Olfactory Intelligence</div>', unsafe_allow_html=True)
with col_toggle:
    dark_mode_val = st.toggle("Dark Mode", value=st.session_state["dark_mode"], key="theme_toggle_key")
    if dark_mode_val != st.session_state["dark_mode"]:
        st.session_state["dark_mode"] = dark_mode_val
        st.rerun()

st.markdown(f"""
<div class="sub-title">
    <span class="tech-pill-accent"><span class="pulse-indicator"></span>NeuroScent Engine: Active</span>
    <span class="tech-pill">Apple Silicon M4 Series (10-Core GPU)</span>
    <span class="tech-pill">PyTorch Geometric (GINEConv)</span>
    <span class="tech-pill">256-D Latent Odor Map</span>
    <span class="tech-pill">{dataset_stats['num_compounds']:,} Molecules × {dataset_stats['num_descriptors']} Tasks</span>
    <span class="tech-pill">Multi-Task Test ROC-AUC: <b>{gnn_auc:.4f}</b></span>
    <span class="tech-pill">E-Nose Classifier Accuracy: <b>{nose_bundle['accuracy']*100:.1f}%</b></span>
</div>
""", unsafe_allow_html=True)

# 4 Main Organized Tabs
tab_screener, tab_latent, tab_enose, tab_audit = st.tabs([
    "1. In Silico Molecular Screener",
    "2. Principal Odor Map (2D Manifold)",
    "3. Electronic Gas Sensor Array",
    "4. User Guide & Live Model Audit"
])

# =============================================================
# TAB 1: IN SILICO MOLECULAR SCREENER
# =============================================================
with tab_screener:
    presets = {
        "Vanillin (Sweet / Vanilla)": "COc1cc(C=O)ccc1O",
        "L-Menthol (Cooling / Mint)": "CC1CCC(C(C1)O)C(C)C",
        "Isoamyl Acetate (Banana / Fruity)": "CC(C)CCOC(=O)C",
        "Limonene (Citrus / Orange)": "CC1=CCC(CC1)C(=C)C",
        "Guaiacol (Smoky / Phenolic)": "COc1ccccc1O",
        "Cinnamaldehyde (Spicy / Cinnamon)": "O=CC=Cc1ccccc1",
        "Furfuryl Mercaptan (Coffee / Roasted)": "SCC1=CC=CO1",
        "Eugenol (Clove / Warm)": "COc1cc(CC=C)ccc1O",
        "Geosmin (Earthy / Petrichor)": "CC1CCC2(C)CCCC(C)(O)C2C1",
        "Damascenone (Floral / Rose)": "CC1=C(C(CC(C1)(C)C)C=CC(=O)C)C",
        "Ethanol (Alcoholic)": "CCO"
    }

    # Consolidated 1-Card Header: Input, 2D Structure, and Key Physicochemical KPIs
    with st.container(border=True):
        col_in, col_struct, col_props = st.columns([11, 7, 10], gap="large")

        with col_in:
            st.subheader("Chemical Input")
            selected_preset = st.selectbox("Benchmark Presets:", list(presets.keys()))
            default_smiles = presets[selected_preset]
            input_smiles = st.text_input("Molecular SMILES String (Live Input):", value=default_smiles)

            c_eng, c_top = st.columns([3, 2])
            with c_eng:
                model_engine = st.selectbox("Engine:", ["Deep GNN (MPNN)", "Baseline ECFP4"])
            with c_top:
                top_k = st.slider("Top Descriptors:", min_value=3, max_value=12, value=6)

        mol = Chem.MolFromSmiles(input_smiles)

        if mol is not None:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            g_data = mol_to_graph(mol)
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()

            with col_struct:
                st.subheader("2D Structure")
                img = Draw.MolToImage(mol, size=(240, 150))
                st.image(img, use_container_width=True)
                formula_str = rdMolDescriptors.CalcMolFormula(mol)
                st.caption(f"**Formula:** `{formula_str}`")

            with col_props:
                st.subheader("Physicochemical KPIs")
                m1, m2, m3 = st.columns(3)
                m1.metric("Mol Wt", f"{mw:.1f} Da")
                m2.metric("LogP", f"{logp:.2f}")
                m3.metric("TPSA", f"{tpsa:.1f} Å²")

                m4, m5, m6 = st.columns(3)
                m4.metric("H-Bond (D/A)", f"{hbd}/{hba}")
                m5.metric("Atoms", f"{num_atoms}")
                m6.metric("Bonds", f"{num_bonds}")
        else:
            with col_struct:
                st.subheader("2D Structure")
                st.error("Invalid SMILES structure. Please verify notation.")
            with col_props:
                st.subheader("Physicochemical KPIs")
                st.info("Awaiting valid chemical structure...")

    if mol is not None:
        # Run Live GNN Inference
        gnn_preds, pom_embedding = predict_gnn_smell(input_smiles, gnn_model, gnn_descriptors, device=gnn_device, top_n=top_k)

        # Tabbed Sub-navigation to eliminate long scrolling
        subtab_screener, subtab_tensors = st.tabs([
            "Primary Odor Profile & Latent Manifold",
            "Molecular Graph Tensors & 138-Descriptor Search"
        ])

        with subtab_screener:
            col_res, col_pom = st.columns([1, 1], gap="large")

            with col_res:
                with st.container(border=True):
                    if model_engine.startswith("Deep GNN"):
                        st.subheader("Deep GNN Odor Predictions")
                        badge_html = ""
                        for desc, prob in gnn_preds:
                            css_class = "badge-high" if prob >= 0.50 else ("badge-med" if prob >= 0.30 else "badge-low")
                            badge_html += f'<span class="{css_class}">{desc.title()} ({prob*100:.1f}%)</span>'
                        st.markdown(badge_html, unsafe_allow_html=True)

                        res_df = pd.DataFrame(gnn_preds, columns=["Descriptor", "Probability"])
                        res_df["Note Classification"] = res_df["Probability"].apply(
                            lambda p: "Dominant Primary Note" if p >= 0.50 else ("Secondary Note" if p >= 0.30 else "Trace Sub-note")
                        )
                        tooltips_gnn = [
                            alt.Tooltip("Descriptor:N", title="Odor Descriptor"),
                            alt.Tooltip("Probability:Q", format=".1%", title="Model Probability"),
                            alt.Tooltip("Note Classification:N", title="Classification")
                        ]
                        chart_gnn = make_bar_chart(
                            res_df, 
                            "Descriptor", 
                            "Probability", 
                            is_dark, 
                            bar_color="#38BDF8" if is_dark else "#0284C7", 
                            height=220, 
                            tooltips=tooltips_gnn,
                            y_title="Prediction Probability"
                        )
                        st.altair_chart(chart_gnn, theme=None, use_container_width=True)
                    else:
                        st.subheader("Baseline ECFP4 Odor Predictions")
                        ecfp_preds = predict_molecule_descriptors(input_smiles, smell_bundle, top_n=top_k)
                        res_df = pd.DataFrame(ecfp_preds, columns=["Descriptor", "Probability"])
                        res_df["Note Classification"] = res_df["Probability"].apply(
                            lambda p: "Dominant Primary Note" if p >= 0.50 else ("Secondary Note" if p >= 0.30 else "Trace Sub-note")
                        )
                        tooltips_ecfp = [
                            alt.Tooltip("Descriptor:N", title="Odor Descriptor"),
                            alt.Tooltip("Probability:Q", format=".1%", title="Baseline Probability"),
                            alt.Tooltip("Note Classification:N", title="Classification")
                        ]
                        chart_ecfp = make_bar_chart(
                            res_df, 
                            "Descriptor", 
                            "Probability", 
                            is_dark, 
                            bar_color="#10B981" if is_dark else "#059669", 
                            height=220, 
                            tooltips=tooltips_ecfp,
                            y_title="Baseline Probability"
                        )
                        st.altair_chart(chart_ecfp, theme=None, use_container_width=True)

            with col_pom:
                with st.container(border=True):
                    st.subheader("256-D Principal Odor Map Latent Vector")
                    l2_norm = float(np.linalg.norm(pom_embedding))
                    top_dim = int(np.argmax(pom_embedding))
                    max_act = float(pom_embedding[top_dim])

                    st.caption(
                        f"GNN bottleneck vector. "
                        f"**L2 Norm:** `{l2_norm:.2f}` | **Dominant:** `D{top_dim}` (`{max_act:.2f}`). "
                        f"*(Hover curve for dimensions; scroll to zoom, drag to pan)*"
                    )
                    latent_df = pd.DataFrame({
                        "DimIndex": list(range(len(pom_embedding))),
                        "Dimension": [f"D{i}" for i in range(len(pom_embedding))],
                        "Activation": pom_embedding.astype(float),
                        "Status": [
                            "Major Positive Driver" if val >= 0.8 else ("Moderate Active Feature" if val >= 0.3 else ("Suppression / Negative Inhibitor" if val <= -0.5 else "Baseline Feature"))
                            for val in pom_embedding
                        ]
                    })
                    chart_latent = make_interactive_latent_chart(latent_df, is_dark, line_color="#38BDF8" if is_dark else "#0284C7", height=195)
                    st.altair_chart(chart_latent, theme=None, use_container_width=True)

                    # Export button right under the latent map
                    export_payload = {
                        "smiles": input_smiles,
                        "formula": rdMolDescriptors.CalcMolFormula(mol),
                        "molecular_weight": mw,
                        "logp": logp,
                        "tpsa": tpsa,
                        "h_bond_donors": hbd,
                        "h_bond_acceptors": hba,
                        "top_predictions": [{"descriptor": d, "probability": p} for d, p in gnn_preds],
                        "pom_latent_vector_256d": pom_embedding.tolist()
                    }
                    st.download_button(
                        "Export Screening Report (JSON)",
                        data=json.dumps(export_payload, indent=2),
                        file_name=f"neuroscent_report_{input_smiles[:10]}.json",
                        mime="application/json",
                        use_container_width=True
                    )

        with subtab_tensors:
            col_search, col_tensors = st.columns([1, 1], gap="large")

            with col_search:
                with st.container(border=True):
                    st.subheader("Search All 138 Olfactory Descriptors (Live)")
                    query_term = st.text_input("Filter descriptors (e.g., 'sweet', 'smoke', 'rose', 'leather'):", "").strip().lower()
                    all_preds, _ = predict_gnn_smell(input_smiles, gnn_model, gnn_descriptors, device=gnn_device, top_n=len(gnn_descriptors))
                    filtered = [item for item in all_preds if query_term in item[0].lower()] if query_term else all_preds[:12]
                    df_filtered = pd.DataFrame(filtered, columns=["Odor Descriptor", "Predicted Probability"])
                    df_filtered["Predicted Probability"] = df_filtered["Predicted Probability"].apply(lambda p: f"{p*100:.2f}%")
                    render_styled_table(df_filtered, is_dark, max_height=280)

            with col_tensors:
                with st.container(border=True):
                    st.subheader("Molecular Graph Tensor Inspector")
                    tensor_atoms, tensor_bonds = st.tabs([
                        f"Atom Node Tensor (N={g_data.x.shape[0]})",
                        f"Bond Edge Tensor (M={g_data.edge_index.shape[1]})"
                    ])
                    with tensor_atoms:
                        atom_rows = []
                        for a in mol.GetAtoms():
                            atom_rows.append({
                                "Idx": a.GetIdx(),
                                "Symbol": a.GetSymbol(),
                                "AtomicNum": a.GetAtomicNum(),
                                "Degree": a.GetTotalDegree(),
                                "Hybridization": str(a.GetHybridization()).replace("HybridizationType.", ""),
                                "Aromatic": "Yes" if a.GetIsAromatic() else "No"
                            })
                        render_styled_table(pd.DataFrame(atom_rows), is_dark, max_height=245)
                    with tensor_bonds:
                        bond_rows = []
                        for b in mol.GetBonds():
                            bond_rows.append({
                                "Bond": f"{b.GetBeginAtomIdx()} ↔ {b.GetEndAtomIdx()}",
                                "Type": str(b.GetBondType()),
                                "Conjugated": "Yes" if b.GetIsConjugated() else "No",
                                "InRing": "Yes" if b.IsInRing() else "No"
                            })
                        render_styled_table(pd.DataFrame(bond_rows), is_dark, max_height=245)

# =============================================================
# TAB 2: PRINCIPAL ODOR MAP (2D LATENT MANIFOLD)
# =============================================================
with tab_latent:
    with st.container(border=True):
        st.subheader("Principal Odor Map: Latent Perceptual Space")
        var_pc1 = pca.explained_variance_ratio_[0] * 100
        var_pc2 = pca.explained_variance_ratio_[1] * 100
        total_var = var_pc1 + var_pc2

        st.write(
            f"Projection of the 256-dimensional neural latent space into a 2D perceptual manifold. "
            f"**PCA Explained Variance:** PC1 = `{var_pc1:.1f}%`, PC2 = `{var_pc2:.1f}%` (Total = `{total_var:.1f}%`). "
            f"Molecules with similar scents cluster together in this coordinate space. "
            f"*(Chart is fully interactive: scroll/pinch to zoom into clusters, click and drag to pan across the manifold)*"
        )

        plot_rows = []
        for item in ref_data:
            plot_rows.append({
                "Olfactory Cluster": item["category"],
                "Compound SMILES": item["smiles"],
                "PC1": float(item["x"]),
                "PC2": float(item["y"]),
                "Sample Type": "Reference Standard"
            })

        has_query = mol is not None and "pom_embedding" in locals() and pom_embedding is not None
        if has_query:
            query_coords = pca.transform(pom_embedding.reshape(1, -1))[0]
            plot_rows.append({
                "Olfactory Cluster": f"ACTIVE QUERY: {input_smiles[:18]}",
                "Compound SMILES": input_smiles,
                "PC1": float(query_coords[0]),
                "PC2": float(query_coords[1]),
                "Sample Type": "Active Molecular Query"
            })

        df_map = pd.DataFrame(plot_rows)
        tooltips_scatter = [
            alt.Tooltip("Olfactory Cluster:N", title="Olfactory Cluster"),
            alt.Tooltip("Compound SMILES:N", title="Chemical SMILES"),
            alt.Tooltip("PC1:Q", format=".2f", title="PC1 Coordinate"),
            alt.Tooltip("PC2:Q", format=".2f", title="PC2 Coordinate"),
            alt.Tooltip("Sample Type:N", title="Sample Identity")
        ]
        chart_scatter = make_scatter_chart(df_map, "PC1", "PC2", "Olfactory Cluster", is_dark, height=450, tooltips=tooltips_scatter)
        st.altair_chart(chart_scatter, theme=None, use_container_width=True)

        if has_query:
            # Live Nearest Neighbors in 256-D Space
            st.subheader("Nearest Olfactory Neighbors in 256-D Latent Space")
            dist_list = []
            for item in ref_data:
                d = float(np.linalg.norm(pom_embedding - item["embedding"]))
                dist_list.append({
                    "Olfactory Category": item["category"],
                    "Compound SMILES": item["smiles"],
                    "Distance (256-D)": round(d, 3)
                })
            df_dist = pd.DataFrame(dist_list).sort_values("Distance (256-D)").head(5)
            render_styled_table(df_dist, is_dark)
        else:
            st.info("Input a valid chemical SMILES string in Tab 1 to plot your active query molecule on this perceptual manifold.")

# =============================================================
# TAB 3: ELECTRONIC GAS SENSOR ARRAY
# =============================================================
with tab_enose:
    sensor = GasSensorArray()
    with st.container(border=True):
        col_ctrl, col_display = st.columns([1, 2], gap="large")

        with col_ctrl:
            st.subheader("1. Sample Configuration")
            blend_mode = st.radio("Aroma Source:", ["Preset Benchmark Aroma", "Custom Aroma Blend Studio"])

            if blend_mode == "Preset Benchmark Aroma":
                sample_options = list(samples.keys())
                selected_sample = st.selectbox(
                    "Benchmark Aroma:",
                    sample_options,
                    format_func=lambda x: f"{x.title()} (Blend of {len(samples[x])} aroma molecules)"
                )
                active_molecules = samples[selected_sample]
            else:
                st.markdown("**Custom Aroma Blend Studio:**")
                custom_s1 = st.text_input("Constituent 1 SMILES (e.g. Vanillin):", value="COc1cc(C=O)ccc1O")
                w1 = st.slider("Constituent 1 Concentration:", 0.1, 1.0, 0.6, 0.1)
                custom_s2 = st.text_input("Constituent 2 SMILES (e.g. Menthol):", value="CC1CCC(C(C1)O)C(C)C")
                w2 = st.slider("Constituent 2 Concentration:", 0.1, 1.0, 0.4, 0.1)

                active_molecules = [
                    {"name": "Constituent 1", "smiles": custom_s1, "weight": w1},
                    {"name": "Constituent 2", "smiles": custom_s2, "weight": w2}
                ]

            st.subheader("2. Ambient Atmosphere")
            humidity = st.slider("Relative Humidity (%)", min_value=20, max_value=80, value=50, step=5) / 100.0
            noise_level = st.slider("Sensor Noise (Std Dev)", min_value=0.01, max_value=0.10, value=0.04, step=0.01)

            sniff_clicked = st.button("Press Sniff", type="primary", use_container_width=True)

            with st.expander("View Active Chemical Constituents"):
                for m in active_molecules:
                    st.markdown(f"- **{m.get('name', 'Compound')}** (Weight: {m.get('weight', 1.0)})  \n  `{m['smiles']}`")

        with col_display:
            # 100% Real-Time Physical Sensor Simulation & Identification
            # Evaluated dynamically on every aroma select, slider change, and Sniff button press!
            readings = sensor.sample_response(
                active_molecules,
                humidity=humidity,
                noise_level=noise_level
            )
            predicted_smell, confidence, prob_dict = predict_nose(readings, nose_bundle)
            mixture_descriptors, _ = predict_gnn_mixture(active_molecules, gnn_model, gnn_descriptors, device=gnn_device, top_n=6)

            if sniff_clicked:
                st.toast("Physical electronic gas sensor sniff captured in real time.")

            st.subheader("3. Gas Sensor Reaction (8 Channels)")
            st.caption(r"Live differential voltage ($\Delta V$) across 8 specialized chemical sensing channels. *(Hover over any bar for channel target & response; scroll/pinch to zoom, drag to pan)*")

            channel_descriptions = {
                "S1_Volatiles": "Low MW Volatiles (Vapor Pressure / Volatility)",
                "S2_Lipophilic": "High LogP Non-Polar / Lipophilic Chains",
                "S3_Oxygenated": "Oxygenated Groups (Alcohols, Carbonyls, TPSA)",
                "S4_Sulfur": "Sulfur Heteroatoms (Thiols, Mercaptans, Sulfides)",
                "S5_Aromatic": "Aromatic & Phenolic Rings (Pyrazines, Benzene)",
                "S6_Terpenic": "Terpenic Hydrocarbons & Cyclic Monoterpenes",
                "S7_Esters": "Esters & Fragrance Esters",
                "S8_Broadband": "Broadband General VOC Aggregate Signal"
            }

            def signal_intensity(val):
                if val >= 2.5: return "Dominant Peak"
                if val >= 1.5: return "High Signal"
                if val >= 0.8: return "Moderate Activity"
                return "Baseline / Low Signal"

            sensor_df = pd.DataFrame({
                "Sensor Channel": sensor.sensor_names,
                "Response": [round(float(r), 3) for r in readings],
                "Chemical Target": [channel_descriptions.get(s, "VOC Signal") for s in sensor.sensor_names],
                "Signal Level": [signal_intensity(r) for r in readings]
            })

            tooltips_sensor = [
                alt.Tooltip("Sensor Channel:N", title="Sensor Channel"),
                alt.Tooltip("Chemical Target:N", title="Target Selectivity"),
                alt.Tooltip("Response:Q", format=".3f", title="Voltage Differential (ΔV)"),
                alt.Tooltip("Signal Level:N", title="Response Status")
            ]

            chart_sensor = make_bar_chart(
                sensor_df, 
                "Sensor Channel", 
                "Response", 
                is_dark, 
                bar_color="#38BDF8" if is_dark else "#0284C7", 
                height=220,
                tooltips=tooltips_sensor,
                y_title="Differential Voltage (ΔV / a.u.)"
            )
            st.altair_chart(chart_sensor, theme=None, use_container_width=True)

            st.subheader("4. Electronic Nose Identification")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.metric("Identified Sample Profile", predicted_smell.title())
            with c2:
                st.metric("Identification Confidence", f"{confidence * 100:.1f}%")

            st.progress(min(1.0, float(confidence)))

            # Full confidence distribution
            with st.expander("Full Classifier Probability Distribution"):
                df_probs = pd.DataFrame(
                    [{"Sample Class": k.title(), "Probability": f"{v * 100:.2f}%"} for k, v in prob_dict.items()]
                ).sort_values("Probability", ascending=False)
                render_styled_table(df_probs, is_dark)

            st.subheader("5. Deep GNN Predicted Odor Descriptors for this Blend")
            badge_html = "".join([f'<span class="badge-high">{desc.title()} ({prob*100:.1f}%)</span>' for desc, prob in mixture_descriptors])
            st.markdown(badge_html, unsafe_allow_html=True)

            blend_df = pd.DataFrame(mixture_descriptors, columns=["Odor Descriptor", "Probability"])
            blend_df["Note Strength"] = blend_df["Probability"].apply(
                lambda p: "Dominant Note" if p >= 0.50 else ("Secondary Note" if p >= 0.30 else "Sub-threshold Note")
            )
            tooltips_blend = [
                alt.Tooltip("Odor Descriptor:N", title="Odor Note"),
                alt.Tooltip("Probability:Q", format=".1%", title="GNN Probability"),
                alt.Tooltip("Note Strength:N", title="Strength")
            ]
            chart_blend = make_bar_chart(
                blend_df, 
                "Odor Descriptor", 
                "Probability", 
                is_dark, 
                bar_color="#F59E0B" if is_dark else "#D97706", 
                height=220,
                tooltips=tooltips_blend,
                y_title="Predicted Probability"
            )
            st.altair_chart(chart_blend, theme=None, use_container_width=True)

# =============================================================
# TAB 4: USER GUIDE & LIVE MODEL AUDIT
# =============================================================
with tab_audit:
    with st.expander("Step-by-Step Guide: How to Use This App (Click to Expand)", expanded=False):
        st.caption("A simple, clear workflow to screen odorants, explore olfactory latent space, and simulate electronic nose reactions:")

        g1, g2 = st.columns(2, gap="large")
        with g1:
            st.markdown("""
            **Step 1: Screen Chemical Molecules (Tab 1)**
            - Type any SMILES chemical notation or select a benchmark molecule (e.g., Vanillin, Eugenol, Citronellol) from the dropdown.
            - Toggle between the **Deep GNN (GINEConv)** or **Baseline ECFP4** prediction engine.
            - Instantly inspect the 2D molecular diagram, top 6 predicted odor notes with confidence %, and hover over points on the 256-D Principal Odor Map latent vector.
            - Click **Download Inference Report** to export predictions as JSON.

            **Step 2: Explore the Principal Odor Map (Tab 2)**
            - Visualize where your molecule is positioned in 2D perceptual space relative to 7 reference aroma families (Sweet, Minty, Citrus, Fruity, Woody, Roasted, Spicy).
            - Scroll or pinch to zoom into dense clusters, click and drag to pan, and point at any node to inspect coordinates.
            """)
        with g2:
            st.markdown(r"""
            **Step 3: Simulate Electronic Gas Sensors (Tab 3)**
            - Select a preset commercial aroma (e.g., Vanilla Candle, Coffee, Lemon, Wood Smoke) or build a custom aroma mixture in the Studio.
            - Adjust **Relative Humidity (%)** and **Sensor Noise** to simulate ambient environmental factors.
            - Click **Press Sniff!** to capture the 8-channel physical differential voltages ($\Delta V$), classify the aroma identity with the E-Nose, and compute GNN blend notes.

            **Step 4: Audit Model & Dataset Metrics (Tab 4)**
            - Review live trainable weights, multi-task test ROC-AUC performance (0.8583), 4,983-compound dataset label frequencies, and Apple Silicon M4 10-Core GPU acceleration telemetry.
            """)

    with st.container(border=True):
        st.subheader("Live System & Model Introspection")
        total_trainable_params = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)
        ckpt_path = os.path.join(os.path.dirname(__file__), "gnn_pom_model.pt")
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024) if os.path.exists(ckpt_path) else 0.0

        c_a1, c_a2, c_a3, c_a4 = st.columns(4)
        c_a1.metric("GNN Trainable Parameters", f"{total_trainable_params:,}")
        c_a2.metric("Multi-Task Test ROC-AUC", f"{gnn_auc:.4f}")
        c_a3.metric("Curated Compounds", f"{dataset_stats['num_compounds']:,}")
        c_a4.metric("Active Olfactory Tasks", f"{dataset_stats['num_descriptors']}")

    with st.container(border=True):
        st.subheader("Dataset Structure & Label Density (Live from CSV)")
        col_d1, col_d2 = st.columns([1, 1], gap="large")

        with col_d1:
            st.markdown(f"""
            - **Dataset Source:** Curated GoodScents & Leffingwell Datasets (NeuroScent)
            - **Total Labeled Compound Instances:** `{dataset_stats['num_compounds']:,}`
            - **Total Olfactory Annotations:** `{dataset_stats['total_labels']:,}`
            - **Average Descriptors per Molecule:** `{dataset_stats['avg_labels_per_mol']:.2f}`
            - **Checkpoint Size on Disk:** `{ckpt_size_mb:.2f} MB`
            """)

        with col_d2:
            tab_top_chart, tab_top_table = st.tabs(["Top Descriptors Chart", "Frequency Data Table"])
            with tab_top_chart:
                top_desc_chart_df = pd.DataFrame([
                    {
                        "Descriptor": k.title(),
                        "Annotated Compounds": int(v),
                        "Prevalence (% of Dataset)": f"{(int(v) / dataset_stats['num_compounds']) * 100:.1f}%"
                    }
                    for k, v in dataset_stats["top_descriptors"].items()
                ])
                tooltips_d = [
                    alt.Tooltip("Descriptor:N", title="Odor Descriptor"),
                    alt.Tooltip("Annotated Compounds:Q", format=",", title="Total Molecules"),
                    alt.Tooltip("Prevalence (% of Dataset):N", title="Prevalence Ratio")
                ]
                chart_top_desc = make_bar_chart(
                    top_desc_chart_df,
                    "Descriptor",
                    "Annotated Compounds",
                    is_dark,
                    bar_color="#10B981" if is_dark else "#059669",
                    height=165,
                    tooltips=tooltips_d
                )
                st.altair_chart(chart_top_desc, theme=None, use_container_width=True)

            with tab_top_table:
                top_desc_df = pd.DataFrame(
                    [{"Descriptor": k.title(), "Occurrence Count": f"{int(v):,}"} for k, v in dataset_stats["top_descriptors"].items()]
                )
                render_styled_table(top_desc_df, is_dark, max_height=165)

        # Placed directly under the Dataset Structure inside this container to fit everything on one page
        st.markdown("<hr style='margin: 1.1rem 0 0.8rem 0; border: none; border-top: 1px solid #27272A;' />", unsafe_allow_html=True)
        st.subheader("Software & Hardware Acceleration Environment")
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("PyTorch Version", torch.__version__)
        e2.metric("PyTorch Geometric", torch_geometric.__version__)
        e3.metric("RDKit Version", rdkit.__version__)
        e4.metric("Hardware Acceleration", "Apple Silicon M4 Series")
        e5.metric("GPU Cores & Backend", "10-Core GPU (MPS)")

# Modern Deployment Footer
st.markdown("""
<div class="deployment-footer">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div style="font-size: 0.84rem;">
            <b>NeuroScent-GNN</b> • Autonomous Molecular Olfactory Intelligence & E-Nose Platform
        </div>
        <div style="font-size: 0.84rem;">
            Developed by <span class="developer-highlight">Brijesh Rath</span> • All Rights Reserved © 2026
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
