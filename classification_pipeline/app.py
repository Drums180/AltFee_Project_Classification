import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from collections import Counter
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from urllib.parse import quote


st.set_page_config(
    page_title="Historical Matter Dashboard",
    page_icon="📊",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .main {
        background-color: #f7f8fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .hero-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #374151 100%);
        color: white;
        padding: 2rem;
        border-radius: 22px;
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
    }

    .hero-card h1 {
        margin-bottom: 0.35rem;
        font-size: 2.1rem;
        font-weight: 750;
    }

    .hero-card p {
        color: #d1d5db;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 1.35rem;
        box-shadow: 0 8px 26px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.2rem;
    }

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        min-height: 125px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.85rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        color: #111827;
        font-size: 1.75rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }

    .metric-help {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.92rem;
    }

    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 18px;
        padding: 0.75rem;
    }

    .ranking-card {
        background-color: #ffffff;
        border: 1px solid #D9D9D9;
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        min-height: 360px;
        display: flex;
        flex-direction: column;
        flex: 1;
        height: 100%;
    }

    .ranking-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #1F1F1F;
        margin-bottom: 0.9rem;
    }

    .ranking-row {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        padding: 0.55rem 0;
        border-bottom: 1px solid #EAEAEA;
        color: #1F1F1F;
        font-size: 0.95rem;
    }

    .ranking-row:last-child {
        border-bottom: none;
    }

    .ranking-name {
        max-width: 68%;
    }

    .ranking-value {
        color: #7B7B7B;
        white-space: nowrap;
    }

    .ranking-bold-black {
        font-weight: 900;
        color: #1F1F1F;
    }

    .ranking-bold-red {
        font-weight: 900;
        color: #D95F5F;
    }

    .ranking-footnote {
        color: #7B7B7B;
        font-size: 0.8rem;
        margin-top: 0.8rem;
    }

    .ranking-number-1 {
        font-weight: 900;
        color: #54B39A;
    }

    .ranking-number-2 {
        font-weight: 900;
        color: #F28C4C;
    }

    .ranking-number-3 {
        font-weight: 900;
        color: #4E79A7;
    }

    .folio-project-header {
        text-align: center;
        margin: 1.1rem 0 1.2rem;
        padding: 0.35rem 1rem 0.65rem;
    }

    .folio-project-heading {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-bottom: 0.45rem;
    }

    .folio-project-heading h3 {
        color: #1F1F1F;
        font-size: 1.7rem;
        font-weight: 850;
        margin: 0;
    }

    .folio-plus {
        color: #2C8576;
        font-weight: 950;
        font-size: 1.05em;
    }

    .folio-verified-text {
        color: #2C8576;
        font-size: 0.9rem;
        font-weight: 900;
    }

    .folio-project-title {
        color: #2C8576;
        font-size: 0.95rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .folio-project-description {
        color: #4B5563;
        font-size: 1rem;
        line-height: 1.5;
        margin: 0 auto 0.25rem;
        max-width: 820px;
    }

    .folio-project-meta {
        color: #2C8576;
        font-size: 0.9rem;
        font-weight: 900;
    }
</style>
"""

ALTFEE_TEAL = "#54B39A"
ALTFEE_TEAL_DARK = "#2C8576"
ALTFEE_TEAL_SOFT = "#CFEAE2"
ALTFEE_ORANGE = "#F28C4C"
ALTFEE_ORANGE_SOFT = "#EFC7A6"
ALTFEE_TEXT = "#1F1F1F"
ALTFEE_TEXT_SOFT = "#7B7B7B"
ALTFEE_BORDER = "#D9D9D9"
ALTFEE_GRID = "#EAEAEA"
ALTFEE_BG = "#FFFFFF"
ALTFEE_GRAY = "#B8B8B8"
ALTFEE_GRAY_DARK = "#707070"
ALTFEE_RED = "#D95F5F"
INDICATOR_GOOD = "#4CAF7D"
INDICATOR_OK = "#EDC948"
INDICATOR_BAD = "#D95F5F"

PROJECT_BASE_COLORS = [
    "#54B39A",
    "#F28C4C",
    "#4E79A7",
    "#A05EB5",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
]

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


REQUIRED_COLUMN_OPTIONS = {
    "account_name": ["account_name", "account", "firm_name", "company_name", "account_id"],
    "matter_id": ["matter_id", "clio_matter_id", "local_matter_id", "id"],
    "matter_name": ["matter_name", "name", "project_name", "description"],
    "date": ["open_date", "date", "matter_open_date", "created_at", "first_entry_date", "last_entry_date"],
    "billing": ["total_billing", "total_billings", "billings", "billing", "total_amount", "total_revenue"],
}

OPTIONAL_COLUMN_OPTIONS = {
    "total_hours": ["total_hours", "hours", "billable_hours"],
    "avg_rate": ["avg_rate", "average_rate", "rate"],
    "n_time_entries": ["n_time_entries", "time_entries", "number_of_time_entries"],
    "n_unique_users": ["n_unique_users", "unique_users", "number_of_users"],
    "practice_area": ["practice_area", "area_of_law", "matter_practice_area"],
    "matter_category": ["matter_category", "category", "matter_type", "matter_category_name", "case_type"],
    "all_time_entry_text": ["all_time_entry_text", "activity_text", "activity_text_description", "time_entry_text"],
}


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def clean_column_name(column_name: str) -> str:
    column_name = column_name.strip().lower()
    column_name = re.sub(r"[^a-z0-9]+", "_", column_name)
    return column_name.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [clean_column_name(col) for col in cleaned.columns]
    return cleaned


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def infer_columns(df: pd.DataFrame) -> dict[str, str | None]:
    inferred = {}

    for logical_name, candidates in REQUIRED_COLUMN_OPTIONS.items():
        inferred[logical_name] = find_column(df, candidates)

    for logical_name, candidates in OPTIONAL_COLUMN_OPTIONS.items():
        inferred[logical_name] = find_column(df, candidates)

    return inferred


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "$0"
    return f"${value:,.0f}"


def format_number(value: float) -> str:
    if pd.isna(value):
        return "0"
    return f"{value:,.0f}"


def parse_dates(df: pd.DataFrame, date_column: str) -> pd.Series:
    return pd.to_datetime(df[date_column], errors="coerce")


def choose_default_granularity(date_series: pd.Series) -> str:
    valid_dates = date_series.dropna()
    if valid_dates.empty:
        return "Monthly"

    date_span_days = (valid_dates.max() - valid_dates.min()).days
    unique_months = valid_dates.dt.to_period("M").nunique()

    if date_span_days > 900 or unique_months > 36:
        return "Yearly"
    return "Monthly"


def build_time_series(df: pd.DataFrame, date_column: str, billing_column: str, matter_column: str, granularity: str) -> pd.DataFrame:
    working = df.copy()
    working["parsed_date"] = parse_dates(working, date_column)
    working[billing_column] = pd.to_numeric(working[billing_column], errors="coerce").fillna(0)
    working = working.dropna(subset=["parsed_date"])

    if granularity == "Yearly":
        working["period"] = working["parsed_date"].dt.to_period("Y").dt.to_timestamp()
    else:
        working["period"] = working["parsed_date"].dt.to_period("M").dt.to_timestamp()

    time_series = (
        working.groupby("period")
        .agg(
            number_of_matters=(matter_column, "nunique"),
            total_billing=(billing_column, "sum"),
        )
        .reset_index()
        .sort_values("period")
    )

    return time_series


def calculate_tick_angle(time_series: pd.DataFrame) -> int:
    if len(time_series) > 18:
        return -45
    return 0


def calculate_tick_frequency(time_series: pd.DataFrame, granularity: str) -> int:
    periods = len(time_series)

    if periods <= 12:
        return 1
    if periods <= 24:
        return 2 if granularity == "Monthly" else 1
    if periods <= 48:
        return 4 if granularity == "Monthly" else 2
    return 6 if granularity == "Monthly" else 3


def build_trend_chart(time_series: pd.DataFrame, granularity: str) -> go.Figure:
    tick_frequency = calculate_tick_frequency(time_series, granularity)
    tick_values = time_series["period"].iloc[::tick_frequency]
    tick_format = "%b %Y" if granularity == "Monthly" else "%Y"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_series["period"],
            y=time_series["number_of_matters"],
            mode="lines+markers",
            name="Number of matters",
            line=dict(width=3),
            marker=dict(size=7),
            yaxis="y1",
            hovertemplate="%{x|" + tick_format + "}<br>Matters: %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=time_series["period"],
            y=time_series["total_billing"],
            name="Total billing",
            opacity=0.42,
            yaxis="y2",
            hovertemplate="%{x|" + tick_format + "}<br>Billing: $%{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Matter Volume and Billing Over Time — {granularity}",
            x=0.02,
            xanchor="left",
            font=dict(size=22),
        ),
        height=560,
        margin=dict(l=40, r=55, t=80, b=55),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            title="",
            tickvals=tick_values,
            tickformat=tick_format,
            tickangle=calculate_tick_angle(time_series),
            showgrid=False,
            linecolor="#e5e7eb",
        ),
        yaxis=dict(
            title="Number of matters",
            showgrid=True,
            gridcolor="#eef2f7",
            zeroline=False,
        ),
        yaxis2=dict(
            title="Total billing",
            overlaying="y",
            side="right",
            tickprefix="$",
            showgrid=False,
            zeroline=False,
        ),
        bargap=0.35,
    )

    return fig


def render_metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_missing_columns(missing: list[str]) -> None:
    st.error(
        "The uploaded file is missing required columns: " + ", ".join(missing) + ". "
        "Rename the columns or adjust the accepted column names inside REQUIRED_COLUMN_OPTIONS."
    )


BASE_LEGAL_SERVICE_KEYWORDS = {
    "agreement", "agreements", "contract", "contracts", "lease", "leasing", "nda", "mnda", "confidentiality",
    "incorporation", "incorporate", "corporate", "corporation", "company", "business", "shareholder", "shareholders",
    "estate", "probate", "will", "wills", "trust", "trusts", "poa", "power", "attorney", "representation",
    "litigation", "dispute", "claim", "claims", "settlement", "court", "motion", "pleading", "pleadings",
    "employment", "employee", "employer", "termination", "severance", "immigration", "visa", "permit",
    "trademark", "copyright", "ip", "intellectual", "property", "real", "transaction", "purchase", "sale",
    "family", "divorce", "dissolution", "separation", "custody", "support", "tax", "planning", "financing", "loan",
    "privacy", "policy", "terms", "service", "licensing", "license", "review", "advisory", "compliance",
}

LEGAL_PROJECT_ACRONYMS = {
    # Very high-confidence legal service acronyms
    "nda": "NDA",
    "mnda": "MNDA",
    "baa": "BAA",          # Business Associate Agreement
    "poa": "POA",
    "msa": "MSA",
    "sow": "SOW",
    "loi": "LOI",
    "mou": "MOU",
    "dpa": "DPA",

    # Common contract / transaction acronyms, but still potentially ambiguous
    "spa": "SPA",
    "apa": "APA",
    "sha": "SHA",

    # IP / dispute
    "ip": "IP",
    "tm": "TM",
    "adr": "ADR",

    # Existing one
    "loa": "LOA",

    # M&A
    "ma": "M&A",
    "m&a": "M&A",
}

PROJECT_STOPWORDS = set(ENGLISH_STOP_WORDS).union({
    "matter", "client", "file", "general", "legal", "services", "service", "work", "review",
    "draft", "drafting", "prepare", "preparation", "advice", "consultation", "conference",
    "email", "call", "meeting", "internal", "admin", "administrative", "misc", "miscellaneous",
    "new", "old", "ltd", "inc", "corp", "company", "corporation", "limited", "llc", "pllc", "group",
    "na", "none", "unknown", "untitled", "mattername", "project", "projects", "regarding", "phone",
})

ALL_STOP_WORDS = PROJECT_STOPWORDS
MIN_FIRST_PASS_CLUSTER_SIZE = 5
PREFER_MORE_FIRST_PASS_CLUSTERS = True
TEMPLATE_MAX_TOKEN_CATEGORY_SHARE = 0.25
TEMPLATE_MIN_WEIGHTED_SCORE = 1.75
TEMPLATE_MIN_MATCHED_SIGNALS = 1
TEMPLATE_SECOND_BEST_MARGIN = 0.35


CANONICAL_LABEL_RULES = []

SUBPROJECT_LABEL_RULES = {}


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(value: object) -> list[str]:
    return clean_text(value).split()


def normalize_legal_token(token: str) -> str:
    token_map = {
        "agreements": "agreement",
        "contracts": "contract",
        "shareholders": "shareholder",
        "wills": "will",
        "trusts": "trust",
        "leases": "lease",
        "corporations": "corporation",
        "companies": "company",
        "claims": "claim",
        "pleadings": "pleading",
        "motions": "motion",
        "trademarks": "trademark",
        "licenses": "license",
        "licences": "license",
        "licensing": "license",
    }
    return token_map.get(token, token)


def apply_legal_phrase_normalization(text: str) -> str:
    replacements = {
        "non disclosure": "nda",
        "non disclosure agreement": "nda",
        "mutual non disclosure agreement": "mnda",
        "power of attorney": "power attorney",
        "shareholders agreement": "shareholder agreement",
        "shareholder agreement": "shareholder agreement",
        "independent contractor agreement": "contractor agreement",
        "real estate": "real estate",
        "estate planning": "estate planning",
        "commercial lease": "commercial lease",
    }
    clean = clean_text(text)
    for source, target in replacements.items():
        clean = clean.replace(source, target)
    return clean


def remove_junk_tokens(tokens: list[str]) -> list[str]:
    return [
        normalize_legal_token(token)
        for token in tokens
        if token not in PROJECT_STOPWORDS and not token.isdigit() and len(token) > 2
    ]


def normalize_legal_text_for_clustering(value: object) -> str:
    text = apply_legal_phrase_normalization(value)
    tokens = remove_junk_tokens(tokenize_text(text))
    return " ".join(tokens)


def make_ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def clean_project_text(value: object) -> str:
    return normalize_legal_text_for_clustering(value)


# === Project signature normalization and labeling functions ===
def normalize_matter_name_signature(value: object) -> str:
    """
    Build a stable matter-name signature before clustering.
    This is intentionally more literal than template matching, so repeated firm formats
    like "divorce with children" stay together instead of being renamed by templates.
    """
    text = "" if pd.isna(value) else str(value).lower()
    text = re.sub(r"[-_/|]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    junk_phrases = [
        "closed", "trial billing matter", "trial billing", "billing matter",
        "test matter", "sample matter", "new matter", "old matter",
    ]
    for phrase in junk_phrases:
        text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text)

    phrase_replacements = {
        "dissolution of marriage": "dissolution",
        "marriage dissolution": "dissolution",
        "divorce children": "divorce with children",
        "divorce child": "divorce with children",
        "divorce with child": "divorce with children",
        "divorce w children": "divorce with children",
        "divorce w child": "divorce with children",
        "divorce no children": "divorce no children",
        "divorce without children": "divorce no children",
        "divorce no child": "divorce no children",
        "divorce without child": "divorce no children",
        "divorce no kids": "divorce no children",
        "divorce without kids": "divorce no children",
        "dissolution no kids": "dissolution no children",
        "dissolution without kids": "dissolution no children",
        "dissolution with kids": "dissolution with children",
        "dissolution with child": "dissolution with children",
        "contested divorce children": "contested divorce with children",
        "contested divorce with child": "contested divorce with children",
        "uncontested divorce children": "uncontested divorce with children",
        "post judgment": "postjudgment",
        "post decree": "postjudgment",
        "non disclosure agreement": "nda",
        "power of attorney": "power attorney",
        "estate planning": "estate planning",
    }

    for source, target in phrase_replacements.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)

    tokens = tokenize_text(text)

    # Keep terms that are actually project-defining. Do not let generic legal/admin words drive the signature.
    signature_stopwords = PROJECT_STOPWORDS.union({
        "with", "without", "and", "the", "for", "from", "into", "onto", "miscellaneous",
        "misc", "project", "projects", "matter", "matters",
    })
    tokens = [normalize_legal_token(token) for token in tokens if token not in signature_stopwords and not token.isdigit() and len(token) > 2]

    # Preserve important family-law modifiers.
    original_text = text
    if "dissolution" in tokens:
        if "no children" in original_text or "without children" in original_text:
            return "dissolution no children"
        if "child" in tokens or "children" in tokens:
            if "contested" in tokens:
                return "contested dissolution children"
            if "uncontested" in tokens:
                return "uncontested dissolution children"
            return "dissolution children"
        return "dissolution"

    if "divorce" in tokens:
        if "no children" in original_text or "without children" in original_text:
            return "divorce no children"
        if "child" in tokens or "children" in tokens:
            if "contested" in tokens:
                return "contested divorce children"
            if "uncontested" in tokens:
                return "uncontested divorce children"
            return "divorce children"
        return "divorce"

    if "custody" in tokens:
        return "custody"

    if "support" in tokens and ("child" in tokens or "children" in tokens):
        return "child support"

    if "postjudgment" in tokens:
        return "postjudgment"

    if not tokens:
        return ""

    # Bigrams/trigrams are more useful than isolated words for legal project types.
    trigrams = make_ngrams(tokens, 3)
    bigrams = make_ngrams(tokens, 2)
    if trigrams:
        return trigrams[0]
    if bigrams:
        return bigrams[0]
    return tokens[0]


def title_from_signature(signature: str) -> str:
    signature = re.sub(r"\s+", " ", str(signature).strip())
    if not signature:
        return "Unclear / Needs Review"

    label_map = {
        "divorce children": "Divorce With Children",
        "contested divorce children": "Contested Divorce With Children",
        "uncontested divorce children": "Uncontested Divorce With Children",
        "divorce no children": "Divorce Without Children",
        "divorce": "Divorce",
        "dissolution children": "Dissolution With Children",
        "contested dissolution children": "Contested Dissolution With Children",
        "uncontested dissolution children": "Uncontested Dissolution With Children",
        "dissolution no children": "Dissolution Without Children",
        "dissolution": "Dissolution",
        "custody": "Custody",
        "child support": "Child Support",
        "postjudgment": "Post Judgment",
    }
    return label_map.get(signature, signature.title())


def dominant_signature_label(values: pd.Series, min_share: float = 0.45) -> tuple[str | None, float]:
    signatures = values.dropna().map(normalize_matter_name_signature)
    signatures = signatures[signatures.str.len() > 0]
    if signatures.empty:
        return None, 0.0

    shares = signatures.value_counts(normalize=True)
    top_signature = shares.index[0]
    top_share = float(shares.iloc[0])

    if top_share >= min_share:
        return title_from_signature(top_signature), top_share

    return None, top_share


def normalize_phrase_variant_token(token: str) -> str:
    token = normalize_legal_token(str(token).lower().strip())
    variant_map = {
        "parenting": "parent",
        "parental": "parent",
        "parents": "parent",
        "children": "child",
        "kids": "child",
        "plans": "plan",
        "agreements": "agreement",
        "contracts": "contract",
    }
    return variant_map.get(token, token)


def title_from_phrase_key(phrase_key: str) -> str:
    label_map = {
        "parent plan": "Parenting Plan",
        "child support": "Child Support",
        "estate planning": "Estate Planning",
        "real estate": "Real Estate",
        "flat fee": "Flat Fee",
    }
    return label_map.get(phrase_key, phrase_key.title())


CLUSTER_ROOT_PHRASES = [
    "real estate",
    "estate planning",
    "child support",
    "postjudgment",
    "asset purchase",
    "shareholder agreement",
    "contractor agreement",
]
CLUSTER_PURITY_THRESHOLD = 0.95
FLAT_FEE_PROJECT_NAME = "Flat Fee"
OTHERS_PROJECT_NAME = "Others"
TOP_PROJECT_PURITY_COUNT = 10
TOP_PROJECT_PURITY_THRESHOLD = 0.95


def get_project_root_signal(signature: object) -> str:
    text = re.sub(r"\s+", " ", str(signature or "").strip().lower())
    if not text:
        return ""
    for phrase in CLUSTER_ROOT_PHRASES:
        if phrase in text:
            return phrase
    tokens = remove_junk_tokens(tokenize_text(text))
    return tokens[0] if tokens else ""


def extract_consecutive_phrase_keys(value: object, min_n: int = 2, max_n: int = 3) -> set[str]:
    tokens = [normalize_phrase_variant_token(token) for token in tokenize_text(value) if token and not token.isdigit()]
    phrase_keys = set()
    for n in range(min_n, max_n + 1):
        if len(tokens) < n:
            continue
        for index in range(len(tokens) - n + 1):
            phrase_tokens = tokens[index:index + n]
            if any(token in PROJECT_STOPWORDS or len(token) <= 2 for token in phrase_tokens):
                continue
            phrase_keys.add(" ".join(phrase_tokens))
    return phrase_keys


def dominant_consecutive_phrase_label(values: pd.Series, min_share: float = 0.95) -> tuple[str | None, float]:
    examples = values.dropna().astype(str).map(str.strip).loc[lambda s: s != ""]
    if examples.empty:
        return None, 0.0
    phrase_counts = Counter()
    for value in examples:
        phrase_counts.update(extract_consecutive_phrase_keys(value))
    if not phrase_counts:
        return None, 0.0
    top_phrase, top_count = phrase_counts.most_common(1)[0]
    top_share = top_count / len(examples)
    if top_share >= min_share:
        return title_from_phrase_key(top_phrase), float(top_share)
    return None, float(top_share)


def root_first_cluster_label(values: pd.Series, min_share: float = 0.35, phrase_share: float = 0.95) -> tuple[str | None, float]:
    signatures = values.dropna().map(normalize_matter_name_signature)
    signatures = signatures[signatures.str.len() > 0]
    if signatures.empty:
        return None, 0.0
    roots = signatures.map(get_project_root_signal)
    roots = roots[roots.str.len() > 0]
    if roots.empty:
        return dominant_signature_label(values, min_share=min_share)
    root_shares = roots.value_counts(normalize=True)
    top_root = root_shares.index[0]
    top_root_share = float(root_shares.iloc[0])
    if top_root_share >= min_share:
        phrase_label, phrase_share_value = dominant_consecutive_phrase_label(values, min_share=phrase_share)
        if phrase_label:
            return phrase_label, phrase_share_value
        return title_from_signature(top_root), top_root_share
    return dominant_signature_label(values, min_share=min_share)


def matter_name_contains_flat_fee(value: object) -> bool:
    return bool(re.search(r"\bflat\s*fee\b", clean_text(value)))


def is_flat_fee_project(value: object) -> bool:
    return str(value or "").strip().lower() == FLAT_FEE_PROJECT_NAME.lower()


def is_others_project(value: object) -> bool:
    return str(value or "").strip().lower() == OTHERS_PROJECT_NAME.lower()


def enforce_cluster_root_purity(working: pd.DataFrame, cluster_col: str, signature_col: str = "project_signature") -> pd.DataFrame:
    if working.empty or cluster_col not in working.columns or signature_col not in working.columns:
        return working
    working = working.copy()
    working["cluster_root_signal"] = working[signature_col].map(get_project_root_signal)
    next_cluster_id = int(pd.to_numeric(working[cluster_col], errors="coerce").max()) + 1
    split_notes = pd.Series("", index=working.index, dtype="object")
    for cluster_id in sorted(working[cluster_col].dropna().unique()):
        cluster_index = working.index[working[cluster_col] == cluster_id]
        root_counts = working.loc[cluster_index, "cluster_root_signal"].loc[lambda s: s != ""].value_counts()
        if root_counts.empty:
            continue
        dominant_root = root_counts.index[0]
        dominant_share = float((working.loc[cluster_index, "cluster_root_signal"] == dominant_root).mean())
        if dominant_share >= CLUSTER_PURITY_THRESHOLD:
            continue
        for root_signal in root_counts[root_counts.index != dominant_root].index.tolist():
            split_index = cluster_index[working.loc[cluster_index, "cluster_root_signal"] == root_signal]
            if split_index.empty:
                continue
            working.loc[split_index, cluster_col] = next_cluster_id
            split_notes.loc[split_index] = (
                f"Split from cluster {cluster_id}: dominant root '{dominant_root}' share "
                f"{dominant_share:.0%}; separated root '{root_signal}'."
            )
            next_cluster_id += 1
    working["cluster_purity_note"] = split_notes
    return working


def force_flat_fee_cluster(working: pd.DataFrame, cluster_col: str, matter_name_col: str) -> pd.DataFrame:
    if working.empty or cluster_col not in working.columns or matter_name_col not in working.columns:
        return working
    flat_fee_mask = working[matter_name_col].apply(matter_name_contains_flat_fee)
    if not flat_fee_mask.any():
        return working
    working = working.copy()
    next_cluster_id = int(pd.to_numeric(working[cluster_col], errors="coerce").max()) + 1
    working.loc[flat_fee_mask, cluster_col] = next_cluster_id
    working.loc[flat_fee_mask, "project_signature"] = "flat fee"
    working.loc[flat_fee_mask, "primary_project_text"] = "flat fee"
    working.loc[flat_fee_mask, "cluster_root_signal"] = "flat fee"
    working.loc[flat_fee_mask, "cluster_purity_note"] = "Forced flat fee group."
    return working


def get_project_label_terms(project_name: object) -> set[str]:
    acronym_terms = set(extract_legal_project_acronyms(project_name)) if "extract_legal_project_acronyms" in globals() else set()
    normalized_terms = set(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(project_name))))
    if normalized_terms:
        return normalized_terms.union(acronym_terms)
    return set(tokenize_text(clean_text(project_name))).union(acronym_terms)


def project_label_matches_text(value: object, project_name: object) -> bool:
    if not project_name or is_flat_fee_project(project_name) or is_others_project(project_name):
        return False
    project_terms = get_project_label_terms(project_name)
    if not project_terms:
        return False
    value_text = normalize_legal_text_for_clustering(value)
    project_text = normalize_legal_text_for_clustering(project_name)
    if project_text and f" {project_text} " in f" {value_text} ":
        return True
    value_terms = set(tokenize_text(value_text))
    if len(project_terms) == 1:
        return next(iter(project_terms)) in value_terms
    return project_terms.issubset(value_terms)


def top_time_entry_terms_for_row(row: pd.Series, text_col: str | None, top_n: int = 10) -> set[str]:
    if not text_col or text_col not in row.index:
        return set()
    tokens = remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(row.get(text_col, ""))))
    return {term for term, _ in Counter(tokens).most_common(top_n)}


def choose_project_reassignment(
    row: pd.Series,
    current_project: str,
    candidate_projects: list[str],
    project_terms: dict[str, set[str]],
    project_counts: dict[str, int],
    matter_name_col: str,
    text_col: str | None,
) -> tuple[str, str]:
    matter_name = row.get(matter_name_col, "")
    scored_matches = []
    for candidate_project in candidate_projects:
        if candidate_project == current_project or is_flat_fee_project(candidate_project) or is_others_project(candidate_project):
            continue
        terms = project_terms.get(candidate_project, set())
        if not terms:
            continue
        if project_label_matches_text(matter_name, candidate_project):
            scored_matches.append((100 + len(terms), project_counts.get(candidate_project, 0), candidate_project))
            continue
        matter_terms = set(tokenize_text(normalize_legal_text_for_clustering(matter_name)))
        overlap = len(terms.intersection(matter_terms))
        if overlap:
            scored_matches.append((overlap, project_counts.get(candidate_project, 0), candidate_project))

    if scored_matches:
        return sorted(scored_matches, reverse=True)[0][2], "matched_other_project_name"

    time_entry_terms = top_time_entry_terms_for_row(row, text_col, top_n=10)
    if time_entry_terms:
        keyword_matches = [
            (len(time_entry_terms.intersection(project_terms.get(candidate_project, set()))), project_counts.get(candidate_project, 0), candidate_project)
            for candidate_project in candidate_projects
            if candidate_project != current_project and not is_flat_fee_project(candidate_project) and not is_others_project(candidate_project)
        ]
        keyword_matches = [match for match in keyword_matches if match[0] > 0]
        if keyword_matches:
            return sorted(keyword_matches, reverse=True)[0][2], "matched_time_entry_keyword"

    return OTHERS_PROJECT_NAME, "moved_to_others"


def enforce_top_project_name_purity(
    working: pd.DataFrame,
    label_col: str,
    matter_name_col: str,
    count_col: str,
    text_col: str | None,
    top_n: int = TOP_PROJECT_PURITY_COUNT,
    min_purity: float = TOP_PROJECT_PURITY_THRESHOLD,
) -> pd.DataFrame:
    if working.empty or label_col not in working.columns or matter_name_col not in working.columns:
        return working
    working = working.copy()
    working["project_purity_reassignment"] = ""
    for _ in range(10):
        changed = False
        project_counts = (
            working.groupby(label_col)[count_col]
            .nunique()
            .sort_values(ascending=False)
            .to_dict()
        )
        top_projects = [
            project for project in project_counts
            if project and not is_flat_fee_project(project) and not is_others_project(project)
        ][:top_n]
        if not top_projects:
            return working
        candidate_projects = [
            project for project in project_counts
            if project and not is_others_project(project)
        ]
        project_terms = {project: get_project_label_terms(project) for project in candidate_projects}
        for project_name in top_projects:
            project_mask = working[label_col] == project_name
            if not project_mask.any():
                continue
            project_df = working.loc[project_mask]
            pure_mask = project_df[matter_name_col].apply(lambda value: project_label_matches_text(value, project_name))
            total_matters = max(project_df[count_col].nunique(), 1)
            pure_matters = project_df.loc[pure_mask, count_col].nunique()
            if pure_matters / total_matters >= min_purity:
                continue
            impure_index = project_df.index[~pure_mask]
            for row_index in impure_index:
                target_project, method = choose_project_reassignment(
                    working.loc[row_index],
                    current_project=project_name,
                    candidate_projects=candidate_projects,
                    project_terms=project_terms,
                    project_counts=project_counts,
                    matter_name_col=matter_name_col,
                    text_col=text_col,
                )
                if target_project == project_name:
                    continue
                working.at[row_index, label_col] = target_project
                working.at[row_index, "project_purity_reassignment"] = (
                    f"Moved from {project_name} by top-project purity rule: {method}."
                )
                changed = True
        if not changed:
            break
    return working


def rebuild_cluster_summary_from_project_names(
    working: pd.DataFrame,
    cluster_col: str,
    label_col: str,
    matter_name_col: str,
    billing_col: str,
    count_col: str,
    practice_area_col: str | None,
) -> pd.DataFrame:
    if working.empty or label_col not in working.columns:
        return pd.DataFrame()
    rows = []
    project_order = (
        working.groupby(label_col)[count_col]
        .nunique()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    project_to_cluster_id = {project_name: index for index, project_name in enumerate(project_order)}
    working[cluster_col] = working[label_col].map(project_to_cluster_id).astype(int)
    working["display_label"] = working.apply(
        lambda row: f"{row[label_col]} · {cluster_col.replace('_id', '').title()} {row[cluster_col]}",
        axis=1,
    )

    for project_name in project_order:
        project_df = working[working[label_col] == project_name].copy()
        if project_df.empty:
            continue
        source_text_col = "primary_project_text" if "primary_project_text" in project_df.columns else matter_name_col
        top_terms = get_top_terms_from_text(project_df[source_text_col], ngram_n=1, top_n=10)
        top_bigrams = get_top_terms_from_text(project_df[source_text_col], ngram_n=2, top_n=10)
        numeric_billing = pd.to_numeric(project_df[billing_col], errors="coerce").fillna(0)
        practice_area_context = summarize_practice_area_context(project_df, practice_area_col)
        root_signal = (
            project_df["cluster_root_signal"].dropna().astype(str).mode().iloc[0]
            if "cluster_root_signal" in project_df.columns and not project_df["cluster_root_signal"].dropna().empty
            else ""
        )
        root_purity = (
            float((project_df["cluster_root_signal"] == root_signal).mean())
            if root_signal and "cluster_root_signal" in project_df.columns
            else np.nan
        )
        purity_notes = []
        if "cluster_purity_note" in project_df.columns:
            purity_notes.extend(project_df["cluster_purity_note"].dropna().astype(str).loc[lambda s: s != ""].drop_duplicates().tolist())
        if "project_purity_reassignment" in project_df.columns:
            purity_notes.extend(project_df["project_purity_reassignment"].dropna().astype(str).loc[lambda s: s != ""].drop_duplicates().tolist())

        rows.append({
            cluster_col: project_to_cluster_id[project_name],
            label_col: project_name,
            "display_label": f"{project_name} · {cluster_col.replace('_id', '').title()} {project_to_cluster_id[project_name]}",
            "matter_count": project_df[count_col].nunique(),
            "total_billing": numeric_billing.sum(),
            "avg_billing": numeric_billing.mean(),
            "top_terms": ", ".join(top_terms),
            "top_bigrams": ", ".join(top_bigrams),
            "example_matter_names": format_examples_for_label(
                project_df[matter_name_col],
                label=project_name,
                top_terms=top_terms + top_bigrams,
                n=10,
            ),
            "dominant_signature_share": root_first_cluster_label(project_df[matter_name_col], min_share=0.35)[1],
            "practice_area_null_percentage": practice_area_context["practice_area_null_percentage"],
            "most_frequent_practice_area": practice_area_context["most_frequent_practice_area"],
            "cluster_root_signal": root_signal,
            "cluster_root_purity": root_purity,
            "cluster_purity_note": " | ".join(dict.fromkeys(purity_notes)),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["flat_fee_sort"] = summary[label_col].map(is_flat_fee_project)
    summary["others_sort"] = summary[label_col].map(is_others_project)
    return (
        summary
        .sort_values(["flat_fee_sort", "others_sort", "matter_count", "total_billing"], ascending=[True, True, False, False])
        .drop(columns=["flat_fee_sort", "others_sort"])
    )


# --- Cluster label validation ---
def validate_cluster_label_against_examples(project_name: str, examples: pd.Series, min_overlap: int = 1) -> bool:
    label_tokens = set(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(project_name))))
    example_text = " ".join(examples.dropna().astype(str).head(25).tolist())
    example_tokens = set(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(example_text))))

    if not label_tokens or not example_tokens:
        return True

    return len(label_tokens.intersection(example_tokens)) >= min_overlap


def get_template_paths() -> tuple[Path | None, Path | None]:
    """
    Resolve template paths for this repo structure:

    AltFee/
    ├── requirements.txt
    ├── classification_pipeline/
    │   └── app.py
    └── data/
        └── templates/
            ├── template_guidelines.csv
            └── template_categories.csv

    Works whether Streamlit is launched from the repo root or from classification_pipeline/.
    """
    app_file = Path(__file__).resolve()
    app_dir = app_file.parent
    repo_root = app_dir.parent

    candidate_roots = [
        repo_root,
        Path.cwd(),
        Path.cwd().parent,
        app_dir,
    ]

    for root in candidate_roots:
        possible_guidelines = root / "data" / "templates" / "template_guidelines.csv"
        possible_categories = root / "data" / "templates" / "template_categories.csv"

        if possible_guidelines.exists() and possible_categories.exists():
            return possible_guidelines, possible_categories

    return None, None


# === Sample account helpers ===
def get_repo_root() -> Path:
    """Find the repo root by walking upward until requirements.txt or data/ is found."""
    app_file = Path(__file__).resolve()
    for candidate in [app_file.parent, *app_file.parents]:
        if (candidate / "requirements.txt").exists() or (candidate / "data").exists():
            return candidate
    return app_file.parent.parent


def get_sample_account_paths() -> list[Path]:
    """Return bundled sample account CSVs from data/random_account_exports."""
    candidate_dirs = [
        get_repo_root() / "data" / "random_account_exports",
        Path.cwd() / "data" / "random_account_exports",
        Path.cwd().parent / "data" / "random_account_exports",
    ]

    for directory in candidate_dirs:
        if directory.exists():
            return sorted(directory.glob("*.csv"))

    return []


def format_sample_account_label(path: Path) -> str:
    label = path.stem
    label = re.sub(r"^account_", "Account ", label)
    label = re.sub(r"_historical_matter_evidence$", "", label)
    label = label.replace("_", " ")
    return label.title()



@st.cache_data(show_spinner=False)
def load_sample_csv(sample_path: str) -> pd.DataFrame:
    return pd.read_csv(sample_path)


# === Folio Practice Area Classification ===

def get_folio_practice_area_path() -> Path | None:
    candidate_paths = [
        get_repo_root() / "data" / "practice_areas" / "folio_practice_areas.csv",
        Path.cwd() / "data" / "practice_areas" / "folio_practice_areas.csv",
        Path.cwd().parent / "data" / "practice_areas" / "folio_practice_areas.csv",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    return None


@st.cache_data(show_spinner=False)
def load_folio_practice_areas() -> list[str]:
    path = get_folio_practice_area_path()
    if path is None:
        return []

    try:
        practice_area_df = pd.read_csv(path)
    except Exception:
        return []

    practice_area_df.columns = [clean_column_name(col) for col in practice_area_df.columns]

    name_candidates = ["name", "practice_area", "practice_area_name", "area", "label"]
    name_col = find_column(practice_area_df, name_candidates)
    if name_col is None:
        name_col = practice_area_df.columns[0]

    labels = (
        practice_area_df[name_col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )

    return sorted(labels)


@st.cache_resource(show_spinner=False)
def get_zero_shot_classifier():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def build_practice_area_classification_text(row: pd.Series) -> str:
    return (
        f"Project name: {row.get('project_name', '')}\n"
        f"Top terms: {row.get('top_terms', '')}\n"
        f"Top phrases: {row.get('top_bigrams', '')}\n"
        f"Example matters: {row.get('example_matter_names', '')}"
    )


def classify_clusters_to_practice_areas(cluster_summary: pd.DataFrame, candidate_labels: list[str]) -> pd.DataFrame:
    if cluster_summary.empty or not candidate_labels:
        cluster_summary = cluster_summary.copy()
        cluster_summary["predicted_practice_area"] = "Unclassified"
        cluster_summary["practice_area_score"] = np.nan
        return cluster_summary

    cluster_summary = cluster_summary.copy()

    try:
        classifier = get_zero_shot_classifier()
    except Exception as exc:
        st.warning(
            "Practice-area classification could not run because the Hugging Face zero-shot model is unavailable. "
            f"Install transformers/torch or disable this option. Error: {exc}"
        )
        cluster_summary["predicted_practice_area"] = "Unclassified"
        cluster_summary["practice_area_score"] = np.nan
        return cluster_summary

    predicted_areas = []
    scores = []

    for _, row in cluster_summary.iterrows():
        text = build_practice_area_classification_text(row)
        try:
            result = classifier(text, candidate_labels, multi_label=False)
            predicted_areas.append(result["labels"][0])
            scores.append(float(result["scores"][0]))
        except Exception:
            predicted_areas.append("Unclassified")
            scores.append(np.nan)

    cluster_summary["predicted_practice_area"] = predicted_areas
    cluster_summary["practice_area_score"] = scores
    return cluster_summary


FOLIO_API_BASE_URL = "https://folio.openlegalstandard.org"
FOLIO_SEARCH_TIMEOUT_SECONDS = 8
FOLIO_MATCH_THRESHOLD = 0.35
FOLIO_CACHE_VERSION = 2


def first_text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not pd.isna(value):
        return str(value)
    if isinstance(value, list):
        for item in value:
            text = first_text_value(item)
            if text:
                return text
    if isinstance(value, dict):
        for key in ["@value", "value", "label", "name", "title", "description"]:
            text = first_text_value(value.get(key))
            if text:
                return text
    return ""


def get_nested_text(payload: dict, keys: list[str]) -> str:
    for key in keys:
        if key in payload:
            text = first_text_value(payload.get(key))
            if text:
                return text
    return ""


def normalize_folio_class_id(value: object) -> str:
    class_id = first_text_value(value)
    if not class_id:
        return ""
    class_id = class_id.rstrip("/")
    if class_id.startswith("http://") or class_id.startswith("https://"):
        class_id = class_id.split("/")[-1]
    return class_id


def extract_folio_search_results(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ["classes", "results", "matches", "data", "items", "concepts"]:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested_results = extract_folio_search_results(value)
            if nested_results:
                return nested_results
    return [payload] if payload else []


def extract_folio_title(payload: dict) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("@graph"), list):
        for node in payload["@graph"]:
            if isinstance(node, dict):
                text = extract_folio_title(node)
                if text:
                    return text
    return get_nested_text(payload, [
        "title",
        "preferred_label",
        "prefLabel",
        "preferredLabel",
        "label",
        "name",
        "rdfs:label",
        "skos:prefLabel",
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2004/02/skos/core#prefLabel",
    ])


def extract_folio_description(payload: dict) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("@graph"), list):
        for node in payload["@graph"]:
            if isinstance(node, dict):
                text = extract_folio_description(node)
                if text:
                    return text
    return get_nested_text(payload, [
        "description",
        "definition",
        "comment",
        "rdfs:comment",
        "skos:definition",
        "http://www.w3.org/2000/01/rdf-schema#comment",
        "http://www.w3.org/2004/02/skos/core#definition",
    ])


def extract_folio_class_id(payload: dict) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("@graph"), list):
        for node in payload["@graph"]:
            if isinstance(node, dict):
                class_id = extract_folio_class_id(node)
                if class_id:
                    return class_id
    for key in ["class_id", "classId", "id", "@id", "iri", "uri", "concept_id", "conceptId"]:
        class_id = normalize_folio_class_id(payload.get(key))
        if class_id:
            return class_id
    return ""


def folio_match_score(cluster_name: str, folio_title: str) -> float:
    cluster_text = normalize_legal_text_for_clustering(cluster_name)
    title_text = normalize_legal_text_for_clustering(folio_title)
    if not cluster_text or not title_text:
        return 0.0
    cluster_tokens = cluster_text.split()
    title_tokens = title_text.split()
    cluster_token_set = set(cluster_tokens)
    title_token_set = set(title_tokens)
    cluster_bigrams = set(make_ngrams(cluster_tokens, 2))
    title_bigrams = set(make_ngrams(title_tokens, 2))
    if cluster_bigrams.intersection(title_bigrams):
        return 1.0
    overlap = len(cluster_token_set.intersection(title_token_set))
    union = len(cluster_token_set.union(title_token_set))
    token_score = overlap / union if union else 0.0
    containment_score = overlap / len(cluster_token_set) if cluster_token_set else 0.0
    sequence_score = SequenceMatcher(None, cluster_text, title_text).ratio()
    return max(token_score, containment_score * 0.92, sequence_score * 0.82)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_folio_project_match(
    cluster_name: str,
    min_score: float = FOLIO_MATCH_THRESHOLD,
    cache_version: int = FOLIO_CACHE_VERSION,
) -> dict:
    clean_cluster_name = str(cluster_name or "").strip()
    empty_result = {
        "folio_verified": False,
        "folio_title": "",
        "folio_description": "",
        "folio_class_id": "",
        "folio_match_score": np.nan,
        "folio_match_method": "no_match",
        "folio_api_status": "not_called",
        "folio_search_url": "",
    }
    if not clean_cluster_name:
        return empty_result

    search_url = f"{FOLIO_API_BASE_URL}/search/prefix"
    try:
        search_response = requests.get(
            search_url,
            params={"query": clean_cluster_name},
            timeout=FOLIO_SEARCH_TIMEOUT_SECONDS,
        )
        search_response.raise_for_status()
        search_results = extract_folio_search_results(search_response.json())
    except Exception:
        result = empty_result.copy()
        result["folio_match_method"] = "folio_search_failed"
        result["folio_api_status"] = "connection_failed"
        result["folio_search_url"] = f"{search_url}?query={quote(clean_cluster_name)}"
        return result

    best_result = None
    best_score = 0.0
    best_title = ""
    for candidate in search_results:
        candidate_title = extract_folio_title(candidate)
        candidate_id = extract_folio_class_id(candidate)
        candidate_label = get_nested_text(candidate, [
            "label",
            "name",
            "rdfs:label",
            "http://www.w3.org/2000/01/rdf-schema#label",
        ])
        candidate_match_labels = [
            label for label in [candidate_label, candidate_title, candidate_id] if label
        ]
        score = max(
            [folio_match_score(clean_cluster_name, label) for label in candidate_match_labels],
            default=0.0,
        )
        if score > best_score:
            best_result = candidate
            best_score = score
            best_title = candidate_title or candidate_label or candidate_id

    if best_result is None or best_score < min_score:
        result = empty_result.copy()
        result["folio_match_score"] = round(float(best_score), 3)
        result["folio_match_method"] = "below_threshold"
        result["folio_api_status"] = "no_match"
        result["folio_search_url"] = f"{search_url}?query={quote(clean_cluster_name)}"
        return result

    class_id = extract_folio_class_id(best_result)
    jsonld_payload = {}
    retrieval_failed = False
    if class_id:
        try:
            jsonld_response = requests.get(
                f"{FOLIO_API_BASE_URL}/{quote(class_id, safe='')}/jsonld",
                timeout=FOLIO_SEARCH_TIMEOUT_SECONDS,
            )
            jsonld_response.raise_for_status()
            jsonld_payload = jsonld_response.json()
        except Exception:
            retrieval_failed = True
            jsonld_payload = {}

    folio_title = extract_folio_title(jsonld_payload) if isinstance(jsonld_payload, dict) else ""
    folio_description = extract_folio_description(jsonld_payload) if isinstance(jsonld_payload, dict) else ""
    if not folio_title:
        folio_title = best_title
    if not folio_description:
        folio_description = extract_folio_description(best_result)

    return {
        "folio_verified": bool(folio_title),
        "folio_title": folio_title,
        "folio_description": folio_description,
        "folio_class_id": class_id,
        "folio_match_score": round(float(best_score), 3),
        "folio_match_method": "folio_retrieval_failed" if retrieval_failed else "prefix_search_top_match",
        "folio_api_status": "connection_failed" if retrieval_failed and not folio_description else "matched",
        "folio_search_url": f"{search_url}?query={quote(clean_cluster_name)}",
    }


def enrich_clusters_with_folio(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    cluster_summary = cluster_summary.copy()
    folio_rows = []
    for project_name in cluster_summary["project_name"].fillna("").astype(str):
        folio_rows.append(fetch_folio_project_match(project_name))
    folio_df = pd.DataFrame(folio_rows)
    return pd.concat([cluster_summary.reset_index(drop=True), folio_df.reset_index(drop=True)], axis=1)


def normalize_firm_practice_area_label(value: object) -> str:
    label = "" if pd.isna(value) else str(value)
    label = re.sub(r"\s+", " ", label).strip().lower()
    return label.title() if label else ""


def practice_area_match_score(firm_label: str, candidate_label: str) -> float:
    firm_text = normalize_legal_text_for_clustering(firm_label)
    candidate_text = normalize_legal_text_for_clustering(candidate_label)
    if not firm_text or not candidate_text:
        return 0.0
    firm_token_list = firm_text.split()
    candidate_token_list = candidate_text.split()
    firm_tokens = set(firm_token_list)
    candidate_tokens = set(candidate_token_list)
    firm_bigrams = set(make_ngrams(firm_token_list, 2))
    candidate_bigrams = set(make_ngrams(candidate_token_list, 2))
    if firm_bigrams.intersection(candidate_bigrams):
        return 1.0
    overlap = len(firm_tokens.intersection(candidate_tokens))
    union = len(firm_tokens.union(candidate_tokens))
    token_score = overlap / union if union else 0.0
    containment_score = overlap / len(firm_tokens) if firm_tokens else 0.0
    reverse_containment_score = overlap / len(candidate_tokens) if candidate_tokens else 0.0
    sequence_score = SequenceMatcher(None, firm_text, candidate_text).ratio()
    return max(token_score, containment_score * 0.92, reverse_containment_score * 0.92, sequence_score * 0.82)


def map_firm_practice_area_to_folio(firm_label: str, folio_labels: list[str]) -> tuple[str | None, float]:
    if not firm_label or not folio_labels:
        return None, 0.0
    best_label = None
    best_score = 0.0
    for folio_label in folio_labels:
        score = practice_area_match_score(firm_label, folio_label)
        if score > best_score:
            best_label = folio_label
            best_score = score
    if best_label and best_score >= 0.80:
        return best_label, best_score
    return None, best_score


def count_label_words(label: object) -> int:
    return len(tokenize_text(label))


PROJECT_LIKE_PRACTICE_AREA_TERMS = {
    "divorce", "dissolution", "custody", "support", "parenting", "probate", "will", "trust",
    "estate", "lease", "contract", "agreement", "nda", "trademark", "copyright", "immigration",
    "visa", "litigation", "dispute", "settlement", "incorporation", "compliance",
}

BROAD_PRACTICE_AREA_PROJECT_NAMES = {
    "family", "personal family", "employment", "labor employment", "real estate", "business",
    "corporate", "commercial", "litigation", "civil litigation", "immigration", "tax",
    "intellectual property", "estate", "trust estates", "personal injury",
}


def label_has_project_like_terms(label: object) -> bool:
    tokens = set(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(label))))
    return bool(tokens.intersection(PROJECT_LIKE_PRACTICE_AREA_TERMS))


def map_project_like_label_to_folio(label: str, folio_labels: list[str]) -> str | None:
    if not folio_labels:
        return None
    tokens = set(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(label))))
    preferred_labels = []
    if tokens.intersection({"divorce", "dissolution", "custody", "support", "parenting"}):
        preferred_labels = ["Personal and Family Law", "Family Law", "Family"]
    elif tokens.intersection({"will", "trust", "probate", "estate"}):
        preferred_labels = ["Estate Planning", "Trusts and Estates", "Trusts And Estates"]
    elif tokens.intersection({"lease", "property"}):
        preferred_labels = ["Real Estate Law", "Real Estate"]
    elif tokens.intersection({"contract", "agreement", "nda", "incorporation", "business", "corporate"}):
        preferred_labels = ["Contract Law", "Business Law", "Corporate Law", "Commercial Law"]
    elif tokens.intersection({"trademark", "copyright", "intellectual"}):
        preferred_labels = ["Intellectual Property Law", "Intellectual Property"]
    elif tokens.intersection({"immigration", "visa"}):
        preferred_labels = ["Immigration Law", "Immigration"]
    elif tokens.intersection({"litigation", "dispute", "settlement"}):
        preferred_labels = ["Dispute Resolution Law", "Litigation", "Civil Litigation"]
    elif tokens.intersection({"tax"}):
        preferred_labels = ["Tax Law", "Tax"]

    for preferred_label in preferred_labels:
        preferred_tokens = set(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(preferred_label))))
        for folio_label in folio_labels:
            folio_tokens = set(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(folio_label))))
            if preferred_tokens and preferred_tokens.issubset(folio_tokens):
                return folio_label
    return None


def build_firm_specific_taxonomy_labels(
    matter_df: pd.DataFrame,
    practice_area_col: str | None,
    folio_labels: list[str],
) -> tuple[list[str], pd.DataFrame]:
    taxonomy_labels = list(dict.fromkeys(folio_labels))
    mapping_rows = []
    if not practice_area_col or practice_area_col not in matter_df.columns:
        return sorted(taxonomy_labels), pd.DataFrame(mapping_rows)
    raw_values = matter_df[practice_area_col].drop_duplicates().tolist()
    kept_labels = set(taxonomy_labels)
    for raw_value in raw_values:
        cleaned_label = normalize_firm_practice_area_label(raw_value)
        if not cleaned_label:
            mapping_rows.append({
                "raw_practice_area": raw_value,
                "cleaned_practice_area": "",
                "matched_folio_label": "",
                "folio_match_score": np.nan,
                "action": "dropped_empty",
                "final_taxonomy_label": "",
            })
            continue
        mapped_label, score = map_firm_practice_area_to_folio(cleaned_label, folio_labels)
        if mapped_label:
            classification_label = mapped_label
            action = "mapped_to_folio"
        elif label_has_project_like_terms(cleaned_label):
            mapped_project_label = map_project_like_label_to_folio(cleaned_label, folio_labels)
            if mapped_project_label:
                classification_label = mapped_project_label
                mapped_label = mapped_project_label
                score = 1.0
                action = "mapped_project_like_to_folio"
            else:
                mapping_rows.append({
                    "raw_practice_area": raw_value,
                    "cleaned_practice_area": cleaned_label,
                    "matched_folio_label": "",
                    "folio_match_score": round(float(score), 3),
                    "action": "dropped_project_like_practice_area",
                    "final_taxonomy_label": "",
                })
                continue
        elif count_label_words(cleaned_label) > 4:
            mapping_rows.append({
                "raw_practice_area": raw_value,
                "cleaned_practice_area": cleaned_label,
                "matched_folio_label": "",
                "folio_match_score": round(float(score), 3),
                "action": "dropped_too_many_words",
                "final_taxonomy_label": "",
            })
            continue
        else:
            classification_label = cleaned_label
            action = "kept_firm_specific"
        if classification_label in kept_labels:
            if not mapped_label:
                action = "dropped_duplicate"
            final_taxonomy_label = classification_label
        else:
            taxonomy_labels.append(classification_label)
            kept_labels.add(classification_label)
            final_taxonomy_label = classification_label
        mapping_rows.append({
            "raw_practice_area": raw_value,
            "cleaned_practice_area": cleaned_label,
            "matched_folio_label": mapped_label or "",
            "folio_match_score": round(float(score), 3),
            "action": action,
            "final_taxonomy_label": final_taxonomy_label,
        })
    return sorted(taxonomy_labels), pd.DataFrame(mapping_rows)


def build_taxonomy_candidate_source_map(candidate_labels: list[str], mapping_df: pd.DataFrame) -> dict[str, str]:
    source_map = {label: "folio" for label in candidate_labels}
    if mapping_df.empty:
        return source_map
    for _, row in mapping_df.iterrows():
        label = row.get("final_taxonomy_label")
        action = row.get("action")
        if not label:
            continue
        if action == "kept_firm_specific":
            source_map[label] = "firm"
        elif label not in source_map:
            source_map[label] = "folio"
    return source_map


def build_taxonomy_level_classification_text(row: pd.Series) -> str:
    return (
        f"Practice area null percentage: {row.get('practice_area_null_percentage', '')}\n"
        f"Most frequent practice area: {row.get('most_frequent_practice_area', '')}\n"
        f"Top terms: {row.get('top_terms', '')}\n"
        f"Top phrases: {row.get('top_bigrams', '')}\n"
        f"Example matters: {row.get('example_matter_names', '')}"
    )


def normalize_taxonomy_candidate_label(value: object) -> str:
    return normalize_legal_text_for_clustering(value)


def find_best_existing_candidate(preferred_labels: list[str], candidate_labels: list[str]) -> str | None:
    lookup = {normalize_taxonomy_candidate_label(label): label for label in candidate_labels if normalize_taxonomy_candidate_label(label)}
    for preferred_label in preferred_labels:
        normalized_preferred = normalize_taxonomy_candidate_label(preferred_label)
        if normalized_preferred in lookup:
            return lookup[normalized_preferred]
    for preferred_label in preferred_labels:
        preferred_tokens = set(remove_junk_tokens(tokenize_text(preferred_label)))
        if not preferred_tokens:
            continue
        best_label = None
        best_score = 0
        for candidate_label in candidate_labels:
            candidate_tokens = set(remove_junk_tokens(tokenize_text(candidate_label)))
            if not candidate_tokens:
                continue
            score = len(preferred_tokens.intersection(candidate_tokens)) / max(len(preferred_tokens), 1)
            if score > best_score:
                best_label = candidate_label
                best_score = score
        if best_label and best_score >= 0.75:
            return best_label
    return None


def infer_obvious_taxonomy_level(row: pd.Series, candidate_labels: list[str], candidate_source_map: dict[str, str] | None = None):
    top_terms = str(row.get("top_terms", ""))
    top_bigrams = str(row.get("top_bigrams", ""))
    examples = str(row.get("example_matter_names", ""))
    most_frequent_practice_area = str(row.get("most_frequent_practice_area", "") or "")
    if most_frequent_practice_area:
        existing_candidate = find_best_existing_candidate([most_frequent_practice_area, f"{most_frequent_practice_area} law"], candidate_labels)
        if existing_candidate:
            candidate_source = (candidate_source_map or {}).get(existing_candidate)
            method = "firm_practice_area" if candidate_source == "firm" else "folio_practice_area_match"
            return existing_candidate, 1.0, method
    evidence_tokens = set(remove_junk_tokens(tokenize_text(" ".join([top_terms, top_bigrams, examples, most_frequent_practice_area]))))
    rule_map = [
        ({"divorce", "dissolution", "custody", "support", "parenting", "separation", "child", "children"}, ["Family Law", "Personal and Family Law", "Family"]),
        ({"will", "trust", "probate", "estate", "poa"}, ["Estate Planning", "Trusts And Estates"]),
        ({"immigration", "visa", "permit", "citizenship"}, ["Immigration Law", "Immigration"]),
        ({"employment", "employee", "employer", "termination", "severance"}, ["Employment Law", "Labor And Employment"]),
        ({"lease", "property", "real", "purchase", "sale", "closing"}, ["Real Estate Law", "Real Estate"]),
        ({"corporate", "incorporation", "shareholder", "business", "commercial", "contract", "agreement", "nda"}, ["Business Law", "Corporate Law", "Commercial Law"]),
        ({"trademark", "copyright", "intellectual", "license", "licensing", "ip"}, ["Intellectual Property", "IP Law"]),
        ({"litigation", "dispute", "claim", "motion", "pleading", "court", "settlement"}, ["Litigation", "Civil Litigation"]),
        ({"tax", "irs", "cra"}, ["Tax Law", "Tax"]),
    ]
    for trigger_tokens, preferred_labels in rule_map:
        if evidence_tokens.intersection(trigger_tokens):
            existing_candidate = find_best_existing_candidate(preferred_labels, candidate_labels)
            if existing_candidate:
                return existing_candidate, 0.98, "deterministic_existing_candidate"
    return None, None, None


def classify_clusters_to_taxonomy_levels(cluster_summary: pd.DataFrame, candidate_labels: list[str], candidate_source_map: dict[str, str] | None = None) -> pd.DataFrame:
    cluster_summary = cluster_summary.copy()
    cluster_summary["taxonomy_classification_input"] = cluster_summary.apply(build_taxonomy_level_classification_text, axis=1)
    if cluster_summary.empty or not candidate_labels:
        cluster_summary["predicted_taxonomy_level"] = "Unclassified"
        cluster_summary["taxonomy_level_score"] = np.nan
        cluster_summary["taxonomy_classification_method"] = "unclassified_no_valid_candidate"
        return cluster_summary
    try:
        classifier = get_zero_shot_classifier()
    except Exception as exc:
        st.warning(
            "Taxonomy-level classification could not run because the local classification dependency is unavailable. "
            f"Install the required dependencies or disable this option. Error: {exc}"
        )
        cluster_summary["predicted_taxonomy_level"] = "Unclassified"
        cluster_summary["taxonomy_level_score"] = np.nan
        cluster_summary["taxonomy_classification_method"] = "unclassified_no_valid_candidate"
        return cluster_summary
    predicted_levels = []
    scores = []
    methods = []
    for _, row in cluster_summary.iterrows():
        label, score, method = infer_obvious_taxonomy_level(row, candidate_labels, candidate_source_map)
        if label:
            predicted_levels.append(label)
            scores.append(score)
            methods.append(method)
            continue
        try:
            result = classifier(row.get("taxonomy_classification_input", ""), candidate_labels, multi_label=False)
            predicted_levels.append(result["labels"][0])
            scores.append(float(result["scores"][0]))
            methods.append("zero_shot_valid_candidates")
        except Exception:
            predicted_levels.append("Unclassified")
            scores.append(np.nan)
            methods.append("unclassified_no_valid_candidate")
    cluster_summary["predicted_taxonomy_level"] = predicted_levels
    cluster_summary["taxonomy_level_score"] = scores
    cluster_summary["taxonomy_classification_method"] = methods
    return cluster_summary


def validate_taxonomy_predictions(cluster_summary: pd.DataFrame, valid_taxonomy_labels: list[str]) -> pd.DataFrame:
    if cluster_summary.empty or "predicted_taxonomy_level" not in cluster_summary.columns:
        return cluster_summary
    cluster_summary = cluster_summary.copy()
    valid_label_set = set(valid_taxonomy_labels)
    invalid_mask = ~cluster_summary["predicted_taxonomy_level"].isin(valid_label_set)
    invalid_mask = invalid_mask & cluster_summary["predicted_taxonomy_level"].ne("Unclassified")
    if invalid_mask.any():
        cluster_summary.loc[invalid_mask, "predicted_taxonomy_level"] = "Unclassified"
        cluster_summary.loc[invalid_mask, "taxonomy_level_score"] = np.nan
        cluster_summary.loc[invalid_mask, "taxonomy_classification_method"] = "invalid_taxonomy_label_blocked"
    return cluster_summary


@st.cache_data(show_spinner=False)
def load_template_taxonomy_labels() -> list[str]:
    guideline_path, category_path = get_template_paths()
    labels = []

    for path in [guideline_path, category_path]:
        if path is None:
            continue
        try:
            template_df = pd.read_csv(path)
            template_df.columns = [clean_column_name(col) for col in template_df.columns]
            if "name" in template_df.columns:
                labels.extend(template_df["name"].dropna().astype(str).tolist())
        except Exception:
            continue

    return sorted(set(label.strip() for label in labels if label and label.strip()))


def extract_keywords_from_template_labels(labels: list[str]) -> set[str]:
    keywords = set()
    for label in labels:
        tokens = remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(label)))
        keywords.update(tokens)
        keywords.update(make_ngrams(tokens, 2))
        keywords.update(make_ngrams(tokens, 3))
    return keywords


@st.cache_data(show_spinner=False)
def load_template_category_keyword_map() -> dict:
    """
    Build category-level signal map from template_categories and template_guidelines.
    This keeps the original approach: category names plus linked guideline names seed clustering.
    """
    guideline_path, category_path = get_template_paths()
    if category_path is None or guideline_path is None:
        return {}

    try:
        categories = pd.read_csv(category_path)
        guidelines = pd.read_csv(guideline_path)
    except Exception:
        return {}

    categories.columns = [clean_column_name(col) for col in categories.columns]
    guidelines.columns = [clean_column_name(col) for col in guidelines.columns]

    if "id" not in categories.columns or "name" not in categories.columns:
        return {}
    if "template_category_id" not in guidelines.columns or "name" not in guidelines.columns:
        return {}

    category_signal_map = {}

    for _, category_row in categories.iterrows():
        category_id = category_row["id"]
        category_name = str(category_row["name"]).strip()
        linked_guidelines = (
            guidelines[guidelines["template_category_id"] == category_id]["name"]
            .dropna()
            .astype(str)
            .tolist()
        )

        label_texts = [category_name] + linked_guidelines
        unigrams = set()
        bigrams = set()
        trigrams = set()

        for label_text in label_texts:
            tokens = remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(label_text)))
            unigrams.update(tokens)
            bigrams.update(make_ngrams(tokens, 2))
            trigrams.update(make_ngrams(tokens, 3))

        category_signal_map[category_name] = {
            "unigrams": unigrams,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "all_signals": unigrams.union(bigrams).union(trigrams),
        }

    return category_signal_map


def compute_template_signal_idf(category_signal_map: dict) -> dict[str, float]:
    """
    Compute IDF-like weights for template signals.
    Very common taxonomy words are ignored, and multi-word signals are weighted more strongly.
    """
    if not category_signal_map:
        return {}

    n_categories = len(category_signal_map)
    signal_category_counts = Counter()

    for category_data in category_signal_map.values():
        signals = category_data.get("all_signals")
        if signals is None:
            signals = set(category_data.get("unigrams", set())).union(
                category_data.get("bigrams", set())
            ).union(
                category_data.get("trigrams", set())
            )

        for signal in signals:
            signal_category_counts[signal] += 1

    signal_weights = {}
    for signal, category_count in signal_category_counts.items():
        category_share = category_count / n_categories
        if category_share > TEMPLATE_MAX_TOKEN_CATEGORY_SHARE:
            continue

        idf_weight = np.log((1 + n_categories) / (1 + category_count)) + 1
        n_tokens = len(signal.split())

        if n_tokens == 1:
            ngram_multiplier = 1.0
        elif n_tokens == 2:
            ngram_multiplier = 2.25
        else:
            ngram_multiplier = 3.25

        signal_weights[signal] = float(idf_weight * ngram_multiplier)

    return signal_weights


def match_matter_to_template_category(matter_name: object, category_signal_map: dict, signal_weights: dict | None = None):
    if not category_signal_map:
        return None, 0, []

    signal_weights = signal_weights or {}
    tokens = remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(matter_name)))
    if not tokens:
        return None, 0, []

    matter_unigrams = set(tokens)
    matter_bigrams = set(make_ngrams(tokens, 2))
    matter_trigrams = set(make_ngrams(tokens, 3))
    scored_categories = []

    for category_name, category_data in category_signal_map.items():
        matched_unigrams = matter_unigrams.intersection(category_data["unigrams"])
        matched_bigrams = matter_bigrams.intersection(category_data["bigrams"])
        matched_trigrams = matter_trigrams.intersection(category_data["trigrams"])
        matched_signals = sorted(
            matched_unigrams.union(matched_bigrams).union(matched_trigrams),
            key=lambda x: (len(x.split()), signal_weights.get(x, 0)),
            reverse=True,
        )

        if not matched_signals:
            continue

        weighted_score = sum(signal_weights.get(signal, 0) for signal in matched_signals)
        scored_categories.append({
            "category_name": category_name,
            "weighted_score": weighted_score,
            "n_matched_signals": len(matched_signals),
            "matched_signals": matched_signals,
            "matched_bigrams": sorted(matched_bigrams),
            "matched_trigrams": sorted(matched_trigrams),
        })

    if not scored_categories:
        return None, 0, []

    scored_categories = sorted(
        scored_categories,
        key=lambda x: (
            x["weighted_score"],
            len(x["matched_trigrams"]),
            len(x["matched_bigrams"]),
            x["n_matched_signals"],
        ),
        reverse=True,
    )

    best = scored_categories[0]
    second_score = scored_categories[1]["weighted_score"] if len(scored_categories) > 1 else 0
    margin = best["weighted_score"] - second_score

    has_phrase_match = len(best["matched_bigrams"]) > 0 or len(best["matched_trigrams"]) > 0
    has_enough_score = best["weighted_score"] >= TEMPLATE_MIN_WEIGHTED_SCORE
    has_enough_signals = best["n_matched_signals"] >= TEMPLATE_MIN_MATCHED_SIGNALS
    clearly_better_than_second = margin >= TEMPLATE_SECOND_BEST_MARGIN

    if has_enough_score and has_enough_signals and (has_phrase_match or clearly_better_than_second):
        return best["category_name"], best["weighted_score"], best["matched_signals"]

    return None, best["weighted_score"], best["matched_signals"]


def assign_template_seed_categories(account_input: pd.DataFrame, category_signal_map: dict, signal_weights: dict | None = None) -> pd.DataFrame:
    if account_input.empty:
        return account_input

    matched_rows = account_input["matter_name"].fillna("").apply(
        lambda name: match_matter_to_template_category(name, category_signal_map, signal_weights)
    )

    account_input = account_input.copy()
    account_input["seed_template_category"] = matched_rows.apply(lambda x: x[0])
    account_input["seed_template_category_score"] = matched_rows.apply(lambda x: x[1])
    account_input["seed_template_category_keywords"] = matched_rows.apply(lambda x: ", ".join(x[2]))
    return account_input


def get_candidate_k_values(n_usable_matters: int) -> list[int]:
    if n_usable_matters < 12:
        return []
    if n_usable_matters < 30:
        return [2]
    if n_usable_matters < 75:
        return [2, 3]
    if n_usable_matters < 150:
        return [2, 3, 4, 5]
    if n_usable_matters < 300:
        return [3, 4, 5, 6, 7]
    if n_usable_matters < 750:
        return [4, 5, 6, 7, 8, 10]
    return [5, 6, 8, 10, 12, 15]


def select_best_k_by_silhouette(tfidf_matrix, candidate_ks: list[int], min_cluster_size: int = 5, random_state: int = 42):
    n_rows = tfidf_matrix.shape[0]
    results = []

    for k in candidate_ks:
        if k < 2 or k >= n_rows:
            continue
        try:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix)
            cluster_sizes = np.bincount(labels)
            score = silhouette_score(tfidf_matrix, labels)
            status = "accepted" if cluster_sizes.min() >= min_cluster_size else "accepted_with_small_cluster"
            results.append({
                "k": k,
                "silhouette_score": score,
                "smallest_cluster": int(cluster_sizes.min()),
                "largest_cluster": int(cluster_sizes.max()),
                "status": status,
            })
        except Exception as exc:
            results.append({
                "k": k,
                "silhouette_score": np.nan,
                "smallest_cluster": np.nan,
                "largest_cluster": np.nan,
                "status": f"error: {str(exc)[:80]}",
            })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return None, results_df

    accepted = results_df[results_df["status"].isin(["accepted", "accepted_with_small_cluster"])].copy()
    if accepted.empty:
        return max(candidate_ks) if candidate_ks else None, results_df

    best_row = accepted.sort_values(["silhouette_score", "smallest_cluster"], ascending=[False, False]).iloc[0]
    return int(best_row["k"]), results_df


def estimate_project_keyword_diversity(text_series: pd.Series, legal_keywords: set[str] | None = None) -> dict:
    legal_keywords = legal_keywords or BASE_LEGAL_SERVICE_KEYWORDS
    keyword_hits = []
    bigram_hits = []

    for text in text_series.fillna(""):
        tokens = remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(text)))
        keyword_hits.extend([token for token in tokens if token in legal_keywords])
        bigram_hits.extend(make_ngrams(tokens, 2))

    unique_keyword_count = len(set(keyword_hits))
    frequent_bigram_count = sum(1 for _, count in Counter(bigram_hits).items() if count >= 2)
    diversity_estimate = unique_keyword_count + int(frequent_bigram_count * 0.35)

    return {
        "unique_project_keyword_count": unique_keyword_count,
        "frequent_project_bigram_count": frequent_bigram_count,
        "estimated_project_signal_count": diversity_estimate,
    }


def expand_candidate_ks_with_keyword_diversity(candidate_ks: list[int], keyword_signal_count: int, n_usable_matters: int) -> list[int]:
    if not candidate_ks or not keyword_signal_count:
        return candidate_ks

    keyword_target = int(np.ceil(keyword_signal_count / 2))
    if n_usable_matters < 75:
        max_allowed = 5
    elif n_usable_matters < 150:
        max_allowed = 8
    elif n_usable_matters < 300:
        max_allowed = 12
    elif n_usable_matters < 750:
        max_allowed = 18
    elif n_usable_matters < 1500:
        max_allowed = 25
    else:
        max_allowed = 30

    target_max = min(max(max(candidate_ks), keyword_target), max_allowed)
    target_min = max(2, min(candidate_ks))
    return sorted(set(candidate_ks).union(range(target_min, target_max + 1)))


def expand_candidate_ks_with_seeded_categories(candidate_ks: list[int], seeded_category_count: int, n_usable_matters: int) -> list[int]:
    if not candidate_ks or not seeded_category_count:
        return candidate_ks

    if n_usable_matters < 75:
        max_allowed = 5
    elif n_usable_matters < 150:
        max_allowed = 8
    elif n_usable_matters < 300:
        max_allowed = 12
    elif n_usable_matters < 750:
        max_allowed = 18
    elif n_usable_matters < 1500:
        max_allowed = 25
    else:
        max_allowed = 30

    target_max = min(max(max(candidate_ks), int(seeded_category_count)), max_allowed)
    return sorted(set(candidate_ks).union(range(max(2, min(candidate_ks)), target_max + 1)))


def choose_more_granular_k(k_selection_results: pd.DataFrame, fallback_k: int, keyword_signal_count: int | None = None, n_usable_matters: int | None = None, seeded_category_count: int | None = None) -> int:
    if k_selection_results is None or k_selection_results.empty:
        return fallback_k

    accepted = k_selection_results[k_selection_results["status"].isin(["accepted", "accepted_with_small_cluster"])].copy()
    if accepted.empty:
        return fallback_k

    if keyword_signal_count is None:
        min_desired_k = fallback_k
    else:
        min_desired_k = int(np.ceil(keyword_signal_count / 12))

    if seeded_category_count is not None and seeded_category_count > 0:
        min_desired_k = max(min_desired_k, int(seeded_category_count))

    if n_usable_matters is not None:
        if n_usable_matters >= 1500:
            min_desired_k = max(min_desired_k, 15)
        elif n_usable_matters >= 1000:
            min_desired_k = max(min_desired_k, 12)
        elif n_usable_matters >= 500:
            min_desired_k = max(min_desired_k, 10)
        elif n_usable_matters >= 250:
            min_desired_k = max(min_desired_k, 8)
        elif n_usable_matters >= 100:
            min_desired_k = max(min_desired_k, 5)

    max_accepted_k = int(accepted["k"].max())
    min_desired_k = min(min_desired_k, max_accepted_k)
    granular = accepted[accepted["k"] >= min_desired_k].copy()
    if granular.empty:
        return max_accepted_k

    return int(granular.sort_values("k", ascending=False).iloc[0]["k"])



def infer_canonical_project_label(label: str, top_terms: str = "", examples: str = "") -> str:
    cleaned_label = str(label).strip() if label else "Unclassified Project"
    cleaned_label = re.sub(r"\s+", " ", cleaned_label)
    return cleaned_label.title()


def extract_legal_project_acronyms(value: object) -> list[str]:
    raw_text = str(value or "").lower()
    raw_text = raw_text.replace("m&a", " ma ")
    tokens = re.findall(r"[a-z0-9]+", raw_text)
    acronyms = []
    for token in tokens:
        if token in LEGAL_PROJECT_ACRONYMS:
            acronyms.append(token)
    return acronyms


def simplify_project_name_from_acronyms(project_name: str, cluster_df: pd.DataFrame, matter_name_col: str) -> str:
    project_acronyms = extract_legal_project_acronyms(project_name)
    if project_acronyms:
        chosen = sorted(project_acronyms, key=lambda token: (len(token), token), reverse=True)[0]
        return LEGAL_PROJECT_ACRONYMS[chosen]

    if matter_name_col not in cluster_df.columns:
        return project_name

    acronym_counts = Counter()
    for value in cluster_df[matter_name_col].fillna("").astype(str):
        acronym_counts.update(extract_legal_project_acronyms(value))
    if not acronym_counts:
        return project_name

    chosen = sorted(
        acronym_counts.items(),
        key=lambda item: (item[1], len(item[0]), item[0]),
        reverse=True,
    )[0][0]
    return LEGAL_PROJECT_ACRONYMS[chosen]


def infer_specific_subproject_label(parent_project: str, label: str, top_terms: str = "", examples: str = "") -> str:
    cleaned_label = str(label).strip() if label else ""
    cleaned_parent = str(parent_project).strip()

    if cleaned_label and normalize_legal_text_for_clustering(cleaned_label) != normalize_legal_text_for_clustering(cleaned_parent):
        return re.sub(r"\s+", " ", cleaned_label).title()

    unigram_terms = [term.strip() for term in top_terms.split(",") if term.strip() and " " not in term.strip()]
    bigram_terms = [term.strip() for term in top_terms.split(",") if term.strip() and " " in term.strip()]
    suggested = suggest_project_name(unigram_terms, bigram_terms)

    if suggested and normalize_legal_text_for_clustering(suggested) != normalize_legal_text_for_clustering(cleaned_parent):
        return suggested.title()

    for candidate in unigram_terms + bigram_terms:
        if normalize_legal_text_for_clustering(candidate) != normalize_legal_text_for_clustering(cleaned_parent):
            return candidate.title()

    return f"Other {cleaned_parent}".strip().title()


def clean_subproject_display_name(subproject_name: str) -> str:
    return re.sub(r"\s*·\s*Subcluster\s*\d+\s*$", "", str(subproject_name)).strip()


def get_top_terms_from_text(text_series: pd.Series, ngram_n: int = 1, top_n: int = 10) -> list[str]:
    all_terms = []
    for text in text_series.fillna(""):
        tokens = remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(text)))
        all_terms.extend(make_ngrams(tokens, ngram_n))
    return [term for term, _ in Counter(all_terms).most_common(top_n)]


def format_examples(values: pd.Series, n: int = 10) -> str:
    return " | ".join(
        values.dropna().astype(str).map(str.strip).loc[lambda s: s != ""].drop_duplicates().head(n).tolist()
    )


def format_examples_for_label(values: pd.Series, label: str, top_terms: list[str] | None = None, n: int = 10) -> str:
    """Prioritize examples that resemble the generated title/top terms."""
    examples = (
        values.dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )

    if not examples:
        return ""

    label_tokens = set(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(label))))
    term_tokens = set()

    for term in top_terms or []:
        term_tokens.update(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(term))))

    signal_tokens = label_tokens.union(term_tokens)

    def score_example(example: str) -> tuple[int, int, int]:
        example_tokens = set(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(example))))
        label_overlap = len(example_tokens.intersection(label_tokens))
        total_overlap = len(example_tokens.intersection(signal_tokens))
        return (label_overlap, total_overlap, -len(example))

    ranked_examples = sorted(examples, key=score_example, reverse=True)
    return " | ".join(ranked_examples[:n])


def suggest_project_name(top_terms: list[str], top_bigrams: list[str]) -> str:
    legal_bigrams = [bigram for bigram in top_bigrams if any(token in BASE_LEGAL_SERVICE_KEYWORDS for token in bigram.split())]
    if legal_bigrams:
        return legal_bigrams[0].title()

    legal_terms = [term for term in top_terms if term in BASE_LEGAL_SERVICE_KEYWORDS]
    if len(legal_terms) >= 2:
        return " ".join(legal_terms[:2]).title()
    if len(legal_terms) == 1:
        return legal_terms[0].title()
    if top_bigrams:
        return top_bigrams[0].title()
    if top_terms:
        return " ".join(top_terms[:3]).title()
    return "Unclear / Needs Review"


def fallback_project_name(top_terms: list[str], example_names: list[str]) -> str:
    top_bigrams = [term for term in top_terms if " " in term]
    unigrams = [term for term in top_terms if " " not in term]
    suggested = suggest_project_name(unigrams, top_bigrams)
    return re.sub(r"\s+", " ", suggested).strip().title()


def safe_json_loads(raw_text: str) -> dict:
    try:
        return json.loads(raw_text)
    except Exception:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def call_ollama_project_label(top_terms: list[str], examples: list[str]) -> str | None:
    examples_text = "\n- ".join(examples[:8])
    taxonomy_labels = load_template_taxonomy_labels()
    taxonomy_preview = ", ".join(taxonomy_labels[:120]) if taxonomy_labels else ""

    prompt = f"""
You are labeling a cluster of legal matters.
Return JSON only with this format: {{"project_name": "short reusable legal project type"}}.
Prefer an existing AltFee taxonomy label when it fits.
Do not use client names. Do not use vague labels like General Legal Work.

Available taxonomy labels:
{taxonomy_preview}

Top terms: {', '.join(top_terms[:12])}
Example matter names:
- {examples_text}
"""

    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        parsed = safe_json_loads(data.get("response", ""))
        project_name = parsed.get("project_name")
        if project_name and isinstance(project_name, str):
            return re.sub(r"\s+", " ", project_name.strip()).title()
    except Exception:
        return None

    return None


def choose_cluster_count(n_matters: int) -> int:
    candidate_ks = get_candidate_k_values(n_matters)
    if not candidate_ks:
        return 1
    return max(candidate_ks)


def choose_subcluster_count(n_matters: int) -> int:
    if n_matters < 50:
        return 1
    if n_matters < 100:
        return 2
    if n_matters < 250:
        return 3
    if n_matters < 500:
        return 4
    return 5


def summarize_practice_area_context(cluster_df: pd.DataFrame, practice_area_col: str | None) -> dict[str, object]:
    if not practice_area_col or practice_area_col not in cluster_df.columns or cluster_df.empty:
        return {"practice_area_null_percentage": np.nan, "most_frequent_practice_area": ""}
    practice_area_values = cluster_df[practice_area_col]
    non_null_values = practice_area_values.dropna().astype(str).map(str.strip).loc[lambda s: s != ""]
    null_mask = practice_area_values.isna() | practice_area_values.astype(str).str.strip().eq("")
    most_frequent = non_null_values.mode().iloc[0] if not non_null_values.empty else ""
    return {
        "practice_area_null_percentage": round(float(null_mask.mean() * 100), 2),
        "most_frequent_practice_area": most_frequent,
    }


def value_has_legal_signal(value: object) -> bool:
    tokens = set(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(value))))
    if tokens.intersection(BASE_LEGAL_SERVICE_KEYWORDS):
        return True
    phrase_keys = extract_consecutive_phrase_keys(value)
    return bool(phrase_keys.intersection(CLUSTER_ROOT_PHRASES))


def cluster_text_has_project_name(cluster_df: pd.DataFrame, project_name: str, columns: list[str | None]) -> bool:
    project_text = normalize_legal_text_for_clustering(project_name)
    if not project_text:
        return False
    for column in columns:
        if not column or column not in cluster_df.columns:
            continue
        for value in cluster_df[column].dropna().astype(str):
            value_text = normalize_legal_text_for_clustering(value)
            if f" {project_text} " in f" {value_text} ":
                return True
    return False


def has_useful_structured_project_signal(cluster_df: pd.DataFrame, columns: list[str | None]) -> bool:
    for column in columns:
        if not column or column not in cluster_df.columns:
            continue
        if cluster_df[column].dropna().astype(str).apply(value_has_legal_signal).any():
            return True
    return False


def suggest_project_name_from_time_entries(cluster_df: pd.DataFrame, text_col: str | None) -> str | None:
    if not text_col or text_col not in cluster_df.columns:
        return None
    top_terms = get_top_terms_from_text(cluster_df[text_col], ngram_n=1, top_n=12)
    top_bigrams = get_top_terms_from_text(cluster_df[text_col], ngram_n=2, top_n=12)
    top_trigrams = get_top_terms_from_text(cluster_df[text_col], ngram_n=3, top_n=8)
    legal_phrases = [
        phrase for phrase in top_bigrams + top_trigrams
        if any(token in BASE_LEGAL_SERVICE_KEYWORDS for token in phrase.split())
    ]
    if legal_phrases:
        return legal_phrases[0].title()
    legal_terms = [term for term in top_terms if term in BASE_LEGAL_SERVICE_KEYWORDS]
    if legal_terms:
        return suggest_project_name(legal_terms, top_bigrams).title()
    return None


def is_broad_practice_area_project_name(project_name: str, practice_area_value: object = "") -> bool:
    project_key = normalize_legal_text_for_clustering(project_name)
    if not project_key:
        return False
    if project_key in BROAD_PRACTICE_AREA_PROJECT_NAMES:
        return True
    practice_key = normalize_legal_text_for_clustering(practice_area_value)
    if practice_key and (project_key == practice_key or f" {project_key} " in f" {practice_key} "):
        return not label_has_project_like_terms(project_name)
    return False


def refine_project_name_with_context(
    project_name: str,
    cluster_df: pd.DataFrame,
    matter_name_col: str,
    practice_area_col: str | None,
    matter_category_col: str | None,
    text_col: str | None,
) -> str:
    structured_cols = [matter_name_col, practice_area_col, matter_category_col]
    practice_area_value = ""
    if practice_area_col and practice_area_col in cluster_df.columns:
        values = cluster_df[practice_area_col].dropna().astype(str).map(str.strip).loc[lambda s: s != ""]
        practice_area_value = values.mode().iloc[0] if not values.empty else ""

    if is_broad_practice_area_project_name(project_name, practice_area_value):
        time_entry_name = suggest_project_name_from_time_entries(cluster_df, text_col)
        if time_entry_name and normalize_legal_text_for_clustering(time_entry_name) != normalize_legal_text_for_clustering(project_name):
            return time_entry_name
        return "Unclear / Needs Review"

    count_col = "matter_id_for_count" if "matter_id_for_count" in cluster_df.columns else matter_name_col
    is_tiny_cluster = cluster_df[count_col].nunique() <= 2
    direct_project_signal = cluster_text_has_project_name(cluster_df, project_name, structured_cols)
    useful_structured_signal = has_useful_structured_project_signal(cluster_df, structured_cols)
    if is_tiny_cluster and not direct_project_signal and not useful_structured_signal:
        time_entry_name = suggest_project_name_from_time_entries(cluster_df, text_col)
        if time_entry_name:
            return time_entry_name

    return project_name


def analyze_time_entry_keywords(value: object, top_n: int = 8) -> dict[str, list[str]]:
    tokens = remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(value)))
    return {
        "top_keywords": [term for term, _ in Counter(tokens).most_common(top_n)],
        "top_phrases": [term for term, _ in Counter(make_ngrams(tokens, 2)).most_common(top_n)],
    }


SUBCLUSTER_MIN_SHARE = 0.25
SUBCLUSTER_MIN_COUNT = 3
SUBCLUSTER_BROAD_SINGLE_TOKENS = {"child", "children", "kid", "kids"}


def normalize_branch_text(value: object) -> str:
    text = clean_text(value)
    replacements = {
        "with child": "with children",
        "with kids": "with children",
        "with kid": "with children",
        "w children": "with children",
        "w child": "with children",
        "without child": "without children",
        "without kids": "without children",
        "without kid": "without children",
        "no child": "without children",
        "no children": "without children",
        "no kids": "without children",
        "no kid": "without children",
        "parent plan": "parenting plan",
        "parental plan": "parenting plan",
        "parenting plans": "parenting plan",
    }
    for source, target in replacements.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    return re.sub(r"\s+", " ", text).strip()


def branch_phrase_keys_for_matter(value: object, parent_project: str) -> set[str]:
    text = normalize_branch_text(value)
    if not text:
        return set()
    parent_root = get_project_root_signal(normalize_matter_name_signature(parent_project))
    normalized_parent_project = normalize_branch_text(parent_project)
    normalized_parent_signature = normalize_branch_text(normalize_matter_name_signature(parent_project))
    normalized_parent_root = normalize_branch_text(parent_root)
    parent_tokens = set()
    for source in [normalized_parent_project, normalized_parent_signature, normalized_parent_root]:
        for token in tokenize_text(source):
            parent_tokens.add(token)
            parent_tokens.add(normalize_phrase_variant_token(token))

    def is_parent_duplicate(branch_key: str) -> bool:
        normalized_branch_key = normalize_branch_text(branch_key)
        if not normalized_branch_key:
            return True
        for parent_value in [normalized_parent_project, normalized_parent_signature, normalized_parent_root]:
            if not parent_value:
                continue
            if normalized_branch_key == parent_value:
                return True
            if f" {normalized_branch_key} " in f" {parent_value} ":
                return True
            if f" {parent_value} " in f" {normalized_branch_key} ":
                return True
        return False

    branch_keys = set()
    for phrase in ["with children", "without children", "contested", "uncontested"]:
        if re.search(rf"\b{re.escape(phrase)}\b", text) and not is_parent_duplicate(phrase):
            branch_keys.add(phrase)
    tokens = [
        normalize_phrase_variant_token(token)
        for token in tokenize_text(text)
        if token not in parent_tokens and token not in PROJECT_STOPWORDS and not token.isdigit() and len(token) > 2
    ]
    for n in [3, 2, 1]:
        if len(tokens) < n:
            continue
        for index in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[index:index + n])
            if n == 1 and phrase in SUBCLUSTER_BROAD_SINGLE_TOKENS:
                continue
            if is_parent_duplicate(phrase):
                continue
            branch_keys.add(phrase)
    return branch_keys


def create_keyword_subclusters(
    cluster_df: pd.DataFrame,
    parent_project: str,
    matter_col: str,
    matter_name_col: str,
    billing_col: str,
    min_share: float = SUBCLUSTER_MIN_SHARE,
    min_count: int = SUBCLUSTER_MIN_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cluster_df.empty or matter_name_col not in cluster_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    working = cluster_df.copy()
    count_col = "matter_id_for_count" if "matter_id_for_count" in working.columns else matter_col
    total_matters = max(working[count_col].nunique(), 1)
    row_branch_keys = working[matter_name_col].apply(lambda value: branch_phrase_keys_for_matter(value, parent_project))
    branch_to_matter_ids: dict[str, set] = {}
    for matter_id, keys in zip(working[count_col], row_branch_keys):
        for key in keys:
            branch_to_matter_ids.setdefault(key, set()).add(matter_id)
    parent_root = get_project_root_signal(normalize_matter_name_signature(parent_project))
    candidates = []
    for branch_key, matter_ids in branch_to_matter_ids.items():
        if branch_key == parent_root:
            continue
        branch_count = len(matter_ids)
        branch_share = branch_count / total_matters
        if branch_count >= min_count and branch_share >= min_share:
            candidates.append({"branch_key": branch_key, "matter_count": branch_count, "matter_share": branch_share})
    if not candidates:
        return pd.DataFrame(), pd.DataFrame()
    candidates = sorted(candidates, key=lambda item: (item["matter_share"], len(item["branch_key"].split()), item["matter_count"]), reverse=True)

    def assign_branch(keys: set[str]) -> str:
        for candidate in candidates:
            if candidate["branch_key"] in keys:
                return candidate["branch_key"]
        return "Other"

    working["subcluster_name"] = row_branch_keys.apply(assign_branch)
    working["subcluster_id"] = pd.Categorical(
        working["subcluster_name"],
        categories=[candidate["branch_key"] for candidate in candidates] + ["Other"],
        ordered=True,
    ).codes
    summary_rows = []
    for subcluster_name, sub_df in working.groupby("subcluster_name", sort=False):
        if subcluster_name == "Other":
            continue
        numeric_billing = pd.to_numeric(sub_df[billing_col], errors="coerce").fillna(0)
        candidate_data = next(candidate for candidate in candidates if candidate["branch_key"] == subcluster_name)
        summary_rows.append({
            "subcluster_id": int(sub_df["subcluster_id"].iloc[0]),
            "subcluster_name": subcluster_name,
            "display_label": subcluster_name,
            "matter_count": sub_df[count_col].nunique(),
            "matter_share": candidate_data["matter_share"],
            "total_billing": numeric_billing.sum(),
            "avg_billing": numeric_billing.mean(),
            "top_terms": ", ".join(get_top_terms_from_text(sub_df[matter_name_col], ngram_n=1, top_n=10)),
            "top_bigrams": ", ".join(get_top_terms_from_text(sub_df[matter_name_col], ngram_n=2, top_n=10)),
            "example_matter_names": format_examples(sub_df[matter_name_col], n=10),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["matter_share", "matter_count"], ascending=[False, False])
    return summary, working


def create_project_clusters(
    df: pd.DataFrame,
    matter_name_col: str,
    billing_col: str,
    use_ai: bool,
    progress_bar=None,
    status_box=None,
    text_col: str | None = None,
    practice_area_col: str | None = None,
    matter_category_col: str | None = None,
    cluster_prefix: str = "cluster",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    source_col = text_col if text_col and text_col in working.columns else matter_name_col
    cluster_col = f"{cluster_prefix}_id"
    label_col = "project_name" if cluster_prefix == "cluster" else "subcluster_name"
    parent_project_name = None
    if cluster_prefix == "subcluster" and "project_name" in working.columns and working["project_name"].notna().any():
        parent_project_name = working["project_name"].mode().iloc[0]

    working["matter_name"] = working[matter_name_col]
    working["project_signature"] = working[matter_name_col].fillna("").apply(normalize_matter_name_signature)

    if cluster_prefix == "cluster":
        # Main clustering should be driven by normalized matter-name signatures.
        # This prevents repeated formats like "divorce with children" from being split or renamed by unrelated templates.
        working["primary_project_text"] = working["project_signature"]
    else:
        # Sub-projects should follow the same signature-first logic.
        # Time-entry text is useful context, but if it becomes the main driver it can randomly split similar matters.
        normalized_source = working[source_col].fillna("").apply(normalize_legal_text_for_clustering)
        working["primary_project_text"] = np.where(
            working["project_signature"].astype(str).str.len() >= 3,
            working["project_signature"],
            normalized_source,
        )
        working["secondary_project_text"] = normalized_source

    working = working[working["primary_project_text"].astype(str).str.len() >= 3].copy().reset_index(drop=True)

    if working.empty:
        empty_summary_cols = [
            cluster_col,
            label_col,
            "matter_count",
            "total_billing",
            "avg_billing",
            "top_terms",
            "top_bigrams",
            "example_matter_names",
            "practice_area_null_percentage",
            "most_frequent_practice_area",
            "cluster_root_signal",
            "cluster_root_purity",
            "cluster_purity_note",
        ]
        return pd.DataFrame(columns=empty_summary_cols), pd.DataFrame()

    count_col = "matter_id_for_count" if "matter_id_for_count" in working.columns else matter_name_col
    n_usable = working[count_col].nunique()

    if cluster_prefix == "subcluster":
        unique_signature_count = working["project_signature"].nunique(dropna=True)
        k_selected = min(choose_subcluster_count(n_usable), max(1, unique_signature_count))
        candidate_ks = [k_selected] if k_selected > 1 else []
        k_selection_results = pd.DataFrame()
        seeded_category_count = 0
        keyword_diversity = estimate_project_keyword_diversity(working[matter_name_col])
    else:
        if status_box is not None:
            status_box.info("Loading template taxonomy and matching seed categories...")
        template_labels = load_template_taxonomy_labels()
        template_keywords = extract_keywords_from_template_labels(template_labels)
        legal_keywords = BASE_LEGAL_SERVICE_KEYWORDS.union(template_keywords)
        template_category_keywords = load_template_category_keyword_map()
        template_signal_weights = compute_template_signal_idf(template_category_keywords)
        working = assign_template_seed_categories(working, template_category_keywords, template_signal_weights)
        seeded_category_count = working["seed_template_category"].nunique(dropna=True)
        candidate_ks = get_candidate_k_values(n_usable)

        # Removed premature expansion with keyword diversity here.

        if not candidate_ks:
            k_selected = 1
            k_selection_results = pd.DataFrame()
        else:
            keyword_diversity = estimate_project_keyword_diversity(working[matter_name_col], legal_keywords)
            candidate_ks = expand_candidate_ks_with_keyword_diversity(
                candidate_ks,
                keyword_diversity["estimated_project_signal_count"],
                n_usable,
            )
            candidate_ks = expand_candidate_ks_with_seeded_categories(candidate_ks, seeded_category_count, n_usable)

    if progress_bar is not None:
        progress_bar.progress(20)
    if status_box is not None:
        status_box.info("Building TF-IDF vectors with legal normalization...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2 if len(working) >= 30 else 1,
        max_df=0.5 if len(working) >= 30 else 1.0,
        stop_words=list(ALL_STOP_WORDS),
        sublinear_tf=True,
        norm="l2",
    )

    try:
        X = vectorizer.fit_transform(working["primary_project_text"])
    except Exception:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words=list(ALL_STOP_WORDS),
            sublinear_tf=True,
            norm="l2",
        )
        X = vectorizer.fit_transform(working["primary_project_text"])

    if progress_bar is not None:
        progress_bar.progress(45)
    if status_box is not None:
        status_box.info("Selecting cluster count using silhouette diagnostics...")

    if cluster_prefix == "cluster" and candidate_ks:
        silhouette_k, k_selection_results = select_best_k_by_silhouette(
            X,
            candidate_ks,
            min_cluster_size=MIN_FIRST_PASS_CLUSTER_SIZE,
            random_state=42,
        )
        k_selected = silhouette_k or max(candidate_ks)
        if PREFER_MORE_FIRST_PASS_CLUSTERS:
            k_selected = choose_more_granular_k(
                k_selection_results,
                fallback_k=k_selected,
                keyword_signal_count=keyword_diversity["estimated_project_signal_count"],
                n_usable_matters=n_usable,
                seeded_category_count=seeded_category_count,
            )

    if k_selected <= 1 or len(working) < 2:
        working[cluster_col] = 0
    else:
        k_selected = min(k_selected, len(working) - 1)
        model = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
        working[cluster_col] = model.fit_predict(X)

    if cluster_prefix == "cluster":
        working = force_flat_fee_cluster(working, cluster_col, matter_name_col)
        working = enforce_cluster_root_purity(working, cluster_col)

    if progress_bar is not None:
        progress_bar.progress(65)
    if status_box is not None:
        status_box.info("Naming clusters and building summaries...")

    cluster_rows = []
    for index, cluster_id in enumerate(sorted(working[cluster_col].unique())):
        cluster_df = working[working[cluster_col] == cluster_id].copy()
        top_terms = get_top_terms_from_text(cluster_df["primary_project_text"], ngram_n=1, top_n=10)
        top_bigrams = get_top_terms_from_text(cluster_df["primary_project_text"], ngram_n=2, top_n=10)
        raw_examples = cluster_df[matter_name_col].dropna().astype(str).tolist()
        root_signal = (
            cluster_df["cluster_root_signal"].dropna().astype(str).mode().iloc[0]
            if "cluster_root_signal" in cluster_df.columns and not cluster_df["cluster_root_signal"].dropna().empty
            else ""
        )
        root_purity = (
            float((cluster_df["cluster_root_signal"] == root_signal).mean())
            if root_signal and "cluster_root_signal" in cluster_df.columns
            else np.nan
        )
        purity_notes = (
            " | ".join(cluster_df["cluster_purity_note"].dropna().astype(str).loc[lambda s: s != ""].drop_duplicates().tolist())
            if "cluster_purity_note" in cluster_df.columns
            else ""
        )

        combined_terms = ", ".join(top_terms + top_bigrams)
        combined_examples = " | ".join(raw_examples[:10])

        signature_name, signature_share = root_first_cluster_label(cluster_df[matter_name_col], min_share=0.35)

        project_name = FLAT_FEE_PROJECT_NAME if root_signal == "flat fee" else signature_name

        if not project_name and use_ai:
            project_name = call_ollama_project_label(top_terms + top_bigrams, raw_examples[:10])

        if not project_name:
            dominant_seed = None
            if "seed_template_category" in cluster_df.columns and cluster_df["seed_template_category"].notna().any():
                seed_counts = cluster_df["seed_template_category"].dropna().value_counts(normalize=True)
                if not seed_counts.empty and seed_counts.iloc[0] >= 0.75:
                    dominant_seed = seed_counts.index[0]
            project_name = dominant_seed or fallback_project_name(top_terms + top_bigrams, raw_examples[:10])

        if not validate_cluster_label_against_examples(project_name, cluster_df[matter_name_col]):
            fallback_signature_name, fallback_signature_share = root_first_cluster_label(cluster_df[matter_name_col], min_share=0.25)
            if fallback_signature_name:
                project_name = fallback_signature_name
                signature_share = fallback_signature_share
            else:
                project_name = fallback_project_name(top_terms + top_bigrams, raw_examples[:10])

        if cluster_prefix == "subcluster" and parent_project_name:
            project_name = infer_specific_subproject_label(parent_project_name, project_name, combined_terms, combined_examples)
            display_label = project_name
        else:
            project_name = infer_canonical_project_label(project_name, combined_terms, combined_examples)
            project_name = refine_project_name_with_context(
                project_name,
                cluster_df,
                matter_name_col=matter_name_col,
                practice_area_col=practice_area_col,
                matter_category_col=matter_category_col,
                text_col=text_col,
            )
            project_name = simplify_project_name_from_acronyms(project_name, cluster_df, matter_name_col)
            display_label = f"{project_name} · {cluster_prefix.title()} {cluster_id}"

        examples = format_examples_for_label(
            cluster_df[matter_name_col],
            label=project_name,
            top_terms=top_terms + top_bigrams,
            n=10,
        ).split(" | ")

        working.loc[cluster_df.index, label_col] = project_name
        working.loc[cluster_df.index, "display_label"] = display_label
        numeric_billing = pd.to_numeric(cluster_df[billing_col], errors="coerce").fillna(0)
        practice_area_context = summarize_practice_area_context(cluster_df, practice_area_col)

        cluster_rows.append({
            cluster_col: cluster_id,
            label_col: project_name,
            "display_label": display_label,
            "matter_count": cluster_df[count_col].nunique(),
            "total_billing": numeric_billing.sum(),
            "avg_billing": numeric_billing.mean(),
            "top_terms": ", ".join(top_terms),
            "top_bigrams": ", ".join(top_bigrams),
            "example_matter_names": " | ".join(examples),
            "dominant_signature_share": signature_share,
            "practice_area_null_percentage": practice_area_context["practice_area_null_percentage"],
            "most_frequent_practice_area": practice_area_context["most_frequent_practice_area"],
            "cluster_root_signal": root_signal,
            "cluster_root_purity": root_purity,
            "cluster_purity_note": purity_notes,
        })

        if progress_bar is not None:
            progress_value = 65 + int(((index + 1) / max(1, len(working[cluster_col].unique()))) * 30)
            progress_bar.progress(min(progress_value, 95))

    if progress_bar is not None:
        progress_bar.progress(100)
    if status_box is not None:
        status_box.success("Clustering complete.")

    if cluster_prefix == "cluster":
        working = enforce_top_project_name_purity(
            working,
            label_col=label_col,
            matter_name_col=matter_name_col,
            count_col=count_col,
            text_col=text_col,
        )
        cluster_summary = rebuild_cluster_summary_from_project_names(
            working,
            cluster_col=cluster_col,
            label_col=label_col,
            matter_name_col=matter_name_col,
            billing_col=billing_col,
            count_col=count_col,
            practice_area_col=practice_area_col,
        )
    else:
        cluster_summary = pd.DataFrame(cluster_rows)
    if not cluster_summary.empty:
        cluster_summary["flat_fee_sort"] = cluster_summary[label_col].map(is_flat_fee_project)
        cluster_summary["others_sort"] = cluster_summary[label_col].map(is_others_project)
        cluster_summary = (
            cluster_summary
            .sort_values(["flat_fee_sort", "others_sort", "matter_count", "total_billing"], ascending=[True, True, False, False])
            .drop(columns=["flat_fee_sort", "others_sort"])
        )
    return cluster_summary, working


def add_numeric_columns(df: pd.DataFrame, billing_col: str, hours_col: str | None, rate_col: str | None, entries_col: str | None, users_col: str | None) -> pd.DataFrame:
    working = df.copy()
    working["numeric_billing"] = pd.to_numeric(working[billing_col], errors="coerce").fillna(0)

    if hours_col:
        working["numeric_hours"] = pd.to_numeric(working[hours_col], errors="coerce")
    else:
        working["numeric_hours"] = np.nan

    if rate_col:
        working["numeric_rate"] = pd.to_numeric(working[rate_col], errors="coerce")
    else:
        working["numeric_rate"] = np.nan

    if entries_col:
        working["numeric_time_entries"] = pd.to_numeric(working[entries_col], errors="coerce")
    else:
        working["numeric_time_entries"] = np.nan

    if users_col:
        working["numeric_unique_users"] = pd.to_numeric(working[users_col], errors="coerce")
    else:
        working["numeric_unique_users"] = np.nan

    working["effective_hourly_rate"] = np.where(
        working["numeric_hours"] > 0,
        working["numeric_billing"] / working["numeric_hours"],
        np.nan,
    )
    return working


def coefficient_of_variation(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or clean.mean() == 0:
        return np.nan
    return clean.std() / clean.mean()


# --- Variability interpretation and card rendering ---
def interpret_variability_coefficient(cv_value: float) -> dict[str, str]:
    if pd.isna(cv_value):
        return {
            "label": "Not available",
            "description": "Not enough hour data to estimate variability.",
            "color": ALTFEE_GRAY_DARK,
        }

    if cv_value < 0.60:
        return {
            "label": "Low variability",
            "description": "Effort is relatively predictable across matters.",
            "color": INDICATOR_GOOD,
        }

    if cv_value <= 1.20:
        return {
            "label": "Medium variability",
            "description": "Effort changes meaningfully between matters.",
            "color": INDICATOR_OK,
        }

    return {
        "label": "High variability",
        "description": "Inconsistent effort, scope the project more carefully.",
        "color": INDICATOR_BAD,
    }


def render_variability_card(cv_value: float) -> None:
    interpretation = interpret_variability_coefficient(cv_value)
    value = "N/A" if pd.isna(cv_value) else f"{cv_value:.2f}"

    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 7px solid {interpretation['color']};">
            <div class="metric-label">Hours Variability</div>
            <div style="display: flex; align-items: baseline; gap: 0.65rem; margin-bottom: 0.15rem;">
                <span class="metric-value" style="color: {interpretation['color']}; margin-bottom: 0;">{value}</span>
                <span style="color: {interpretation['color']}; font-size: 1rem; font-weight: 800;">{interpretation['label']}</span>
            </div>
            <div class="metric-help">{interpretation['description']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summarize_cluster_detail(cluster_df: pd.DataFrame) -> dict[str, float]:
    return {
        "matters": cluster_df["matter_id_for_count"].nunique() if "matter_id_for_count" in cluster_df.columns else len(cluster_df),
        "total_billing": cluster_df["numeric_billing"].sum(),
        "avg_billing": cluster_df["numeric_billing"].mean(),
        "total_hours": cluster_df["numeric_hours"].sum(skipna=True),
        "avg_hours": cluster_df["numeric_hours"].mean(skipna=True),
        "effective_rate": cluster_df["effective_hourly_rate"].mean(skipna=True),
        "hours_cv": coefficient_of_variation(cluster_df["numeric_hours"]),
    }


def build_cluster_distribution_chart(cluster_summary: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        cluster_summary.head(12),
        x="project_name",
        y="matter_count",
        hover_data=["total_billing", "avg_billing", "top_terms"],
        title="Top Clusters by Matter Count",
    )
    fig.update_layout(height=430, xaxis_title="", yaxis_title="Matters", plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(tickangle=-35)
    return fig


def build_cluster_billing_chart(cluster_summary: pd.DataFrame) -> go.Figure:
    billing_summary = cluster_summary.sort_values("total_billing", ascending=False).head(12)
    fig = px.bar(
        billing_summary,
        x="project_name",
        y="total_billing",
        hover_data=["matter_count", "avg_billing", "top_terms"],
        title="Top Clusters by Total Billing",
    )
    fig.update_layout(height=430, xaxis_title="", yaxis_title="Total Billing", plot_bgcolor="white", paper_bgcolor="white")
    fig.update_yaxes(tickprefix="$")
    fig.update_xaxes(tickangle=-35)
    return fig


def build_cumulative_coverage_chart(cluster_summary: pd.DataFrame) -> go.Figure:
    coverage = cluster_summary.sort_values("matter_count", ascending=False).copy()
    coverage["cluster_rank"] = np.arange(1, len(coverage) + 1)
    total_matters = coverage["matter_count"].sum()
    coverage["cumulative_coverage"] = coverage["matter_count"].cumsum() / total_matters * 100 if total_matters else 0

    fig = px.line(
        coverage,
        x="cluster_rank",
        y="cumulative_coverage",
        markers=True,
        title="Cumulative Matter Coverage by Cluster Rank",
    )
    fig.update_layout(height=380, xaxis_title="Cluster Rank", yaxis_title="Cumulative Coverage (%)", plot_bgcolor="white", paper_bgcolor="white")
    fig.update_yaxes(range=[0, 105], ticksuffix="%")
    return fig


def build_hours_histogram(cluster_df: pd.DataFrame, title: str) -> go.Figure:
    clean = cluster_df.dropna(subset=["numeric_hours"])
    fig = px.histogram(clean, x="numeric_hours", nbins=25, title=title)
    fig.update_layout(height=380, xaxis_title="Total Hours", yaxis_title="Matters", plot_bgcolor="white", paper_bgcolor="white")
    return fig


def build_hours_boxplot(cluster_df: pd.DataFrame, title: str) -> go.Figure:
    clean = cluster_df.dropna(subset=["numeric_hours"])
    fig = px.box(clean, y="numeric_hours", points="outliers", title=title)
    fig.update_layout(height=380, yaxis_title="Total Hours", plot_bgcolor="white", paper_bgcolor="white")
    return fig


def build_billing_hours_scatter(cluster_df: pd.DataFrame, title: str) -> go.Figure:
    clean = cluster_df.dropna(subset=["numeric_hours"])
    fig = px.scatter(
        clean,
        x="numeric_hours",
        y="numeric_billing",
        hover_data=["project_name"],
        title=title,
    )
    fig.update_layout(height=420, xaxis_title="Total Hours", yaxis_title="Total Billing", plot_bgcolor="white", paper_bgcolor="white")
    fig.update_yaxes(tickprefix="$")
    return fig



def build_subcluster_summary_chart(subcluster_summary: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    fig = px.bar(subcluster_summary.sort_values(value_col, ascending=False), x="subcluster_name", y=value_col, hover_data=["matter_count", "total_billing", "avg_billing", "top_terms", "top_bigrams"], title=title)
    fig.update_layout(height=400, xaxis_title="", plot_bgcolor="white", paper_bgcolor="white")
    if value_col in ["total_billing", "avg_billing"]:
        fig.update_yaxes(tickprefix="$")
    fig.update_xaxes(tickangle=-30)
    return fig


# === New project-level charting and ranking functions ===
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_project_color_map(project_names: list[str]) -> dict[str, str]:
    color_map = {}
    for index, project_name in enumerate(project_names[:3]):
        color_map[project_name] = PROJECT_BASE_COLORS[index % len(PROJECT_BASE_COLORS)]
    color_map["Other"] = "#E6E6E6"
    return color_map


def get_subproject_tones(base_color: str, n: int) -> list[str]:
    if n <= 1:
        return [base_color]
    alpha_values = np.linspace(0.95, 0.35, n)
    return [hex_to_rgba(base_color, float(alpha)) for alpha in alpha_values]


def aggregate_project_categories(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "total_matters": ("matter_count", "sum"),
        "total_billing": ("total_billing", "sum"),
        "n_clusters": ("cluster_id", "nunique"),
    }

    if "predicted_taxonomy_level" in cluster_summary.columns:
        agg_spec["predicted_taxonomy_level"] = (
            "predicted_taxonomy_level",
            lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "Unclassified",
        )

    if "taxonomy_level_score" in cluster_summary.columns:
        agg_spec["taxonomy_level_score"] = ("taxonomy_level_score", "max")

    if "folio_verified" in cluster_summary.columns:
        agg_spec["folio_verified"] = ("folio_verified", "max")

    for folio_col in ["folio_title", "folio_description", "folio_class_id", "folio_match_method", "folio_api_status", "folio_search_url"]:
        if folio_col in cluster_summary.columns:
            agg_spec[folio_col] = (
                folio_col,
                lambda s: s.dropna().astype(str).loc[lambda values: values != ""].head(1).iloc[0]
                if not s.dropna().astype(str).loc[lambda values: values != ""].empty
                else "",
            )

    if "folio_match_score" in cluster_summary.columns:
        agg_spec["folio_match_score"] = ("folio_match_score", "max")

    project_summary = cluster_summary.groupby("project_name", as_index=False).agg(**agg_spec)

    project_summary["avg_billing"] = np.where(
        project_summary["total_matters"] > 0,
        project_summary["total_billing"] / project_summary["total_matters"],
        0,
    )

    project_summary["flat_fee_sort"] = project_summary["project_name"].map(is_flat_fee_project)
    project_summary["others_sort"] = project_summary["project_name"].map(is_others_project)
    return project_summary.sort_values(
        ["flat_fee_sort", "others_sort", "total_matters", "total_billing"],
        ascending=[True, True, False, False],
    ).drop(columns=["flat_fee_sort", "others_sort"])


def build_project_ring_chart(project_summary: pd.DataFrame, color_map: dict[str, str]) -> go.Figure:
    top_three = project_summary.head(3).copy()
    other_matters = project_summary.iloc[3:]["total_matters"].sum()
    total_matters = project_summary["total_matters"].sum()

    ring_df = top_three[["project_name", "total_matters"]].copy()
    if other_matters > 0:
        ring_df = pd.concat(
            [ring_df, pd.DataFrame([{"project_name": "Other", "total_matters": other_matters}])],
            ignore_index=True,
        )

    colors = [color_map.get(name, ALTFEE_GRAY) for name in ring_df["project_name"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=ring_df["project_name"],
                values=ring_df["total_matters"],
                hole=0.66,
                sort=False,
                marker=dict(colors=colors, line=dict(color=ALTFEE_BG, width=3)),
                textinfo="label+percent",
                textposition="outside",
                hovertemplate="%{label}<br>Matters: %{value:,.0f}<br>%{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Project Mix",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor=ALTFEE_BG,
        plot_bgcolor=ALTFEE_BG,
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{total_matters:,.0f}</b><br><span style='font-size:13px;color:{ALTFEE_TEXT_SOFT}'>matters</span>",
                x=0.5,
                y=0.5,
                font=dict(size=24, color=ALTFEE_TEXT),
                showarrow=False,
            )
        ],
    )
    return fig


def text_contains_phrase(value: object, phrase: object) -> bool:
    clean_value = clean_text(value)
    clean_phrase = clean_text(phrase)
    if not clean_value or not clean_phrase:
        return False
    return f" {clean_phrase} " in f" {clean_value} "


def build_project_purity_ring(match_pct: float, term: str, project_name: str) -> go.Figure:
    safe_pct = max(0.0, min(100.0, float(match_pct)))
    fig = go.Figure(
        data=[
            go.Pie(
                values=[safe_pct, 100 - safe_pct],
                hole=0.68,
                sort=False,
                direction="clockwise",
                marker=dict(colors=[ALTFEE_TEAL, "#E6E6E6"], line=dict(color=ALTFEE_BG, width=3)),
                textinfo="none",
                hoverinfo="skip",
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"Term Presence in {project_name}", x=0.5, xanchor="center", font=dict(size=18)),
        height=360,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor=ALTFEE_BG,
        plot_bgcolor=ALTFEE_BG,
        annotations=[
            dict(
                text=f"<b>{safe_pct:.0f}%</b>",
                x=0.5,
                y=0.5,
                font=dict(size=32, color=ALTFEE_TEXT),
                showarrow=False,
            )
        ],
    )
    return fig


def project_name_collision_stats(project_df: pd.DataFrame, project_name: str, all_project_names: list[str], matter_name_col: str) -> dict[str, object]:
    if project_df.empty or matter_name_col not in project_df.columns:
        return {
            "collision_pct": 0.0,
            "cooccurrence_pct": 0.0,
            "impurity_pct": 0.0,
            "top_colliding_projects": [],
        }

    count_col = "matter_id_for_count" if "matter_id_for_count" in project_df.columns else matter_name_col
    total_matters = max(project_df[count_col].nunique(), 1)
    own_project_present = project_df[matter_name_col].apply(lambda value: text_contains_phrase(value, project_name))

    other_projects = [
        name for name in all_project_names
        if name != project_name and normalize_legal_text_for_clustering(name) and not is_flat_fee_project(name)
    ]
    collision_project_counts = Counter()
    collision_matter_ids = set()
    cooccurrence_matter_ids = set()
    impurity_matter_ids = set()

    for row_position, (_, row) in enumerate(project_df.iterrows()):
        matter_name = row.get(matter_name_col, "")
        matter_id = row.get(count_col, row_position)
        matched_other_projects = [
            other_project for other_project in other_projects
            if text_contains_phrase(matter_name, other_project)
        ]
        if not matched_other_projects:
            continue
        collision_matter_ids.add(matter_id)
        for other_project in matched_other_projects:
            collision_project_counts[other_project] += 1
        if bool(own_project_present.iloc[row_position]):
            cooccurrence_matter_ids.add(matter_id)
        else:
            impurity_matter_ids.add(matter_id)

    return {
        "collision_pct": len(collision_matter_ids) / total_matters * 100,
        "cooccurrence_pct": len(cooccurrence_matter_ids) / total_matters * 100,
        "impurity_pct": len(impurity_matter_ids) / total_matters * 100,
        "top_colliding_projects": collision_project_counts.most_common(5),
    }


def render_project_ranking(project_summary: pd.DataFrame, color_map: dict[str, str] | None = None) -> None:
    top_five = project_summary.head(5).copy()
    rows = []
    for index, row in top_five.reset_index(drop=True).iterrows():
        rank = index + 1
        number_class = ""
        if rank <= 3:
            number_class = f" ranking-number-{rank}"
        folio_marker = " <span class='folio-plus'>+</span>" if bool(row.get("folio_verified", False)) else ""
        
        rows.append(
            f"<div class='ranking-row'>"
            f"<span class='ranking-name'><span class='{number_class}'>{rank}.</span> {row['project_name']}{folio_marker}</span>"
            f"<span class='ranking-value'>{row['total_matters']:,.0f}</span>"
            f"</div>"
        )

    html = (
        "<div class='ranking-card'>"
        "<div class='ranking-title'>Top Projects</div>"
        f"{''.join(rows)}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_revenue_ranking(project_summary: pd.DataFrame, color_map: dict[str, str] | None = None) -> None:
    revenue_df = project_summary.copy()
    revenue_df["flat_fee_sort"] = revenue_df["project_name"].map(is_flat_fee_project)
    revenue_df["others_sort"] = revenue_df["project_name"].map(is_others_project)
    revenue_df = revenue_df.sort_values(
        ["flat_fee_sort", "others_sort", "total_billing"],
        ascending=[True, True, False],
    ).drop(columns=["flat_fee_sort", "others_sort"]).reset_index(drop=True)
    top_five = revenue_df.head(5).copy()
    top_total_project = revenue_df.iloc[0]["project_name"]
    avg_df = project_summary.copy()
    avg_df["flat_fee_sort"] = avg_df["project_name"].map(is_flat_fee_project)
    avg_df["others_sort"] = avg_df["project_name"].map(is_others_project)
    top_avg_project = avg_df.sort_values(
        ["flat_fee_sort", "others_sort", "avg_billing"],
        ascending=[True, True, False],
    ).iloc[0]["project_name"]

    rows = []
    # Display top 5 by total revenue
    for index, row in top_five.iterrows():
        rank = index + 1
        name_class = ""
        value_class = ""
        suffix = ""
        number_class = ""
        folio_marker = " <span class='folio-plus'>+</span>" if bool(row.get("folio_verified", False)) else ""

        if row["project_name"] == top_total_project:
            name_class = "ranking-bold-black"
            value_class = "ranking-bold-black"

        if row["project_name"] == top_avg_project:
            suffix = " <span class='ranking-bold-red'>*</span>"
            if row["project_name"] != top_total_project:
                name_class = "ranking-bold-red"
                value_class = "ranking-bold-red"

        rows.append(
            f"<div class='ranking-row'>"
            f"<span class='ranking-name {name_class}'>{rank}. {row['project_name']}{suffix}{folio_marker}</span>"
            f"<span class='ranking-value {value_class}'>{format_currency(row['total_billing'])}</span>"
            f"</div>"
        )

    # If highest revenue per project is not in top 5, show it as additional row with actual rank
    if top_avg_project not in top_five["project_name"].values:
        actual_rank = revenue_df[revenue_df["project_name"] == top_avg_project].index[0] + 1
        top_avg_row = revenue_df[revenue_df["project_name"] == top_avg_project].iloc[0]
        rows.append(
            f"<div class='ranking-row'>"
            f"<span class='ranking-name ranking-bold-red'>{actual_rank}. {top_avg_project} <span class='ranking-bold-red'>*</span></span>"
            f"<span class='ranking-value ranking-bold-red'>{format_currency(top_avg_row['total_billing'])}</span>"
            f"</div>"
        )

    html = (
        "<div class='ranking-card'>"
        "<div class='ranking-title'>Total Revenue</div>"
        f"{''.join(rows)}"
        "<div style='padding-top: 0.8rem;'></div>"
        "<div class='ranking-footnote'><span class='ranking-bold-red'>*</span> Project type with higher revenue per project.</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def get_period_column(df: pd.DataFrame, date_col: str, granularity: str) -> pd.DataFrame:
    working = df.copy()
    working["period_date"] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.dropna(subset=["period_date"])
    if granularity == "Yearly":
        working["period"] = working["period_date"].dt.to_period("Y").dt.to_timestamp()
    else:
        working["period"] = working["period_date"].dt.to_period("M").dt.to_timestamp()
    return working


def build_project_timeline_chart(
    selected_df: pd.DataFrame,
    date_col: str,
    billing_col: str,
    matter_col: str,
    granularity: str,
    project_color: str,
    subcluster_df: pd.DataFrame | None = None,
    selected_subcluster_id: int | None = None,
) -> go.Figure:
    working = get_period_column(selected_df, date_col, granularity)
    tick_format = "%Y" if granularity == "Yearly" else "%b %Y"

    if selected_subcluster_id is not None and subcluster_df is not None and not subcluster_df.empty:
        working = subcluster_df[subcluster_df["subcluster_id"] == selected_subcluster_id].copy()
        working = get_period_column(working, date_col, granularity)

    total_billing = (
        working.groupby("period")
        .agg(total_billing=(billing_col, "sum"))
        .reset_index()
        .sort_values("period")
    )

    fig = go.Figure()

    if subcluster_df is not None and not subcluster_df.empty and selected_subcluster_id is None:
        sub_working = get_period_column(subcluster_df, date_col, granularity)
        sub_working["subproject_clean"] = sub_working["subcluster_name"].map(clean_subproject_display_name)
        sub_counts = (
            sub_working.groupby(["period", "subproject_clean"])
            .agg(n_matters=(matter_col, "nunique"))
            .reset_index()
        )
        sub_names = sub_counts.groupby("subproject_clean")["n_matters"].sum().sort_values(ascending=False).index.tolist()
        tones = get_subproject_tones(project_color, len(sub_names))
        for sub_name, color in zip(sub_names, tones):
            plot_df = sub_counts[sub_counts["subproject_clean"] == sub_name]
            fig.add_trace(
                go.Bar(
                    x=plot_df["period"],
                    y=plot_df["n_matters"],
                    name=sub_name,
                    marker_color=color,
                    hovertemplate="%{x|" + tick_format + "}<br>Sub-project: " + sub_name + "<br>Matters: %{y:,.0f}<extra></extra>",
                )
            )
    else:
        matter_counts = (
            working.groupby("period")
            .agg(n_matters=(matter_col, "nunique"))
            .reset_index()
            .sort_values("period")
        )
        fig.add_trace(
            go.Bar(
                x=matter_counts["period"],
                y=matter_counts["n_matters"],
                name="Matters",
                marker_color=hex_to_rgba(project_color, 0.45),
                marker_line=dict(color=project_color, width=1),
                hovertemplate="%{x|" + tick_format + "}<br>Project: Matters<br>Matters: %{y:,.0f}<extra></extra>",
            )
        )

    max_matters = 1
    for trace in fig.data:
        if hasattr(trace, "y") and len(trace.y) > 0:
            max_matters = max(max_matters, np.nanmax(trace.y))

    max_billing = total_billing["total_billing"].max() if not total_billing.empty else 0
    scale_factor = max_billing / max_matters if max_matters and max_billing else 1

    fig.add_trace(
        go.Scatter(
            x=total_billing["period"],
            y=total_billing["total_billing"] / scale_factor,
            mode="lines+markers",
            name="Revenue",
            line=dict(color=ALTFEE_RED, width=2.6),
            marker=dict(color=ALTFEE_RED, size=6),
            yaxis="y1",
            hovertemplate="%{x|" + tick_format + "}<br>Revenue: $%{customdata:,.0f}<extra></extra>",
            customdata=total_billing["total_billing"],
        )
    )

    fig.update_layout(
        title="Matters Over Time",
        barmode="stack",
        height=460,
        paper_bgcolor=ALTFEE_BG,
        plot_bgcolor=ALTFEE_BG,
        margin=dict(l=40, r=55, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="", showgrid=False, tickformat=tick_format),
        yaxis=dict(title="Matters", gridcolor=ALTFEE_GRID, zeroline=False),
        yaxis2=dict(
            title="Revenue",
            overlaying="y",
            side="right",
            tickprefix="$",
            showgrid=False,
            zeroline=False,
        ),
    )
    return fig


def build_hours_distribution_chart(cluster_df: pd.DataFrame, project_color: str) -> go.Figure:
    clean = cluster_df.dropna(subset=["numeric_hours"]).copy()
    avg_hours = clean["numeric_hours"].mean() if not clean.empty else 0
    fig = px.histogram(clean, x="numeric_hours", nbins=24, title="Hours Distribution")
    fig.update_traces(marker_color=hex_to_rgba(project_color, 0.32), marker_line_color=project_color, marker_line_width=1)
    fig.add_vline(
        x=avg_hours,
        line_width=2.8,
        line_dash="dot",
        line_color=ALTFEE_RED,
        annotation_text="Avg",
        annotation_position="top right",
    )
    fig.add_annotation(
        x=avg_hours,
        y=1,
        yref="paper",
        text=f"Average: {avg_hours:,.1f}h",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color=ALTFEE_RED, size=12),
        bgcolor="rgba(255,255,255,0.85)",
    )
    fig.update_layout(height=390, xaxis_title="Hours", yaxis_title="Matters", paper_bgcolor=ALTFEE_BG, plot_bgcolor=ALTFEE_BG)
    fig.update_yaxes(gridcolor=ALTFEE_GRID)
    return fig



def build_effort_trend_chart(cluster_df: pd.DataFrame, date_col: str, granularity: str, project_color: str) -> go.Figure:
    clean = get_period_column(cluster_df.dropna(subset=["numeric_hours"]), date_col, granularity)
    tick_format = "%Y" if granularity == "Yearly" else "%b %Y"

    if clean.empty:
        return go.Figure()

    summary = (
        clean.groupby("period")["numeric_hours"]
        .quantile([0.1, 0.25, 0.75, 0.9])
        .unstack()
        .reset_index()
        .rename(columns={0.1: "p10", 0.25: "p25", 0.75: "p75", 0.9: "p90"})
        .sort_values("period")
    )

    means = clean.groupby("period", as_index=False)["numeric_hours"].mean().rename(columns={"numeric_hours": "mean"})
    summary = summary.merge(means, on="period", how="left")
    summary["middle_height"] = summary["p75"] - summary["p25"]
    summary["period_number"] = np.arange(len(summary))

    fig = go.Figure()

    for _, row in summary.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["period"], row["period"]],
                y=[row["p10"], row["p90"]],
                mode="lines",
                line=dict(color=hex_to_rgba(project_color, 0.45), width=2.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Bar(
            x=summary["period"],
            y=summary["middle_height"],
            base=summary["p25"],
            name="Middle 50%",
            marker=dict(
                color=hex_to_rgba(project_color, 0.24),
                line=dict(color=project_color, width=1.4),
            ),
            width=0.85 if granularity == "Yearly" else None,
            hovertemplate=(
                "%{x|" + tick_format + "}"
                "<br>P25: %{base:,.1f}"
                "<br>P75: %{customdata:,.1f}"
                "<extra></extra>"
            ),
            customdata=summary["p75"],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=summary["period"],
            y=summary["mean"],
            mode="markers",
            name="Mean",
            marker=dict(color=project_color, size=10, line=dict(color=ALTFEE_BG, width=1.6)),
            hovertemplate="%{x|" + tick_format + "}<br>Mean: %{y:,.1f}<extra></extra>",
        )
    )

    if len(summary) >= 2:
        degree = 1 if len(summary) < 4 else 2
        coefficients = np.polyfit(summary["period_number"], summary["mean"], degree)
        trend_values = np.polyval(coefficients, summary["period_number"])
        fig.add_trace(
            go.Scatter(
                x=summary["period"],
                y=trend_values,
                mode="lines",
                name="Trend",
                line=dict(color=ALTFEE_RED, width=2.2, dash="dot"),
                hovertemplate="%{x|" + tick_format + "}<br>Trend: %{y:,.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Effort Over Time",
        height=390,
        xaxis_title="",
        yaxis_title="Hours",
        barmode="overlay",
        paper_bgcolor=ALTFEE_BG,
        plot_bgcolor=ALTFEE_BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(tickformat=tick_format, showgrid=False)
    fig.update_yaxes(gridcolor=ALTFEE_GRID, zeroline=False)
    return fig


# --- Helper functions for project option labels ---
FOLIO_OPTION_SUFFIX = " +"
SUBPROJECT_OPTION_SUFFIX = " *"


def make_project_option_label(project_name: str, has_subprojects: bool, folio_verified: bool = False) -> str:
    label = str(project_name)
    if folio_verified:
        label = f"{label}{FOLIO_OPTION_SUFFIX}"
    if has_subprojects:
        label = f"{label}{SUBPROJECT_OPTION_SUFFIX}"
    return label


def strip_project_option_label(option_label: str) -> str:
    return option_label.replace(SUBPROJECT_OPTION_SUFFIX, "").replace(FOLIO_OPTION_SUFFIX, "")


def render_project_report_heading(project_name: str, project_row: pd.Series | None, folio_mapper_active: bool) -> None:
    folio_title = ""
    folio_description = ""
    folio_api_status = ""
    folio_verified = False
    if project_row is not None:
        folio_title = str(project_row.get("folio_title", "") or "").strip()
        folio_description = str(project_row.get("folio_description", "") or "").strip()
        folio_api_status = str(project_row.get("folio_api_status", "") or "").strip()
        folio_verified = bool(project_row.get("folio_verified", False))

    title_html = escape(folio_title or project_name)
    description_html = ""
    if folio_mapper_active:
        if folio_verified:
            if folio_description:
                description_html = escape(folio_description)
            else:
                description_html = "No description available"
        elif folio_api_status == "connection_failed":
            description_html = "API connection failed"

    badge_html = '<span class="folio-plus">+</span>' if folio_verified else ""
    meta_html = (
        '<div class="folio-project-meta">+ FOLIO verified</div>'
        if folio_verified
        else ""
    )
    description_block = (
        f'<div class="folio-project-description">{description_html}</div>'
        if description_html
        else ""
    )

    st.markdown(
        f"""
        <div class="folio-project-header">
            <div class="folio-project-heading">
                <h3>{title_html}</h3>
                {badge_html}
            </div>
            {description_block}
            {meta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )



st.markdown(
    """
    <div class="hero-card">
        <h1>Historical Matter Dashboard</h1>
        <p>Upload one account-level historical_matter_evidence CSV and get an initial billing overview before categorization begins.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

sample_paths = get_sample_account_paths()
sample_options = {format_sample_account_label(path): str(path) for path in sample_paths}

input_cols = st.columns((1.25, 1))

with input_cols[0]:
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Expected structure: one row per matter, similar to historical_matter_evidence, filtered to one account.",
    )

with input_cols[1]:
    selected_sample_label = st.selectbox(
        "Or use a sample account",
        options=["None"] + list(sample_options.keys()),
        index=0,
        help="Bundled sample accounts are loaded from data/random_account_exports/.",
    )

if uploaded_file is None and selected_sample_label == "None":
    st.markdown(
        """
        <div class="section-card">
            <h3>Expected input</h3>
            <p class="small-muted">
                Upload a CSV with one account only, or choose one of the bundled sample accounts.
                The app will infer the account, matter, date, and billing columns,
                then show a quick historical overview before categorization begins.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    if uploaded_file is not None:
        raw_df = load_csv(uploaded_file)
        data_source_label = "Uploaded file"
    else:
        raw_df = load_sample_csv(sample_options[selected_sample_label])
        data_source_label = selected_sample_label
except Exception as exc:
    st.error(f"Could not read the CSV file. Error: {exc}")
    st.stop()

if raw_df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

matter_df = normalize_columns(raw_df)
columns = infer_columns(matter_df)

missing_required = [logical_name for logical_name in REQUIRED_COLUMN_OPTIONS if columns[logical_name] is None]
if missing_required:
    render_missing_columns(missing_required)

    with st.expander("Detected columns in uploaded file"):
        st.write(list(matter_df.columns))
    st.stop()

account_col = columns["account_name"]
matter_col = columns["matter_id"]
date_col = columns["date"]
billing_col = columns["billing"]
matter_name_col = columns["matter_name"]
hours_col = columns["total_hours"]
rate_col = columns["avg_rate"]
entries_col = columns["n_time_entries"]
users_col = columns["n_unique_users"]
practice_area_col = columns["practice_area"]
matter_category_col = columns["matter_category"]
time_entry_text_col = columns["all_time_entry_text"]

matter_df[billing_col] = pd.to_numeric(matter_df[billing_col], errors="coerce").fillna(0)
matter_df["parsed_date"] = parse_dates(matter_df, date_col)
matter_df["matter_id_for_count"] = matter_df[matter_col]

account_names = matter_df[account_col].dropna().astype(str).unique()
account_name = account_names[0] if len(account_names) > 0 else "Unknown Account"

if len(account_names) > 1:
    st.warning(
        f"This file contains {len(account_names)} accounts. The dashboard is designed for one account. "
        f"Showing combined results, with primary account displayed as: {account_name}."
    )

number_of_matters = matter_df[matter_col].nunique()

analysis_data_warning = number_of_matters < 120


def render_low_data_warning() -> None:
    if analysis_data_warning:
        st.warning(
            f"Not enough data for strong analysis: this account has {number_of_matters:,.0f} matters. "
            "Use the clustering output as directional, not definitive. A stronger analysis usually needs at least 120 matters."
        )
        
total_billing = matter_df[billing_col].sum()
average_billing = matter_df[billing_col].mean()

valid_dates = matter_df["parsed_date"].dropna()
if valid_dates.empty:
    date_range_label = "No valid dates"
else:
    date_range_label = f"{valid_dates.min().date()} → {valid_dates.max().date()}"

default_granularity = choose_default_granularity(matter_df["parsed_date"])

overview_tab, clustering_tab, matter_search_tab, project_purity_tab = st.tabs([
    "Initial Overview",
    "Cluster Analysis",
    "Matter Search",
    "Project Purity",
])

with overview_tab:
    st.markdown(f"### {account_name}")
    st.caption(f"Date range: {date_range_label}")
    st.caption(f"Data source: {data_source_label}")
    st.markdown(
        """
        <div class="section-card">
            <h3 style="margin-bottom: 0.25rem;">Initial Historical Billing Overview</h3>
            <p class="small-muted" style="margin-bottom: 0;">
                This view summarizes the account before project categorization starts.
                It focuses only on matter volume and billing trends.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Account", account_name, "Primary account detected")
    with metric_cols[1]:
        render_metric_card("Matters", format_number(number_of_matters), "Unique matters")
    with metric_cols[2]:
        render_metric_card("Total billing", format_currency(total_billing), "Sum of matter billings")
    with metric_cols[3]:
        render_metric_card("Avg. billing", format_currency(average_billing), "Average per matter")

    if valid_dates.empty:
        st.error("No valid dates were found, so the trend chart cannot be created.")
        st.stop()

    chart_header_cols = st.columns((3, 1))

    with chart_header_cols[0]:
        st.markdown("### Matter Volume and Billing Over Time")
        st.caption("Toggle between monthly and annual aggregation depending on the time span you want to inspect.")

    with chart_header_cols[1]:
        granularity = st.radio(
            "Time grouping",
            options=["Monthly", "Yearly"],
            index=0 if default_granularity == "Monthly" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="overview_time_grouping_radio",
        )

    if granularity is None:
        granularity = default_granularity

    time_series = build_time_series(matter_df, date_col, billing_col, matter_col, granularity)

    if time_series.empty:
        st.error("No usable rows remained after parsing dates and billing values.")
        st.stop()

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.plotly_chart(build_trend_chart(time_series, granularity), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Preview uploaded data"):
        st.dataframe(matter_df.drop(columns=["parsed_date"], errors="ignore").head(50), use_container_width=True)

with matter_search_tab:
    st.markdown("### Matter Search")
    if "clustered_matters" not in st.session_state:
        st.info("Run the cluster analysis first to search assigned matters.")
    else:
        search_df = st.session_state["clustered_matters"].copy()
        search_query = st.text_input("Search by matter name", key="matter_search_query")
        if search_query.strip():
            result_df = search_df[
                search_df[matter_name_col]
                .fillna("")
                .astype(str)
                .str.contains(re.escape(search_query.strip()), case=False, na=False)
            ].copy()
        else:
            result_df = search_df.head(50).copy()

        if result_df.empty:
            st.info("No matters matched that search.")
        else:
            result_df["_matter_search_label"] = result_df.apply(
                lambda row: f"{row.get(matter_name_col, 'Unnamed matter')} | {row.get(matter_col, row.name)}",
                axis=1,
            )
            display_cols = [matter_name_col, "project_name"]
            if hours_col and hours_col in result_df.columns:
                display_cols.append(hours_col)
            elif "numeric_hours" in result_df.columns:
                display_cols.append("numeric_hours")
            if billing_col and billing_col in result_df.columns:
                display_cols.append(billing_col)
            elif "numeric_billing" in result_df.columns:
                display_cols.append("numeric_billing")

            st.dataframe(result_df[display_cols].head(100), use_container_width=True, hide_index=True)
            selected_matter_label = st.selectbox(
                "Select a matter",
                options=result_df["_matter_search_label"].head(100).tolist(),
                key="matter_search_selected_matter",
            )

            selected_row = result_df[result_df["_matter_search_label"] == selected_matter_label].iloc[0]
            assigned_project = selected_row.get("project_name", "Unclear / Needs Review")
            project_df = search_df[search_df["project_name"] == assigned_project].copy()
            selected_hours = pd.to_numeric(selected_row.get("numeric_hours", selected_row.get(hours_col, np.nan)), errors="coerce")
            selected_billing = pd.to_numeric(selected_row.get("numeric_billing", selected_row.get(billing_col, np.nan)), errors="coerce")
            project_hours = pd.to_numeric(project_df.get("numeric_hours", pd.Series(dtype=float)), errors="coerce").dropna()
            project_billing = pd.to_numeric(project_df.get("numeric_billing", pd.Series(dtype=float)), errors="coerce").dropna()
            project_count = project_df["matter_id_for_count"].nunique() if "matter_id_for_count" in project_df.columns else len(project_df)

            def percentile_label(value: float, series: pd.Series) -> str:
                if pd.isna(value) or series.empty:
                    return "Not available"
                return f"{(series.le(value).mean() * 100):.0f}th percentile"

            detail_cols = st.columns(3)
            with detail_cols[0]:
                render_metric_card("Assigned Project", str(assigned_project), "Current cluster label")
            with detail_cols[1]:
                render_metric_card(
                    "Total Billing",
                    format_currency(selected_billing) if not pd.isna(selected_billing) else "Billing not available",
                    percentile_label(selected_billing, project_billing),
                )
            with detail_cols[2]:
                render_metric_card(
                    "Total Hours",
                    format_number(selected_hours) if not pd.isna(selected_hours) else "Hours not available",
                    percentile_label(selected_hours, project_hours),
                )

            project_cols = st.columns(3)
            with project_cols[0]:
                render_metric_card("Project Matters", format_number(project_count), "Matter count in assigned project")
            with project_cols[1]:
                render_metric_card(
                    "Project Avg. Hours",
                    format_number(project_hours.mean()) if not project_hours.empty else "Hours not available",
                    "Average within assigned project",
                )
            with project_cols[2]:
                render_metric_card(
                    "Project Avg. Billing",
                    format_currency(project_billing.mean()) if not project_billing.empty else "Billing not available",
                    "Average within assigned project",
                )

with project_purity_tab:
    st.markdown("### Project Purity")
    if "clustered_matters" not in st.session_state:
        st.info("Run the cluster analysis first to inspect project purity.")
    else:
        purity_df = st.session_state["clustered_matters"].copy()
        project_names = (
            purity_df["project_name"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if not project_names:
            st.info("No project clusters are available yet.")
        else:
            purity_controls = st.columns((1.2, 1.4))
            with purity_controls[0]:
                selected_purity_project = st.selectbox(
                    "Project cluster",
                    options=project_names,
                    key="project_purity_project_selector",
                )
            with purity_controls[1]:
                purity_term = st.text_input(
                    "Term to test",
                    value=selected_purity_project,
                    key="project_purity_term_input",
                )

            selected_purity_df = purity_df[purity_df["project_name"] == selected_purity_project].copy()
            count_col = "matter_id_for_count" if "matter_id_for_count" in selected_purity_df.columns else matter_name_col
            total_project_matters = max(selected_purity_df[count_col].nunique(), 1)
            term_match_matter_ids = set(
                selected_purity_df.loc[
                    selected_purity_df[matter_name_col].apply(lambda value: text_contains_phrase(value, purity_term)),
                    count_col,
                ].tolist()
            )
            term_match_pct = len(term_match_matter_ids) / total_project_matters * 100

            top_terms = get_top_terms_from_text(selected_purity_df[matter_name_col], ngram_n=1, top_n=5)
            top_bigrams = get_top_terms_from_text(selected_purity_df[matter_name_col], ngram_n=2, top_n=5)
            collision_stats = project_name_collision_stats(
                selected_purity_df,
                selected_purity_project,
                project_names,
                matter_name_col,
            )

            purity_layout = st.columns((1, 1.15))
            with purity_layout[0]:
                st.plotly_chart(
                    build_project_purity_ring(term_match_pct, purity_term, selected_purity_project),
                    use_container_width=True,
                )
            with purity_layout[1]:
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    render_metric_card("Matters", format_number(total_project_matters), "Selected project")
                with metric_cols[1]:
                    render_metric_card("Term Match", f"{term_match_pct:.0f}%", "Matter names containing term")
                with metric_cols[2]:
                    render_metric_card("Project Collision", f"{collision_stats['collision_pct']:.0f}%", "Matter names containing another project")

                detail_cols = st.columns(2)
                with detail_cols[0]:
                    st.markdown("#### Top Terms")
                    st.write(", ".join(top_terms) if top_terms else "No terms found.")
                    st.markdown("#### Top Bigrams")
                    st.write(", ".join(top_bigrams) if top_bigrams else "No bigrams found.")
                with detail_cols[1]:
                    st.markdown("#### Collision Detail")
                    st.write(f"Co-occurrence: {collision_stats['cooccurrence_pct']:.0f}%")
                    st.write(f"Impurity: {collision_stats['impurity_pct']:.0f}%")
                    if collision_stats["top_colliding_projects"]:
                        collision_df = pd.DataFrame(
                            collision_stats["top_colliding_projects"],
                            columns=["other_project", "matter_count"],
                        )
                        st.dataframe(collision_df, use_container_width=True, hide_index=True)
                    else:
                        st.write("No other project names found in this project's matter names.")

with clustering_tab:
    st.markdown("### Cluster Analysis")
    st.caption("Run a quick matter-name clustering pass, then inspect the main clusters and optional subclusters.")
    render_low_data_warning()

    folio_practice_areas = load_folio_practice_areas()
    firm_taxonomy_labels, firm_practice_area_mapping = build_firm_specific_taxonomy_labels(
        matter_df=matter_df,
        practice_area_col=practice_area_col,
        folio_labels=folio_practice_areas,
    )
    taxonomy_candidate_source_map = build_taxonomy_candidate_source_map(
        firm_taxonomy_labels,
        firm_practice_area_mapping,
    )

    suspicious_taxonomy_terms = ["general", "issues", "misc", "other", "unknown", "case", "matter"]
    suspicious_firm_labels = []
    if not firm_practice_area_mapping.empty:
        retained_firm_rows = firm_practice_area_mapping[firm_practice_area_mapping["action"] == "kept_firm_specific"]
        suspicious_firm_labels = (
            retained_firm_rows[
                retained_firm_rows["final_taxonomy_label"]
                .astype(str)
                .str.lower()
                .apply(lambda value: any(term in value for term in suspicious_taxonomy_terms))
            ]["final_taxonomy_label"]
            .drop_duplicates()
            .tolist()
        )

    with st.expander("Taxonomy label source debug"):
        st.write(f"Detected practice_area column: {practice_area_col or 'None'}")
        if practice_area_col and practice_area_col in matter_df.columns:
            st.caption("Unique raw practice_area values from uploaded CSV")
            st.write(matter_df[practice_area_col].drop_duplicates().tolist())
        else:
            st.caption("No practice_area column was detected in the uploaded CSV.")
        st.caption("Practice-area source mapping")
        if firm_practice_area_mapping.empty:
            st.write("No uploaded practice_area values were available for mapping.")
        else:
            st.dataframe(firm_practice_area_mapping, use_container_width=True, hide_index=True)
        if suspicious_firm_labels:
            st.warning("Suspicious firm-specific taxonomy labels remain: " + ", ".join(suspicious_firm_labels))
        st.caption(f"Final valid taxonomy labels used for classification: {len(firm_taxonomy_labels):,}")
        st.write(", ".join(firm_taxonomy_labels) if firm_taxonomy_labels else "None")

    control_cols = st.columns((1.05, 1.05, 1.05, 0.85, 1))
    with control_cols[0]:
        use_ai = st.toggle(
            "Use AI labels with Ollama",
            value=False,
            help="When enabled, the app asks local Ollama to name each cluster. If Ollama fails, it falls back to TF-IDF terms.",
        )
    with control_cols[1]:
        classify_taxonomy_levels = st.toggle(
            "Classify taxonomy level",
            value=bool(firm_taxonomy_labels),
            disabled=not bool(firm_taxonomy_labels),
            help="Compares each project cluster against FOLIO practice areas plus this firm's non-colliding practice_area values.",
        )
    with control_cols[2]:
        enrich_with_folio = st.toggle(
            "Enrich with FOLIO",
            value=True,
            help="Calls the public FOLIO API with each project name and displays verified ontology context when a match is found.",
        )
    with control_cols[3]:
        run_clustering = st.button("Run cluster analysis", type="primary")
    with control_cols[4]:
        if firm_taxonomy_labels:
            n_firm_specific = int((firm_practice_area_mapping["action"] == "kept_firm_specific").sum()) if not firm_practice_area_mapping.empty else 0
            st.caption(f"Using {len(firm_taxonomy_labels):,.0f} taxonomy labels; {n_firm_specific:,.0f} firm-specific.")
        else:
            st.caption("No taxonomy labels available.")

    if use_ai:
        st.info("AI labeling requires Ollama running locally at http://localhost:11434 and the llama3.1:8b model pulled.")

    if run_clustering:
        st.session_state.pop("cluster_summary", None)
        st.session_state.pop("clustered_matters", None)
        st.session_state.pop("subclusters_by_cluster", None)
        st.session_state.pop("firm_practice_area_mapping", None)
        st.session_state.pop("firm_taxonomy_labels", None)
        st.session_state.pop("taxonomy_candidate_source_map", None)
        st.session_state.pop("enrich_with_folio", None)
        st.session_state.pop("folio_enrichment_version", None)

        progress_bar = st.progress(0)
        status_box = st.empty()

        with st.spinner("Creating clusters. This may take a bit if AI labeling is enabled..."):
            cluster_summary, clustered_matters = create_project_clusters(
                matter_df,
                matter_name_col=matter_name_col,
                billing_col=billing_col,
                use_ai=use_ai,
                progress_bar=progress_bar,
                status_box=status_box,
                text_col=time_entry_text_col,
                practice_area_col=practice_area_col,
                matter_category_col=matter_category_col,
            )

        if cluster_summary.empty:
            st.error("No clusters could be created. Check that the file has usable matter names.")
        else:
            if classify_taxonomy_levels:
                with st.spinner("Classifying clusters into taxonomy levels..."):
                    cluster_summary = classify_clusters_to_taxonomy_levels(
                        cluster_summary,
                        firm_taxonomy_labels,
                        taxonomy_candidate_source_map,
                    )
                    cluster_summary = validate_taxonomy_predictions(cluster_summary, firm_taxonomy_labels)

            if enrich_with_folio:
                with st.spinner("Checking project names against FOLIO..."):
                    cluster_summary = enrich_clusters_with_folio(cluster_summary)

            clustered_matters = add_numeric_columns(clustered_matters, billing_col, hours_col, rate_col, entries_col, users_col)

            if "predicted_taxonomy_level" in cluster_summary.columns:
                taxonomy_lookup = cluster_summary.set_index("project_name")["predicted_taxonomy_level"].to_dict()
                taxonomy_score_lookup = cluster_summary.set_index("project_name")["taxonomy_level_score"].to_dict()
                taxonomy_method_lookup = cluster_summary.set_index("project_name")["taxonomy_classification_method"].to_dict()
                clustered_matters["predicted_taxonomy_level"] = (
                    clustered_matters["project_name"].map(taxonomy_lookup).fillna("Unclassified")
                )
                clustered_matters["taxonomy_level_score"] = clustered_matters["project_name"].map(taxonomy_score_lookup)
                clustered_matters["taxonomy_classification_method"] = clustered_matters["project_name"].map(taxonomy_method_lookup)

            st.session_state["cluster_summary"] = cluster_summary
            st.session_state["clustered_matters"] = clustered_matters
            st.session_state["use_ai"] = use_ai
            st.session_state["classify_taxonomy_levels"] = classify_taxonomy_levels
            st.session_state["firm_practice_area_mapping"] = firm_practice_area_mapping
            st.session_state["firm_taxonomy_labels"] = firm_taxonomy_labels
            st.session_state["taxonomy_candidate_source_map"] = taxonomy_candidate_source_map
            st.session_state["enrich_with_folio"] = enrich_with_folio
            st.session_state["folio_enrichment_version"] = FOLIO_CACHE_VERSION if enrich_with_folio else None

    if "cluster_summary" not in st.session_state or "clustered_matters" not in st.session_state:
        st.info("Run the cluster analysis to see project names, charts, and cluster details.")
        st.stop()

    cluster_summary = st.session_state["cluster_summary"]
    clustered_matters = st.session_state["clustered_matters"]

    if (
        st.session_state.get("enrich_with_folio", False)
        and st.session_state.get("folio_enrichment_version") != FOLIO_CACHE_VERSION
    ):
        with st.spinner("Refreshing FOLIO enrichment..."):
            cluster_summary = enrich_clusters_with_folio(cluster_summary.drop(columns=[
                col for col in cluster_summary.columns if col.startswith("folio_")
            ], errors="ignore"))
        st.session_state["cluster_summary"] = cluster_summary
        st.session_state["folio_enrichment_version"] = FOLIO_CACHE_VERSION

    if "predicted_taxonomy_level" in cluster_summary.columns:
        cluster_summary = validate_taxonomy_predictions(cluster_summary, firm_taxonomy_labels)
        taxonomy_lookup = cluster_summary.set_index("project_name")["predicted_taxonomy_level"].to_dict()
        taxonomy_score_lookup = cluster_summary.set_index("project_name")["taxonomy_level_score"].to_dict()
        taxonomy_method_lookup = cluster_summary.set_index("project_name")["taxonomy_classification_method"].to_dict()
        clustered_matters["predicted_taxonomy_level"] = (
            clustered_matters["project_name"].map(taxonomy_lookup).fillna("Unclassified")
        )
        clustered_matters["taxonomy_level_score"] = clustered_matters["project_name"].map(taxonomy_score_lookup)
        clustered_matters["taxonomy_classification_method"] = clustered_matters["project_name"].map(taxonomy_method_lookup)
        st.session_state["cluster_summary"] = cluster_summary
        st.session_state["clustered_matters"] = clustered_matters

    actual_clustered_count = (
        clustered_matters["matter_id_for_count"].nunique()
        if "matter_id_for_count" in clustered_matters.columns
        else len(clustered_matters)
    )

    # Check actual clustered matter count and show warning if low
    if actual_clustered_count < 120:
        st.warning(
            f"⚠️ Limited clustering data: only {actual_clustered_count:,.0f} matters passed clustering filters (out of {number_of_matters:,.0f} total uploaded). "
            "Use these results as directional guidance only. Strong analysis typically requires at least 120 usable matters."
        )

    project_summary = aggregate_project_categories(cluster_summary)
    top_project_names = project_summary.head(3)["project_name"].tolist()
    project_color_map = get_project_color_map(top_project_names)

    subproject_candidate_projects = set(
        clustered_matters.groupby("project_name")["matter_id_for_count"]
        .nunique()
        .loc[lambda s: s >= 50]
        .index
        .tolist()
    )

    top_layout = st.columns((1.1, 1, 1))
    with top_layout[0]:
        st.plotly_chart(build_project_ring_chart(project_summary, project_color_map), use_container_width=True)
    with top_layout[1]:
        render_project_ranking(project_summary, project_color_map)
    with top_layout[2]:
        render_revenue_ranking(project_summary, project_color_map)

    st.markdown("### Project Detail")

    detail_controls = st.columns((1.1, 1.2, 1.2, 0.8))

    project_options_df = project_summary.copy()
    project_options_df["has_subprojects"] = project_options_df["project_name"].isin(subproject_candidate_projects)
    project_options_df["option_label"] = project_options_df.apply(
        lambda row: make_project_option_label(
            row["project_name"],
            row["has_subprojects"],
            bool(row.get("folio_verified", False)),
        ),
        axis=1,
    )

    if "predicted_taxonomy_level" in project_options_df.columns:
        practice_area_options = ["All Taxonomy Levels"] + (
            project_options_df["predicted_taxonomy_level"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
    else:
        practice_area_options = ["All Taxonomy Levels"]

    with detail_controls[0]:
        selected_taxonomy_level = st.selectbox(
            "Taxonomy Level",
            options=practice_area_options,
            index=0,
            key="taxonomy_level_detail_selector",
        )

    if selected_taxonomy_level != "All Taxonomy Levels" and "predicted_taxonomy_level" in project_options_df.columns:
        project_options_df = project_options_df[
            project_options_df["predicted_taxonomy_level"] == selected_taxonomy_level
        ].copy()

    if project_options_df.empty:
        st.warning("No projects available for the selected taxonomy level.")
        st.stop()

    with detail_controls[1]:
        selected_project_option = st.selectbox(
            "Project",
            options=project_options_df["option_label"].tolist(),
            index=0,
            key="project_detail_selector",
        )

    selected_project = strip_project_option_label(selected_project_option)
    selected_cluster_df = clustered_matters[clustered_matters["project_name"] == selected_project].copy()
    selected_project_color = project_color_map.get(selected_project, ALTFEE_GRAY)
    selected_project_rows = project_summary[project_summary["project_name"] == selected_project]
    selected_project_row = selected_project_rows.iloc[0] if not selected_project_rows.empty else None

    if "predicted_taxonomy_level" in project_summary.columns:
        selected_taxonomy_value = (
            project_summary.loc[
                project_summary["project_name"] == selected_project,
                "predicted_taxonomy_level",
            ]
            .dropna()
            .astype(str)
            .head(1)
        )
        if not selected_taxonomy_value.empty:
            st.caption(f"Taxonomy level: {selected_taxonomy_value.iloc[0]}")

    selected_cluster_size = selected_cluster_df["matter_id_for_count"].nunique()

    if "subclusters_by_cluster" not in st.session_state:
        st.session_state["subclusters_by_cluster"] = {}

    subcluster_key = selected_project
    if selected_cluster_size >= SUBCLUSTER_MIN_COUNT and subcluster_key not in st.session_state["subclusters_by_cluster"]:
        with st.spinner("Creating keyword branches..."):
            subcluster_summary_auto, subclustered_matters_auto = create_keyword_subclusters(
                selected_cluster_df,
                parent_project=selected_project,
                matter_col=matter_col,
                matter_name_col=matter_name_col,
                billing_col=billing_col,
            )
            if not subcluster_summary_auto.empty:
                subclustered_matters_auto = add_numeric_columns(subclustered_matters_auto, billing_col, hours_col, rate_col, entries_col, users_col)
                st.session_state["subclusters_by_cluster"][subcluster_key] = {
                    "summary": subcluster_summary_auto,
                    "matters": subclustered_matters_auto,
                }

    active_subcluster_summary = None
    active_subclustered_matters = None
    selected_subcluster_id = None

    if subcluster_key in st.session_state["subclusters_by_cluster"]:
        active_subcluster_summary = st.session_state["subclusters_by_cluster"][subcluster_key]["summary"]
        active_subclustered_matters = st.session_state["subclusters_by_cluster"][subcluster_key]["matters"]

    if active_subcluster_summary is not None and not active_subcluster_summary.empty:
        active_subcluster_summary = active_subcluster_summary.copy()
        active_subclustered_matters = active_subclustered_matters.copy()
        active_subcluster_summary["subcluster_name"] = active_subcluster_summary["subcluster_name"].map(clean_subproject_display_name)
        active_subcluster_summary["display_label"] = active_subcluster_summary["subcluster_name"]
        active_subcluster_summary = (
            active_subcluster_summary.groupby("subcluster_name", as_index=False)
            .agg(
                display_label=("display_label", "first"),
                matter_count=("matter_count", "sum"),
                total_billing=("total_billing", "sum"),
                avg_billing=("avg_billing", "mean"),
                top_terms=("top_terms", "first"),
                top_bigrams=("top_bigrams", "first"),
                subcluster_id=("subcluster_id", "first"),
            )
            .sort_values(["matter_count", "total_billing"], ascending=[False, False])
        )
        active_subclustered_matters["subcluster_name"] = active_subclustered_matters["subcluster_name"].map(clean_subproject_display_name)

    with detail_controls[2]:
        if active_subcluster_summary is not None and not active_subcluster_summary.empty:
            sub_options = ["All sub-projects"] + active_subcluster_summary["subcluster_name"].drop_duplicates().tolist()
            selected_subproject = st.selectbox(
                "Sub-project",
                options=sub_options,
                key=f"subproject_selector_{clean_column_name(selected_project)}",
            )
            if selected_subproject != "All sub-projects":
                selected_subcluster_row = active_subcluster_summary[active_subcluster_summary["subcluster_name"] == selected_subproject].iloc[0]
                selected_subcluster_id = selected_subcluster_row["subcluster_id"]
                selected_cluster_df = active_subclustered_matters[active_subclustered_matters["subcluster_name"] == selected_subproject].copy()
        else:
            st.selectbox(
                "Sub-project",
                options=["No sub-projects"],
                disabled=True,
                key=f"subproject_selector_disabled_{clean_column_name(selected_project)}",
            )

    with detail_controls[3]:
        detail_granularity = st.radio(
            "Time",
            options=["Monthly", "Yearly"],
            index=0 if default_granularity == "Monthly" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="project_detail_time_grouping_radio",
        )

    render_project_report_heading(
        selected_project,
        selected_project_row,
        bool(st.session_state.get("enrich_with_folio", False)),
    )

    detail = summarize_cluster_detail(selected_cluster_df)

    detail_cols = st.columns(4)
    with detail_cols[0]:
        render_metric_card("Matters", format_number(detail["matters"]), "Selected project")
    with detail_cols[1]:
        render_metric_card("Total Billing", format_currency(detail["total_billing"]), "Project revenue")
    with detail_cols[2]:
        render_metric_card("Avg. Billing", format_currency(detail["avg_billing"]), "Per matter")
    with detail_cols[3]:
        render_metric_card("Effective Rate", format_currency(detail["effective_rate"]), "Billing / hours")

    st.plotly_chart(
        build_project_timeline_chart(
            selected_cluster_df,
            date_col=date_col,
            billing_col=billing_col,
            matter_col=matter_col,
            granularity=detail_granularity,
            project_color=selected_project_color,
            subcluster_df=active_subclustered_matters,
            selected_subcluster_id=selected_subcluster_id,
        ),
        use_container_width=True,
    )

    hours_available = selected_cluster_df["numeric_hours"].notna().any()

    if hours_available:
        metric_cols_2 = st.columns(3)
        with metric_cols_2[0]:
            render_metric_card("Total Hours", format_number(detail["total_hours"]), "Project workload")
        with metric_cols_2[1]:
            render_metric_card("Avg. Hours", format_number(detail["avg_hours"]), "Per matter")
        with metric_cols_2[2]:
            render_variability_card(detail["hours_cv"])

        detail_chart_cols = st.columns(2)
        with detail_chart_cols[0]:
            st.plotly_chart(
                build_hours_distribution_chart(selected_cluster_df, selected_project_color),
                use_container_width=True,
            )
        with detail_chart_cols[1]:
            st.plotly_chart(
                build_effort_trend_chart(selected_cluster_df, date_col, detail_granularity, selected_project_color),
                use_container_width=True,
            )
    else:
        st.info("No total_hours column was detected, so time distribution and effort charts are hidden.")

    with st.expander("Example matters in this project"):
        display_cols = [matter_name_col, billing_col]
        if hours_col:
            display_cols.append(hours_col)
        if practice_area_col:
            display_cols.append(practice_area_col)

        clean_examples = selected_cluster_df[display_cols].copy()
        title_terms = remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(selected_project)))

        if title_terms:
            def example_similarity_score(row):
                text = " ".join(str(row.get(col, "")) for col in display_cols if col in row.index)
                tokens = set(remove_junk_tokens(tokenize_text(normalize_legal_text_for_clustering(text))))
                return len(tokens.intersection(title_terms))

            clean_examples["_title_match_score"] = clean_examples.apply(example_similarity_score, axis=1)
            clean_examples = clean_examples.sort_values(["_title_match_score", billing_col], ascending=[False, False])
            clean_examples = clean_examples.drop(columns=["_title_match_score"], errors="ignore")
        else:
            clean_examples = clean_examples.sort_values(billing_col, ascending=False)

        top_examples = clean_examples.head(12)
        st.dataframe(top_examples, use_container_width=True, hide_index=True)

        if time_entry_text_col and time_entry_text_col in selected_cluster_df.columns and not top_examples.empty:
            st.markdown("#### Time Entry Keyword Analysis")
            option_lookup = {}
            for option_number, (row_index, row) in enumerate(selected_cluster_df.loc[top_examples.index].iterrows(), 1):
                matter_name = str(row.get(matter_name_col, "")).strip() or "Unnamed matter"
                matter_id = str(row.get(matter_col, "")).strip()
                option_label = f"{option_number}. {matter_name} ({matter_id})" if matter_id else f"{option_number}. {matter_name}"
                option_lookup[option_label] = row_index

            selected_example_label = st.selectbox(
                "Select an example matter",
                options=list(option_lookup.keys()),
                key=f"time_entry_keyword_selector_{clean_column_name(selected_project)}",
            )
            selected_example_row = selected_cluster_df.loc[option_lookup[selected_example_label]]
            keyword_analysis = analyze_time_entry_keywords(selected_example_row.get(time_entry_text_col))

            keyword_cols = st.columns(2)
            with keyword_cols[0]:
                st.caption("Top keywords")
                st.write(", ".join(keyword_analysis["top_keywords"]) if keyword_analysis["top_keywords"] else "No usable keywords found.")
            with keyword_cols[1]:
                st.caption("Top phrases")
                st.write(", ".join(keyword_analysis["top_phrases"]) if keyword_analysis["top_phrases"] else "No usable phrases found.")
        elif not time_entry_text_col:
            st.caption("No time-entry text column was detected for keyword analysis.")
