import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from collections import Counter
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score


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
    "agreement", "agreements", "contract", "contracts", "lease", "leasing", "nda", "mnDA", "confidentiality",
    "incorporation", "incorporate", "corporate", "corporation", "company", "business", "shareholder", "shareholders",
    "estate", "probate", "will", "wills", "trust", "trusts", "poa", "power", "attorney", "representation",
    "litigation", "dispute", "claim", "claims", "settlement", "court", "motion", "pleading", "pleadings",
    "employment", "employee", "employer", "termination", "severance", "immigration", "visa", "permit",
    "trademark", "copyright", "ip", "intellectual", "property", "real", "transaction", "purchase", "sale",
    "family", "divorce", "separation", "custody", "support", "tax", "planning", "financing", "loan",
    "privacy", "policy", "terms", "service", "licensing", "license", "review", "advisory", "compliance",
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
        "dissolution of marriage": "divorce",
        "marriage dissolution": "divorce",
        "divorce children": "divorce with children",
        "divorce child": "divorce with children",
        "divorce with child": "divorce with children",
        "divorce w children": "divorce with children",
        "divorce w child": "divorce with children",
        "divorce no children": "divorce no children",
        "divorce without children": "divorce no children",
        "divorce no child": "divorce no children",
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
    if "divorce" in tokens:
        if "child" in tokens or "children" in tokens:
            if "contested" in tokens:
                return "contested divorce children"
            if "uncontested" in tokens:
                return "uncontested divorce children"
            return "divorce children"
        if "no children" in original_text or "without children" in original_text:
            return "divorce no children"
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


def create_project_clusters(
    df: pd.DataFrame,
    matter_name_col: str,
    billing_col: str,
    use_ai: bool,
    progress_bar=None,
    status_box=None,
    text_col: str | None = None,
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
        # Sub-projects can use the richer time-entry text if available, but still fall back to the normalized signature.
        normalized_source = working[source_col].fillna("").apply(normalize_legal_text_for_clustering)
        working["primary_project_text"] = np.where(
            normalized_source.str.len() >= 3,
            normalized_source,
            working["project_signature"],
        )

    working = working[working["primary_project_text"].astype(str).str.len() >= 3].copy().reset_index(drop=True)

    if working.empty:
        empty_summary_cols = [cluster_col, label_col, "matter_count", "total_billing", "avg_billing", "top_terms", "top_bigrams", "example_matter_names"]
        return pd.DataFrame(columns=empty_summary_cols), pd.DataFrame()

    n_usable = working[matter_name_col].nunique()

    if cluster_prefix == "subcluster":
        k_selected = choose_subcluster_count(n_usable)
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

        candidate_ks = get_candidate_k_values(len(working))
        if not candidate_ks:
            k_selected = 1
            k_selection_results = pd.DataFrame()
        else:
            keyword_diversity = estimate_project_keyword_diversity(working[matter_name_col], legal_keywords)
            candidate_ks = expand_candidate_ks_with_keyword_diversity(
                candidate_ks,
                keyword_diversity["estimated_project_signal_count"],
                len(working),
            )
            candidate_ks = expand_candidate_ks_with_seeded_categories(candidate_ks, seeded_category_count, len(working))

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
                n_usable_matters=len(working),
                seeded_category_count=seeded_category_count,
            )

    if k_selected <= 1 or len(working) < 2:
        working[cluster_col] = 0
    else:
        k_selected = min(k_selected, len(working) - 1)
        model = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
        working[cluster_col] = model.fit_predict(X)

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

        combined_terms = ", ".join(top_terms + top_bigrams)
        combined_examples = " | ".join(raw_examples[:10])

        signature_name, signature_share = dominant_signature_label(cluster_df[matter_name_col], min_share=0.35)

        project_name = signature_name

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
            fallback_signature_name, fallback_signature_share = dominant_signature_label(cluster_df[matter_name_col], min_share=0.25)
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

        cluster_rows.append({
            cluster_col: cluster_id,
            label_col: project_name,
            "display_label": display_label,
            "matter_count": cluster_df[matter_name_col].nunique(),
            "total_billing": numeric_billing.sum(),
            "avg_billing": numeric_billing.mean(),
            "top_terms": ", ".join(top_terms),
            "top_bigrams": ", ".join(top_bigrams),
            "example_matter_names": " | ".join(examples),
            "dominant_signature_share": signature_share,
        })

        if progress_bar is not None:
            progress_value = 65 + int(((index + 1) / max(1, len(working[cluster_col].unique()))) * 30)
            progress_bar.progress(min(progress_value, 95))

    if progress_bar is not None:
        progress_bar.progress(100)
    if status_box is not None:
        status_box.success("Clustering complete.")

    cluster_summary = pd.DataFrame(cluster_rows).sort_values(["matter_count", "total_billing"], ascending=[False, False])
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
    return (
        cluster_summary.groupby("project_name", as_index=False)
        .agg(
            total_matters=("matter_count", "sum"),
            total_billing=("total_billing", "sum"),
            avg_billing=("avg_billing", "mean"),
            n_clusters=("cluster_id", "nunique"),
        )
        .sort_values(["total_matters", "total_billing"], ascending=[False, False])
    )


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


def render_project_ranking(project_summary: pd.DataFrame) -> None:
    top_five = project_summary.head(5).copy()
    rows = []
    for index, row in top_five.reset_index(drop=True).iterrows():
        rows.append(
            f"<div class='ranking-row'>"
            f"<span class='ranking-name'>{index + 1}. {row['project_name']}</span>"
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


def render_revenue_ranking(project_summary: pd.DataFrame) -> None:
    revenue_df = project_summary.sort_values("total_billing", ascending=False).head(5).copy().reset_index(drop=True)
    top_total_project = project_summary.sort_values("total_billing", ascending=False).iloc[0]["project_name"]
    top_avg_project = project_summary.sort_values("avg_billing", ascending=False).iloc[0]["project_name"]

    rows = []
    for index, row in revenue_df.iterrows():
        name_class = ""
        value_class = ""
        suffix = ""

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
            f"<span class='ranking-name {name_class}'>{index + 1}. {row['project_name']}{suffix}</span>"
            f"<span class='ranking-value {value_class}'>{format_currency(row['total_billing'])}</span>"
            f"</div>"
        )

    html = (
        "<div class='ranking-card'>"
        "<div class='ranking-title'>Total Revenue</div>"
        f"{''.join(rows)}"
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
def make_project_option_label(project_name: str, has_subprojects: bool) -> str:
    return f"{project_name} *" if has_subprojects else project_name


def strip_project_option_label(option_label: str) -> str:
    return option_label.replace(" *", "")


st.markdown(
    """
    <div class="hero-card">
        <h1>Historical Matter Dashboard</h1>
        <p>Upload one account-level historical_matter_evidence CSV and get an initial billing overview before categorization begins.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="Expected structure: one row per matter, similar to historical_matter_evidence, filtered to one account.",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="section-card">
            <h3>Expected input</h3>
            <p class="small-muted">
                Upload a CSV with one account only. The app will infer the account, matter, date, and billing columns,
                then show a quick historical overview before categorization begins.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    raw_df = load_csv(uploaded_file)
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
total_billing = matter_df[billing_col].sum()
average_billing = matter_df[billing_col].mean()

valid_dates = matter_df["parsed_date"].dropna()
if valid_dates.empty:
    date_range_label = "No valid dates"
else:
    date_range_label = f"{valid_dates.min().date()} → {valid_dates.max().date()}"

default_granularity = choose_default_granularity(matter_df["parsed_date"])

overview_tab, clustering_tab = st.tabs(["Initial Overview", "Cluster Analysis"])

with overview_tab:
    st.markdown(f"### {account_name}")
    st.caption(f"Date range: {date_range_label}")
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

with clustering_tab:
    st.markdown("### Cluster Analysis")
    st.caption("Run a quick matter-name clustering pass, then inspect the main clusters and optional subclusters.")

    control_cols = st.columns((1.2, 1, 1))
    with control_cols[0]:
        use_ai = st.toggle(
            "Use AI labels with Ollama",
            value=False,
            help="When enabled, the app asks local Ollama to name each cluster. If Ollama fails, it falls back to TF-IDF terms.",
        )
    with control_cols[1]:
        run_clustering = st.button("Run cluster analysis", type="primary")
    with control_cols[2]:
        st.caption("Subclusters become available after a cluster is selected.")

    if use_ai:
        st.info("AI labeling requires Ollama running locally at http://localhost:11434 and the llama3.1:8b model pulled.")

    if run_clustering:
        st.session_state.pop("cluster_summary", None)
        st.session_state.pop("clustered_matters", None)
        st.session_state.pop("subclusters_by_cluster", None)

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
            )

        if cluster_summary.empty:
            st.error("No clusters could be created. Check that the file has usable matter names.")
        else:
            clustered_matters = add_numeric_columns(clustered_matters, billing_col, hours_col, rate_col, entries_col, users_col)
            st.session_state["cluster_summary"] = cluster_summary
            st.session_state["clustered_matters"] = clustered_matters
            st.session_state["use_ai"] = use_ai

    if "cluster_summary" not in st.session_state or "clustered_matters" not in st.session_state:
        st.info("Run the cluster analysis to see project names, charts, and cluster details.")
        st.stop()

    cluster_summary = st.session_state["cluster_summary"]
    clustered_matters = st.session_state["clustered_matters"]

    project_summary = aggregate_project_categories(cluster_summary)
    top_project_names = project_summary.head(3)["project_name"].tolist()
    project_color_map = get_project_color_map(top_project_names)

    subproject_candidate_projects = set(
        clustered_matters.groupby("project_name")[matter_name_col]
        .nunique()
        .loc[lambda s: s >= 50]
        .index
        .tolist()
    )

    top_layout = st.columns((1.1, 1, 1))
    with top_layout[0]:
        st.plotly_chart(build_project_ring_chart(project_summary, project_color_map), use_container_width=True)
    with top_layout[1]:
        render_project_ranking(project_summary)
    with top_layout[2]:
        render_revenue_ranking(project_summary)

    st.markdown("### Project Detail")

    detail_controls = st.columns((1.3, 1.3, 0.8))

    project_options_df = project_summary.copy()
    project_options_df["has_subprojects"] = project_options_df["project_name"].isin(subproject_candidate_projects)
    project_options_df["option_label"] = project_options_df.apply(
        lambda row: make_project_option_label(row["project_name"], row["has_subprojects"]),
        axis=1,
    )

    with detail_controls[0]:
        selected_project_option = st.selectbox(
            "Project",
            options=project_options_df["option_label"].tolist(),
            index=0,
            key="project_detail_selector",
        )

    selected_project = strip_project_option_label(selected_project_option)
    selected_cluster_df = clustered_matters[clustered_matters["project_name"] == selected_project].copy()
    selected_project_color = project_color_map.get(selected_project, ALTFEE_GRAY)

    selected_cluster_size = selected_cluster_df[matter_name_col].nunique()
    subcluster_text_source = time_entry_text_col if time_entry_text_col else matter_name_col

    if "subclusters_by_cluster" not in st.session_state:
        st.session_state["subclusters_by_cluster"] = {}

    subcluster_key = selected_project
    if selected_cluster_size >= 50 and subcluster_key not in st.session_state["subclusters_by_cluster"]:
        with st.spinner("Creating sub-projects..."):
            subcluster_summary_auto, subclustered_matters_auto = create_project_clusters(
                selected_cluster_df,
                matter_name_col=matter_name_col,
                billing_col=billing_col,
                use_ai=st.session_state.get("use_ai", False),
                text_col=subcluster_text_source,
                cluster_prefix="subcluster",
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

    with detail_controls[1]:
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

    with detail_controls[2]:
        detail_granularity = st.radio(
            "Time",
            options=["Monthly", "Yearly"],
            index=0 if default_granularity == "Monthly" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="project_detail_time_grouping_radio",
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

        st.dataframe(clean_examples.head(12), use_container_width=True, hide_index=True)