import sys, os, pickle, warnings, time

import numpy as _np_check
if int(_np_check.__version__.split(".")[0]) >= 2:
    import types, numpy
    # Recreate the numpy.core namespace that Cython extensions expect
    if not hasattr(numpy, "core") or not hasattr(numpy.core, "multiarray"):
        _core = types.ModuleType("numpy.core")
        _ma   = types.ModuleType("numpy.core.multiarray")
        # Expose the symbols Cython __init__ calls look for
        _ma._reconstruct    = numpy.zeros   # pickle fallback
        _ma.scalar          = numpy.generic
        _ma.ndarray         = numpy.ndarray
        _core.multiarray    = _ma
        _core.umath         = getattr(numpy, "core", types.ModuleType("x"))
        numpy.core          = _core
        sys.modules["numpy.core"]             = _core
        sys.modules["numpy.core.multiarray"]  = _ma
del _np_check
warnings.filterwarnings("ignore")

import __main__
def _dummy(*args, **kwargs): return None
__main__.predict_rating          = _dummy
__main__.get_cf_scores_for_user  = _dummy
__main__.get_user_rated_movies   = _dummy
__main__.get_similar_movies      = _dummy
__main__.get_cbf_scores_for_user = _dummy
__main__.recommend               = _dummy

import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI — Hybrid Recommender",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&family=Playfair+Display:wght@400;700&display=swap');

:root {
    --bg-void:     #050a07;
    --bg-deep:     #080f0a;
    --bg-card:     #0d1a10;
    --bg-raised:   #122016;
    --bg-hover:    #172a1a;
    --green-dim:   #1a4d28;
    --green-mid:   #2d7a45;
    --green-bright:#3daa5e;
    --green-glow:  #4dcc72;
    --green-light: #7de89a;
    --gold:        #c8a96e;
    --gold-light:  #e8c98a;
    --text-primary:#e8f5ec;
    --text-muted:  #7aaa88;
    --text-faint:  #3d6648;
    --border:      #1a3d22;
    --border-bright:#2d6640;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-void) !important;
    font-family: 'Cairo', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background: var(--bg-deep) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { font-family: 'Cairo', sans-serif !important; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* ── TOP NAVBAR ── */
.top-navbar {
    display: flex;
    align-items: center;
    gap: 0;
    background: var(--bg-deep);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem;
    margin: -1rem -1rem 2rem -1rem;
    position: sticky;
    top: 0;
    z-index: 999;
}
.top-navbar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--green-glow);
    letter-spacing: 0.05em;
    padding: 1rem 2rem 1rem 0;
    border-right: 1px solid var(--border);
    margin-right: 1.5rem;
    white-space: nowrap;
}
.top-navbar-link {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 1rem 1.2rem;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: color 0.2s, border-color 0.2s;
    text-decoration: none;
    white-space: nowrap;
}
.top-navbar-link:hover { color: var(--green-light); }
.top-navbar-link.active {
    color: var(--green-glow);
    border-bottom: 2px solid var(--green-glow);
}
.top-navbar-stats {
    margin-left: auto;
    display: flex;
    gap: 0.5rem;
    align-items: center;
    font-size: 0.7rem;
    color: var(--text-faint);
}

/* Headings */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--text-primary) !important;
}

/* Inputs */
.stSelectbox > div > div,
.stSlider > div,
.stNumberInput > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
}

.stSelectbox label, .stSlider label, .stNumberInput label,
.stMultiSelect label, .stTextInput label {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-family: 'Cairo', sans-serif !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--green-mid) !important;
    color: var(--green-glow) !important;
    font-family: 'Cairo', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    border-radius: 3px !important;
    transition: all 0.25s ease !important;
    padding: 0.5rem 1.8rem !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
}
.stButton > button:hover {
    background: var(--green-dim) !important;
    border-color: var(--green-glow) !important;
    box-shadow: 0 0 18px rgba(61, 170, 94, 0.25) !important;
}

/* Multiselect tags */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: var(--green-dim) !important;
    color: var(--green-light) !important;
    border: 1px solid var(--green-mid) !important;
    border-radius: 2px !important;
}

/* Metric boxes */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--green-glow) !important;
    font-family: 'Cairo', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    border-radius: 4px !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Cairo', sans-serif !important;
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--green-glow) !important;
    border-bottom-color: var(--green-glow) !important;
}

/* Progress / spinners */
.stProgress > div > div > div {
    background: var(--green-mid) !important;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--green-dim); border-radius: 2px; }

/* Custom card component */
.movie-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.movie-card:hover {
    transform: translateY(-3px);
    border-color: var(--green-mid);
    box-shadow: 0 8px 32px rgba(45, 122, 69, 0.2);
}
.movie-card-body { padding: 0.75rem; }
.movie-title {
    font-family: 'Cairo', sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
    color: var(--text-primary);
    margin: 0 0 0.25rem 0;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.movie-genres {
    font-size: 0.68rem;
    color: var(--text-faint);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    display: -webkit-box;
    -webkit-line-clamp: 1;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.score-bar-wrap {
    background: var(--bg-raised);
    border-radius: 2px;
    height: 3px;
    margin-bottom: 0.4rem;
}
.score-bar {
    height: 3px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--green-mid), var(--green-glow));
}
.score-label {
    font-size: 0.72rem;
    color: var(--green-mid);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.source-badge {
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 2px;
}
.source-hybrid { background: rgba(45,122,69,0.3); color: var(--green-glow); border: 1px solid var(--green-dim); }
.source-cf     { background: rgba(200,169,110,0.15); color: var(--gold); border: 1px solid rgba(200,169,110,0.3); }
.source-cbf    { background: rgba(61,170,94,0.1); color: var(--green-light); border: 1px solid var(--green-dim); }

/* Page header */
.page-header {
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.page-header-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1;
}
.page-header-sub {
    font-size: 0.78rem;
    color: var(--text-faint);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.5rem;
    font-family: 'Cairo', sans-serif;
}
.accent-line {
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, var(--green-glow), transparent);
    margin: 1rem 0;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-faint);
    font-family: 'Cairo', sans-serif;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Poster placeholder */
.poster-placeholder {
    width: 100%;
    aspect-ratio: 2/3;
    background: var(--bg-raised);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: var(--text-faint);
    border-bottom: 1px solid var(--border);
}

/* Stats row */
.stat-chip {
    display: inline-block;
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 4px 10px;
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    font-family: 'Cairo', sans-serif;
    margin-right: 6px;
    margin-bottom: 6px;
}
.stat-chip strong { color: var(--green-glow); font-weight: 700; }

/* Table styling */
[data-testid="stDataFrame"] table {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 0.82rem !important;
}

/* Alert-like info boxes */
.info-box {
    background: rgba(45,122,69,0.08);
    border: 1px solid var(--border);
    border-left: 3px solid var(--green-mid);
    border-radius: 3px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: var(--text-muted);
    margin: 0.8rem 0;
    font-family: 'Cairo', sans-serif;
}

/* Sidebar logo area */
.sidebar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--green-glow);
    letter-spacing: -0.02em;
    padding: 1rem 0 0.5rem 0;
}
.sidebar-tagline {
    font-size: 0.65rem;
    color: var(--text-faint);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-family: 'Cairo', sans-serif;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── PATHS ────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\mohaf\Downloads\Final Section Intelligent\DataSets"
PROC_DIR = os.path.join(BASE_DIR, "processed")

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"  # public demo key — replace if needed
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"
POSTER_FALLBACK = "https://via.placeholder.com/342x513/0d1a10/3daa5e?text=No+Poster"


# ── DATA LOADING ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_artifacts():
    """Load all pkl artifacts once and cache them."""
    with open(os.path.join(PROC_DIR, "preprocessed_data.pkl"), "rb") as f:
        prep = pickle.load(f)
    with open(os.path.join(PROC_DIR, "cbf_artifacts.pkl"), "rb") as f:
        cbf = pickle.load(f)
    with open(os.path.join(PROC_DIR, "svd_model.pkl"), "rb") as f:
        svd = pickle.load(f)
    with open(os.path.join(PROC_DIR, "hybrid_artifacts.pkl"), "rb") as f:
        hyb = pickle.load(f)
    return prep, cbf, svd, hyb


@st.cache_data(show_spinner=False)
def get_poster_url(tmdb_id):
    """Fetch poster URL from TMDB API."""
    if not tmdb_id or tmdb_id == 0:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
        r = requests.get(url, timeout=4)
        data = r.json()
        path = data.get("poster_path")
        if path:
            return TMDB_IMG_BASE + path
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def get_movie_details_tmdb(tmdb_id):
    """Fetch full movie details + credits from TMDB."""
    if not tmdb_id or tmdb_id == 0:
        return {}
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&append_to_response=credits"
        r = requests.get(url, timeout=5)
        return r.json()
    except Exception:
        return {}


# ── LOAD DATA ────────────────────────────────────────────────
import importlib

def _check_versions():
    lines = []
    for pkg, mod in [("numpy","numpy"),("scipy","scipy"),("sklearn","sklearn"),("surprise","surprise")]:
        try:
            m = importlib.import_module(mod)
            lines.append(f"{pkg}=={getattr(m,'__version__','?')}")
        except Exception:
            lines.append(f"{pkg}=NOT FOUND")
    return "  |  ".join(lines)

with st.spinner("Loading system..."):
    try:
        prep, cbf, svd_art, hyb = load_all_artifacts()

        ratings      = prep["ratings"]
        movies       = prep["movies_clean"].copy()
        R_MIN        = prep["R_MIN"]
        R_MAX        = prep["R_MAX"]
        popular_movies = prep.get("popular_movies", [])
        global_mean  = prep.get("global_mean_rating", 3.5)

        tfidf_matrix     = cbf["tfidf_matrix"]
        movie_idx_map    = cbf["cbf_movieId_to_pos"]
        idx_movie_map    = cbf["cbf_pos_to_movieId"]

        svd_eval    = svd_art["svd_eval"]
        svd_prod    = svd_art["svd_prod"]
        testset     = svd_art["testset"]
        eval_metrics = svd_art.get("eval_metrics", {})

        best_alpha   = hyb["best_alpha"]
        hybrid_metrics = hyb.get("metrics", {})
        popular_ids  = hyb.get("popular_ids", [])
        ALL_MOVIE_IDS = hyb.get("all_movie_ids", movies["movieId"].tolist())

        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

        _pop_df = (
            ratings.groupby("movieId")
            .agg(n=("rating","count"), avg=("rating","mean"))
            .reset_index()
        )
        _pop_df = _pop_df[_pop_df["n"] >= 10].sort_values("avg", ascending=False)
        _pop_ids_local = _pop_df["movieId"].tolist()[:100]

        def _cf_scores(user_id, cands):
            if not cands:
                return pd.DataFrame(columns=["movieId","cf_score"])
            rows = [{"movieId": m, "cf_raw": svd_prod.predict(user_id, m).est} for m in cands]
            df = pd.DataFrame(rows)
            lo, hi = df["cf_raw"].min(), df["cf_raw"].max()
            df["cf_score"] = (df["cf_raw"]-lo)/(hi-lo) if hi>lo else 0.5
            return df[["movieId","cf_score"]]

        def _cbf_scores(user_id, cands):
            if not cands:
                return pd.DataFrame(columns=["movieId","cbf_score"])
            urows = ratings[ratings["userId"]==user_id][["movieId","rating"]]
            profile = None
            for _, r in urows.iterrows():
                mid = int(r["movieId"])
                if mid not in movie_idx_map: continue
                vec = tfidf_matrix[movie_idx_map[mid]]
                w = float(r["rating"])/5.0
                profile = vec*w if profile is None else profile + vec*w
            rows = []
            for m in cands:
                if profile is None or m not in movie_idx_map:
                    rows.append({"movieId": m, "cbf_raw": 0.0})
                else:
                    sim = _cos_sim(profile, tfidf_matrix[movie_idx_map[m]])[0][0]
                    rows.append({"movieId": m, "cbf_raw": float(sim)})
            df = pd.DataFrame(rows)
            lo, hi = df["cbf_raw"].min(), df["cbf_raw"].max()
            df["cbf_score"] = (df["cbf_raw"]-lo)/(hi-lo) if hi>lo else 0.5
            return df[["movieId","cbf_score"]]

        def _cbf_scores_from_profile(profile, cands):
            if not cands or profile is None:
                return pd.DataFrame(columns=["movieId","cbf_score"])
            rows = []
            for m in cands:
                if m not in movie_idx_map:
                    rows.append({"movieId": m, "cbf_raw": 0.0})
                else:
                    sim = _cos_sim(profile, tfidf_matrix[movie_idx_map[m]])[0][0]
                    rows.append({"movieId": m, "cbf_raw": float(sim)})
            df = pd.DataFrame(rows)
            lo, hi = df["cbf_raw"].min(), df["cbf_raw"].max()
            df["cbf_score"] = (df["cbf_raw"]-lo)/(hi-lo) if hi>lo else 0.5
            return df[["movieId","cbf_score"]]

        def recommend_fn(user_id, top_n=10, alpha=0.65, genre_filter=None, **_kw):
            rated = set(ratings[ratings["userId"]==user_id]["movieId"].tolist())
            if not rated:
                recs = movies[movies["movieId"].isin(_pop_ids_local)].copy()
                recs = recs.merge(_pop_df[["movieId","avg"]], on="movieId", how="left")
                recs["hybrid_score"] = recs["avg"]/5.0
                recs["cf_score"] = recs["hybrid_score"]
                recs["cbf_score"] = recs["hybrid_score"]
                recs["source"] = "popularity"
                if genre_filter:
                    recs = recs[recs["genres"].apply(
                        lambda g: any(f.lower() in str(g).lower() for f in genre_filter))]
                return recs[["movieId","title","genres","cf_score","cbf_score",
                             "hybrid_score","source"]].head(top_n).reset_index(drop=True)
            cands = [m for m in ALL_MOVIE_IDS if m not in rated]
            cf  = _cf_scores(user_id, cands)
            cbf = _cbf_scores(user_id, cands)
            merged = cf.merge(cbf, on="movieId", how="inner")
            merged["hybrid_score"] = alpha*merged["cf_score"] + (1-alpha)*merged["cbf_score"]
            merged = merged.sort_values("hybrid_score", ascending=False)
            merged = merged.merge(movies[["movieId","title","genres"]], on="movieId", how="left")
            if genre_filter:
                merged = merged[merged["genres"].apply(
                    lambda g: any(f.lower() in str(g).lower() for f in genre_filter))]
            merged["source"] = "hybrid"
            return merged[["movieId","title","genres","cf_score","cbf_score",
                           "hybrid_score","source"]].head(top_n).reset_index(drop=True)

        # Build fast lookup structures
        movies_idx = movies.set_index("movieId")
        ALL_USERS  = sorted(ratings["userId"].unique().tolist())
        ALL_GENRES = sorted(set(
            g for glist in movies["genres"].str.split("|") for g in glist if g != "(no genres listed)"
        ))

        ARTIFACTS_OK = True
    except Exception as e:
        ARTIFACTS_OK = False
        import traceback
        LOAD_ERROR   = str(e)
        LOAD_TRACE   = traceback.format_exc()
        LOAD_VERSIONS = _check_versions()


# ── TOP NAVBAR ───────────────────────────────────────────────
NAV_PAGES = ["Recommendations", "Explorer", "Movie Explorer", "User Profile", "System Analytics", "Dataset Overview"]

# Session state navigation — persists across reruns, works reliably
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Recommendations"

# CSS for the navbar container and nav buttons
st.markdown("""
<style>
/* ── Navbar wrapper ── */
.navbar-wrapper {
    display: flex;
    align-items: stretch;
    background: var(--bg-deep);
    border-bottom: 1px solid var(--border);
    padding: 0 0.5rem;
    margin: -1rem -1rem 2rem -1rem;
    position: sticky;
    top: 0;
    z-index: 999;
    gap: 0;
}
.navbar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--green-glow);
    padding: 0.85rem 1.8rem 0.85rem 0.75rem;
    border-right: 1px solid var(--border);
    margin-right: 0.25rem;
    white-space: nowrap;
    display: flex;
    align-items: center;
    flex-shrink: 0;
    letter-spacing: 0.02em;
}

/* Streamlit column and button resets inside navbar */
.navbar-cols [data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    background: var(--bg-deep) !important;
    align-items: stretch !important;
    flex-wrap: nowrap !important;
}
.navbar-cols [data-testid="column"] {
    padding: 0 !important;
    min-width: 0 !important;
    flex-shrink: 1 !important;
}
.navbar-cols [data-testid="stButton"] {
    width: 100%;
}
.navbar-cols [data-testid="stButton"] > button {
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: var(--text-muted) !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.9rem 0.9rem !important;
    white-space: nowrap !important;
    cursor: pointer !important;
    transition: color 0.2s, border-color 0.2s, background 0.2s !important;
    box-shadow: none !important;
    height: 100% !important;
}
.navbar-cols [data-testid="stButton"] > button:hover {
    color: var(--green-light) !important;
    background: var(--bg-raised) !important;
    border-color: transparent !important;
    box-shadow: none !important;
}
/* Active page button */
.navbar-cols [data-testid="stButton"] > button[kind="primary"] {
    color: var(--green-glow) !important;
    border-bottom: 2px solid var(--green-glow) !important;
    background: transparent !important;
    box-shadow: none !important;
}
.navbar-cols [data-testid="stButton"] > button[kind="primary"]:hover {
    background: var(--bg-raised) !important;
}
</style>
""", unsafe_allow_html=True)

# Render navbar brand + buttons
st.markdown('<div class="navbar-wrapper"><div class="navbar-brand">CineAI</div></div>', unsafe_allow_html=True)

# Inject navbar buttons in columns — wrapped in a div for scoped CSS
st.markdown('<div class="navbar-cols">', unsafe_allow_html=True)
nav_cols = st.columns(len(NAV_PAGES))
for col, page in zip(nav_cols, NAV_PAGES):
    with col:
        is_active = (st.session_state.nav_page == page)
        if st.button(
            page,
            key=f"nav_{page}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.nav_page = page
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

nav = st.session_state.nav_page

# Stats row below navbar
if ARTIFACTS_OK:
    st.markdown(f"""
    <div style="display:flex; gap:0.6rem; flex-wrap:wrap; margin-bottom:1.5rem; margin-top:-1rem;">
      <span class="stat-chip"><strong>{ratings['userId'].nunique():,}</strong> Users</span>
      <span class="stat-chip"><strong>{len(movies):,}</strong> Movies</span>
      <span class="stat-chip"><strong>{len(ratings):,}</strong> Ratings</span>
      <span class="stat-chip">a = {best_alpha} Blend</span>
      <span class="stat-chip">RMSE <strong style="color:var(--green-glow)">{hybrid_metrics.get('rmse', 0):.4f}</strong></span>
      <span class="stat-chip">P@10 <strong style="color:var(--green-glow)">{hybrid_metrics.get('P@10', 0):.4f}</strong></span>
    </div>
    """, unsafe_allow_html=True)


if not ARTIFACTS_OK:
    st.markdown("""
    <div style="background:#0d1a10; border:1px solid #2d7a45; border-left:4px solid #cc4444;
                border-radius:4px; padding:1.5rem; font-family:'Cairo',sans-serif; max-width:800px;">
      <div style="font-size:1rem; color:#e8f5ec; font-weight:700; margin-bottom:0.75rem;">
        Artifact Load Failed
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.code(LOAD_ERROR, language="text")

    if "multiarray" in LOAD_ERROR or "numpy" in LOAD_ERROR.lower():
        st.info("NumPy 2.x compatibility shim failed. Try: conda install numpy=1.26 in your fakenews env.")

    with st.expander("Full traceback"):
        st.code(LOAD_TRACE if 'LOAD_TRACE' in dir() else "—", language="python")
    st.stop()


# ── HELPER FUNCTIONS ─────────────────────────────────────────
def make_poster_img(tmdb_id, width=None):
    """Return a poster URL or None if unavailable."""
    return get_poster_url(tmdb_id)


def render_poster(tmdb_id, title=""):
    url = get_poster_url(tmdb_id)
    if url:
        st.image(url, use_container_width=True)
    else:
        initial = (title.strip()[0].upper()) if title.strip() else "?"
        st.markdown(f"""
        <div style="
            width:100%;
            aspect-ratio:2/3;
            background: linear-gradient(160deg, var(--bg-raised) 0%, var(--bg-card) 100%);
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        ">
            <div style="
                width: 56px; height: 56px;
                border-radius: 50%;
                background: var(--green-dim);
                border: 1px solid var(--green-mid);
                display: flex; align-items: center; justify-content: center;
                font-size: 1.6rem; font-weight: 700;
                color: var(--green-glow);
                font-family: 'Playfair Display', serif;
            ">{initial}</div>
            <div style="font-size:0.6rem; letter-spacing:0.18em; text-transform:uppercase;
                        color:var(--text-faint); font-family:'Cairo',sans-serif;">No Poster</div>
        </div>
        """, unsafe_allow_html=True)


def render_movie_card(movie_row, score=None, source=None, show_poster=True):
    """Render a styled movie card with poster, title, genres, score bar."""
    tmdb_id = movie_row.get("tmdbId", 0)
    title   = movie_row.get("title", "Unknown")

    # Poster — uses styled HTML fallback, never a broken image
    if show_poster:
        render_poster(tmdb_id, title)

    # Metadata
    score_pct = int((score or 0) * 100)
    source_cls = f"source-{str(source).lower()}" if source else "source-hybrid"
    source_label = str(source).upper() if source else ""

    genres_raw = movie_row.get("genres", "")
    genres_display = " / ".join(str(genres_raw).split("|")[:3]) if genres_raw else ""

    year = movie_row.get("year", "")
    rating_count = int(movie_row.get("rating_count", 0))
    rating_mean  = float(movie_row.get("rating_mean", 0))

    # title already extracted above for render_poster; just compute display version
    display_title = title.rsplit("(", 1)[0].strip() if "(" in str(title) else str(title)

    html_parts = [
        '<div class="movie-card"><div class="movie-card-body">',
        f'<div class="movie-title">{display_title}</div>',
        f'<div class="movie-genres">{genres_display}</div>',
    ]

    if score is not None:
        source_badge = (
            f'<span class="source-badge source-{str(source).lower()}">{str(source).upper()}</span>'
            if source else ""
        )
        html_parts.append(
            f'<div class="score-bar-wrap">'
            f'<div class="score-bar" style="width:{score_pct}%"></div></div>'
            f'<div class="score-label"><span>{round(score, 3)}</span>{source_badge}</div>'
        )

    if year or rating_count:
        html_parts.append(
            f'<div style="margin-top:0.35rem;font-size:0.68rem;color:var(--text-faint);">'
            f'{str(year)} | {rating_mean:.1f} avg | {rating_count:,} ratings'
            f'</div>'
        )

    html_parts.append('</div></div>')

    st.markdown("".join(html_parts), unsafe_allow_html=True)


def get_movie_row(movie_id):
    """Safe lookup of a movie by movieId."""
    if movie_id in movies_idx.index:
        return movies_idx.loc[movie_id]
    return None

if nav == "Recommendations":
    st.markdown("""
    <div class="page-header">
      <div class="page-header-title">Recommendations</div>
      <div class="accent-line"></div>
      <div class="page-header-sub">Hybrid SVD + Content-Based Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    col_a, col_b, col_c, col_d = st.columns([2, 1, 2, 1])
    with col_a:
        selected_user = st.selectbox("User ID", ALL_USERS, index=0)
    with col_b:
        top_n = st.number_input("Results", min_value=5, max_value=30, value=10, step=5)
    with col_c:
        genre_filter_ui = st.multiselect("Filter by Genre", ALL_GENRES, default=[])
    with col_d:
        alpha_override = st.slider("CF Weight (alpha)", 0.0, 1.0, float(best_alpha), 0.05)

    run_btn = st.button("Generate Recommendations")

    if run_btn or "last_recs" in st.session_state:
        with st.spinner("Computing hybrid scores..."):
            genre_arg = genre_filter_ui if genre_filter_ui else None
            recs = recommend_fn(
                selected_user,
                top_n=int(top_n),
                alpha=alpha_override,
                genre_filter=genre_arg
            )
            st.session_state["last_recs"] = recs
            st.session_state["last_user"] = selected_user

        recs = st.session_state["last_recs"]

        if recs.empty:
            st.markdown('<div class="info-box">No recommendations found for this user / filter combination.</div>', unsafe_allow_html=True)
        else:
            # User info strip
            user_ratings = ratings[ratings["userId"] == selected_user]
            top_genres_user = (
                pd.Series(
                    [g for glist in user_ratings.merge(movies[["movieId","genres"]], on="movieId", how="left")["genres"].dropna().str.split("|")
                     for g in glist if g != "(no genres listed)"]
                ).value_counts().head(4).index.tolist()
            )
            st.markdown(f"""
            <div class="info-box">
            User <strong style="color:var(--green-glow)">{selected_user}</strong> — 
            {len(user_ratings):,} ratings, avg {user_ratings['rating'].mean():.2f} 
            &nbsp;|&nbsp; Top genres: <strong style="color:var(--green-glow)">{" / ".join(top_genres_user)}</strong>
            </div>
            """, unsafe_allow_html=True)

            # Grid of cards
            st.markdown('<div class="section-label">Top Recommendations</div>', unsafe_allow_html=True)
            cols_per_row = 5
            items = recs.to_dict("records")
            for row_start in range(0, len(items), cols_per_row):
                cols = st.columns(cols_per_row)
                for i, rec in enumerate(items[row_start:row_start+cols_per_row]):
                    with cols[i]:
                        mrow = get_movie_row(rec["movieId"])
                        if mrow is None:
                            continue
                        tmdb_id = mrow.get("tmdbId", 0)
                        render_poster(tmdb_id, str(mrow.get("title", "")))
                        render_movie_card(
                            mrow.to_dict(),
                            score=rec.get("hybrid_score"),
                            source=rec.get("source"),
                            show_poster=False
                        )

            # Score breakdown table
            with st.expander("Score Breakdown"):
                display_cols = ["movieId", "title", "genres", "cf_score", "cbf_score", "hybrid_score", "source"]
                show_cols = [c for c in display_cols if c in recs.columns]
                st.dataframe(
                    recs[show_cols].reset_index(drop=True),
                    use_container_width=True,
                    height=300
                )

elif nav == "Explorer":
    st.markdown("""
    <div class="page-header">
      <div class="page-header-title">Explorer</div>
      <div class="accent-line"></div>
      <div class="page-header-sub">Build your taste profile and get personalised recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Search for movies you have already seen, give each a rating, and the engine will build
    a content profile on the fly and recommend what to watch next.
    The more movies you add the better the results.
    </div>
    """, unsafe_allow_html=True)

    movie_titles_exp = movies[["movieId","title"]].copy().sort_values("title")
    title_to_id_exp  = dict(zip(movie_titles_exp["title"], movie_titles_exp["movieId"]))
    all_exp_labels   = list(title_to_id_exp.keys())

    st.markdown("#### Add Movies You Have Seen")

    if "explorer_liked" not in st.session_state:
        st.session_state["explorer_liked"] = []

    col_search, col_rating, col_add = st.columns([4, 2, 1])
    with col_search:
        picked_label = st.selectbox("Search movie title", all_exp_labels, index=0, key="explorer_search")
    with col_rating:
        picked_rating = st.slider("Your rating", 1.0, 5.0, 4.0, 0.5, key="explorer_rating")
    with col_add:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add", key="explorer_add"):
            pid = title_to_id_exp[picked_label]
            existing_ids = [e["movieId"] for e in st.session_state["explorer_liked"]]
            if pid in existing_ids:
                for e in st.session_state["explorer_liked"]:
                    if e["movieId"] == pid:
                        e["rating"] = picked_rating
                st.toast(f"Updated rating for {picked_label[:40]}")
            else:
                st.session_state["explorer_liked"].append(
                    {"movieId": pid, "title": picked_label, "rating": picked_rating})
                st.toast(f"Added: {picked_label[:40]}")

    liked_list = st.session_state["explorer_liked"]

    if liked_list:
        st.markdown("#### Your Taste Profile")
        remove_idx = None
        for i, item in enumerate(liked_list):
            col_t, col_r, col_x = st.columns([5, 2, 1])
            with col_t:
                st.markdown(f"**{item['title'][:60]}**")
            with col_r:
                stars_filled = int(item["rating"])
                st.markdown(f"`{item['rating']:.1f}` {'*' * stars_filled}{'_' * (5 - stars_filled)}")
            with col_x:
                if st.button("Remove", key=f"exp_rm_{i}"):
                    remove_idx = i
        if remove_idx is not None:
            st.session_state["explorer_liked"].pop(remove_idx)
            st.rerun()

        st.markdown("---")

        col_c1, col_c2, col_c3 = st.columns([2, 2, 2])
        with col_c1:
            exp_top_n = st.number_input("Results", 5, 30, 10, 5, key="exp_topn")
        with col_c2:
            exp_genre = st.multiselect("Filter by Genre", ALL_GENRES, key="exp_genre")
        with col_c3:
            exp_alpha = st.slider("Content Weight", 0.0, 1.0, 0.8, 0.05, key="exp_alpha",
                                  help="Higher = genre/tag similarity drives results. Lower = popularity drives results.")

        if st.button("Get Recommendations", key="explorer_run"):
            liked_ids     = [e["movieId"] for e in liked_list]
            liked_ratings = {e["movieId"]: e["rating"] for e in liked_list}

            with st.spinner("Building taste profile and scoring movies..."):
                profile = None
                for mid, rat in liked_ratings.items():
                    if mid not in movie_idx_map:
                        continue
                    vec = tfidf_matrix[movie_idx_map[mid]]
                    w   = rat / 5.0
                    profile = vec * w if profile is None else profile + vec * w

                cands = [m for m in ALL_MOVIE_IDS if m not in liked_ids]
                cbf_df = _cbf_scores_from_profile(profile, cands)

                pop_merged = cbf_df.merge(_pop_df[["movieId","avg"]], on="movieId", how="left")
                pop_merged["avg"] = pop_merged["avg"].fillna(0)
                pop_lo = pop_merged["avg"].min()
                pop_hi = pop_merged["avg"].max()
                pop_merged["pop_score"] = (
                    (pop_merged["avg"] - pop_lo) / (pop_hi - pop_lo)
                    if pop_hi > pop_lo else 0.5
                )

                pop_merged["hybrid_score"] = (
                    exp_alpha * pop_merged["cbf_score"] +
                    (1.0 - exp_alpha) * pop_merged["pop_score"]
                )

                result = pop_merged.merge(
                    movies[["movieId","title","genres","year","tmdbId","rating_count","rating_mean"]],
                    on="movieId", how="left"
                ).sort_values("hybrid_score", ascending=False)

                if exp_genre:
                    result = result[result["genres"].apply(
                        lambda g: any(f.lower() in str(g).lower() for f in exp_genre))]

                result["source"] = "explorer"
                st.session_state["explorer_recs"] = result.head(int(exp_top_n)).reset_index(drop=True)

        if "explorer_recs" in st.session_state:
            recs_exp = st.session_state["explorer_recs"]
            if recs_exp.empty:
                st.markdown('<div class="info-box">No results — try removing the genre filter or adding more movies.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="section-label">Top Recommendations For You</div>',
                            unsafe_allow_html=True)
                cols_per_row = 5
                for row_start in range(0, len(recs_exp), cols_per_row):
                    row_movies = recs_exp.iloc[row_start:row_start + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, (_, mrow) in zip(cols, row_movies.iterrows()):
                        with col:
                            render_movie_card(mrow.to_dict(), score=mrow["hybrid_score"], source="explorer")
    else:
        st.markdown("""
        <div class="info-box" style="text-align:center; padding:2rem;">
        Search for a movie above and click <strong>Add</strong> to start building your profile.
        </div>
        """, unsafe_allow_html=True)


elif nav == "Movie Explorer":
    st.markdown("""
    <div class="page-header">
      <div class="page-header-title">Movie Explorer</div>
      <div class="accent-line"></div>
      <div class="page-header-sub">Browse, Search & Discover Similar Films</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Search & Details", "Similar Movies", "Popular Catalog"])

    with tab1:
        movie_titles = movies[["movieId","title"]].copy()
        movie_titles["label"] = movie_titles["title"] + "  [" + movie_titles["movieId"].astype(str) + "]"
        title_to_id = dict(zip(movie_titles["label"], movie_titles["movieId"]))

        search_val = st.selectbox("Select a Movie", list(title_to_id.keys()), index=0)
        chosen_id  = title_to_id[search_val]
        mrow       = get_movie_row(chosen_id)

        if mrow is not None:
            tmdb_id  = int(mrow.get("tmdbId", 0))
            tmdb_data = get_movie_details_tmdb(tmdb_id)

            col_poster, col_info = st.columns([1, 3])

            with col_poster:
                render_poster(tmdb_id, str(mrow.get("title", "")))

            with col_info:
                title_clean = str(mrow["title"])
                st.markdown(f"""
                <div style="font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:700;
                            color:var(--text-primary); line-height:1.2; margin-bottom:0.5rem;">
                  {title_clean.rsplit("(",1)[0].strip()}
                </div>
                <div style="font-size:0.75rem; color:var(--text-faint); letter-spacing:0.15em;
                            text-transform:uppercase; margin-bottom:1rem;">
                  {mrow.get('year','')} &nbsp;|&nbsp; {str(mrow.get('genres','')).replace('|', ' / ')}
                </div>
                """, unsafe_allow_html=True)

                # TMDB overview
                overview = tmdb_data.get("overview", "")
                if overview:
                    st.markdown(f"""
                    <div style="font-size:0.85rem; color:var(--text-muted); line-height:1.7;
                                max-width:600px; margin-bottom:1rem;">
                      {overview}
                    </div>
                    """, unsafe_allow_html=True)

                # Stats row
                rc = int(mrow.get("rating_count", 0))
                rm = float(mrow.get("rating_mean", 0))
                runtime = tmdb_data.get("runtime", "")
                vote_avg = tmdb_data.get("vote_average", "")
                tagline = tmdb_data.get("tagline", "")

                st.markdown(f"""
                <div style="margin-bottom:0.8rem;">
                  <span class="stat-chip"><strong>{rc:,}</strong> Ratings</span>
                  <span class="stat-chip"><strong>{rm:.2f}</strong> / 5.0 Avg</span>
                  {f'<span class="stat-chip"><strong>{runtime}</strong> min</span>' if runtime else ''}
                  {f'<span class="stat-chip">TMDB <strong>{vote_avg}</strong></span>' if vote_avg else ''}
                </div>
                {f'<div style="font-size:0.78rem; color:var(--text-faint); font-style:italic;">{tagline}</div>' if tagline else ''}
                """, unsafe_allow_html=True)

                # Cast
                credits = tmdb_data.get("credits", {})
                cast = credits.get("cast", [])[:6]
                crew = credits.get("crew", [])
                director = next((p["name"] for p in crew if p.get("job") == "Director"), None)

                if director:
                    st.markdown(f"""
                    <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.4rem;">
                      <span style="color:var(--text-faint); text-transform:uppercase; letter-spacing:0.1em;">Director</span>
                      &nbsp; {director}
                    </div>
                    """, unsafe_allow_html=True)

                if cast:
                    cast_names = " &nbsp;·&nbsp; ".join(p["name"] for p in cast)
                    st.markdown(f"""
                    <div style="font-size:0.75rem; color:var(--text-muted);">
                      <span style="color:var(--text-faint); text-transform:uppercase; letter-spacing:0.1em;">Cast</span>
                      &nbsp; {cast_names}
                    </div>
                    """, unsafe_allow_html=True)

                # IMDB link
                imdb_id = mrow.get("imdbId")
                if imdb_id and str(imdb_id) != "0":
                    st.markdown(f"""
                    <div style="margin-top:1rem;">
                      <a href="https://www.imdb.com/title/tt{str(int(imdb_id)).zfill(7)}/"
                         target="_blank"
                         style="color:var(--gold); font-size:0.78rem; text-decoration:none;
                                border:1px solid rgba(200,169,110,0.3); padding:4px 12px;
                                border-radius:3px; letter-spacing:0.08em; text-transform:uppercase;">
                        View on IMDB
                      </a>
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-label">Content-Based Similar Movies</div>', unsafe_allow_html=True)
        sim_movie_val = st.selectbox("Movie to find similars for", list(title_to_id.keys()),
                                      index=0, key="sim_select")
        sim_n = st.slider("Number of similar movies", 5, 20, 10, key="sim_n")
        sim_genres_only = st.checkbox("Genre signal only (faster)", value=False)

        sim_id = title_to_id[sim_movie_val]

        if sim_id in movie_idx_map:
            from sklearn.metrics.pairwise import linear_kernel
            mat = cbf.get("tfidf_genres_matrix" if sim_genres_only else "tfidf_matrix", tfidf_matrix)
            pos = movie_idx_map[sim_id]
            sims = linear_kernel(mat[pos], mat).flatten()
            sims[pos] = -1
            top_idx = np.argpartition(sims, -sim_n)[-sim_n:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

            sim_movies_data = movies.iloc[top_idx].copy()
            sim_movies_data["similarity"] = sims[top_idx].round(4)

            cols_per_row = 5
            for row_start in range(0, min(sim_n, len(sim_movies_data)), cols_per_row):
                cols = st.columns(cols_per_row)
                chunk = sim_movies_data.iloc[row_start:row_start+cols_per_row]
                for i, (_, mrow_s) in enumerate(chunk.iterrows()):
                    with cols[i]:
                        render_poster(mrow_s.get("tmdbId", 0), str(mrow_s.get("title", "")))
                        render_movie_card(
                            mrow_s.to_dict(),
                            score=mrow_s["similarity"],
                            source="CBF",
                            show_poster=False
                        )
        else:
            st.markdown('<div class="info-box">Movie not in content-based index.</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-label">Popular Movies in Catalog</div>', unsafe_allow_html=True)

        col_g, col_s, col_mn = st.columns([2, 2, 1])
        with col_g:
            genre_browse = st.selectbox("Genre", ["All"] + ALL_GENRES, key="browse_genre")
        with col_s:
            sort_by = st.selectbox("Sort by", ["Rating Count", "Average Rating", "Year (Newest)"], key="browse_sort")
        with col_mn:
            min_count = st.number_input("Min ratings", 0, 5000, 50, step=50, key="browse_min")

        filtered = movies[movies["rating_count"] >= min_count].copy()
        if genre_browse != "All":
            filtered = filtered[filtered["genres"].str.contains(genre_browse, na=False)]

        if sort_by == "Rating Count":
            filtered = filtered.sort_values("rating_count", ascending=False)
        elif sort_by == "Average Rating":
            filtered = filtered.sort_values("rating_mean", ascending=False)
        else:
            filtered = filtered.sort_values("year", ascending=False)

        filtered = filtered.head(30)

        cols_per_row = 5
        for row_start in range(0, len(filtered), cols_per_row):
            cols = st.columns(cols_per_row)
            chunk = filtered.iloc[row_start:row_start+cols_per_row]
            for i, (_, mr) in enumerate(chunk.iterrows()):
                with cols[i]:
                    render_poster(mr.get("tmdbId", 0), str(mr.get("title", "")))
                    render_movie_card(mr.to_dict(), show_poster=False)

elif nav == "User Profile":
    st.markdown("""
    <div class="page-header">
      <div class="page-header-title">User Profile</div>
      <div class="accent-line"></div>
      <div class="page-header-sub">Rating History & Taste Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    profile_user = st.selectbox("Select User", ALL_USERS, key="profile_user")
    user_r = ratings[ratings["userId"] == profile_user].copy()
    user_r = user_r.merge(movies[["movieId","title","genres","year","tmdbId","rating_count","rating_mean"]], on="movieId", how="left")
    user_r = user_r.sort_values("rating", ascending=False)

    if user_r.empty:
        st.info("No ratings found for this user.")
    else:
        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Ratings",   f"{len(user_r):,}")
        c2.metric("Average Rating",  f"{user_r['rating'].mean():.2f}")
        c3.metric("Movies Rated",    f"{user_r['movieId'].nunique():,}")
        c4.metric("Rating Span",     f"{user_r['year_r'].min()} – {user_r['year_r'].max()}" if 'year_r' in user_r.columns else "—")

        st.markdown('<hr>', unsafe_allow_html=True)

        # Genre taste analysis
        genre_counts = pd.Series(
            [g for glist in user_r["genres"].dropna().str.split("|") for g in glist if g != "(no genres listed)"]
        ).value_counts()

        col_genres, col_ratings = st.columns([1, 2])

        with col_genres:
            st.markdown('<div class="section-label">Genre Distribution</div>', unsafe_allow_html=True)
            for genre, count in genre_counts.head(12).items():
                pct = count / genre_counts.sum()
                st.markdown(f"""
                <div style="margin-bottom:0.6rem;">
                  <div style="display:flex; justify-content:space-between; font-size:0.78rem;
                              color:var(--text-muted); margin-bottom:3px;">
                    <span>{genre}</span><span style="color:var(--green-mid)">{count}</span>
                  </div>
                  <div style="background:var(--bg-raised); height:3px; border-radius:2px;">
                    <div style="width:{pct*100:.1f}%; height:3px; background:linear-gradient(90deg,var(--green-mid),var(--green-glow)); border-radius:2px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col_ratings:
            st.markdown('<div class="section-label">Rating Distribution</div>', unsafe_allow_html=True)
            rating_dist = user_r["rating"].value_counts().sort_index()
            for rv, cnt in rating_dist.items():
                pct = cnt / len(user_r)
                st.markdown(f"""
                <div style="margin-bottom:0.6rem;">
                  <div style="display:flex; justify-content:space-between; font-size:0.78rem;
                              color:var(--text-muted); margin-bottom:3px;">
                    <span>{"&#9733;" * int(rv)} {rv}</span><span style="color:var(--green-mid)">{cnt}</span>
                  </div>
                  <div style="background:var(--bg-raised); height:3px; border-radius:2px;">
                    <div style="width:{pct*100:.1f}%; height:3px; background:linear-gradient(90deg,var(--green-dim),var(--green-bright)); border-radius:2px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)

        # Highest-rated movies
        st.markdown('<div class="section-label">Highest Rated by This User</div>', unsafe_allow_html=True)
        top_rated = user_r[user_r["rating"] >= 4.5].head(10)

        if not top_rated.empty:
            cols_per_row = 5
            for row_start in range(0, len(top_rated), cols_per_row):
                cols = st.columns(cols_per_row)
                chunk = top_rated.iloc[row_start:row_start+cols_per_row]
                for i, (_, mr) in enumerate(chunk.iterrows()):
                    with cols[i]:
                        render_poster(mr.get("tmdbId", 0), str(mr.get("title", "")))
                        render_movie_card(
                            mr.to_dict(),
                            score=mr["rating"] / 5.0,
                            source="rated",
                            show_poster=False
                        )
        else:
            st.markdown('<div class="info-box">No movies rated 4.5 or above.</div>', unsafe_allow_html=True)

        # Full history table
        with st.expander("Full Rating History"):
            show_df = user_r[["title","genres","year","rating","rating_mean","rating_count"]].reset_index(drop=True)
            st.dataframe(show_df, use_container_width=True, height=400)


elif nav == "System Analytics":
    st.markdown("""
    <div class="page-header">
      <div class="page-header-title">System Analytics</div>
      <div class="accent-line"></div>
      <div class="page-header-sub">Model Performance & Evaluation Metrics</div>
    </div>
    """, unsafe_allow_html=True)

    # SVD Metrics
    st.markdown('<div class="section-label">Collaborative Filtering — SVD Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Baseline RMSE", eval_metrics.get("baseline_rmse", "—"))
    c2.metric("Final RMSE",    eval_metrics.get("final_rmse",    "—"))
    c3.metric("Final MAE",     eval_metrics.get("final_mae",     "—"))
    c4.metric("CV RMSE (5-fold)", f"{eval_metrics.get('cv_rmse_mean','—')} ± {eval_metrics.get('cv_rmse_std','—')}")
    c5.metric("CV MAE (5-fold)",  f"{eval_metrics.get('cv_mae_mean','—')} ± {eval_metrics.get('cv_mae_std','—')}")

    st.markdown('<hr>', unsafe_allow_html=True)

    # Hybrid Metrics
    st.markdown('<div class="section-label">Hybrid Engine — Top-N Ranking Metrics (Leave-One-Out)</div>', unsafe_allow_html=True)
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    h1.metric("Best Alpha",   hybrid_metrics.get("best_alpha", best_alpha))
    h2.metric("Precision@5",  hybrid_metrics.get("P@5",  "—"))
    h3.metric("Recall@5",     hybrid_metrics.get("R@5",  "—"))
    h4.metric("F1@5",         hybrid_metrics.get("F1@5", "—"))
    h5.metric("Precision@10", hybrid_metrics.get("P@10", "—"))
    h6.metric("Recall@10",    hybrid_metrics.get("R@10", "—"))

    st.markdown('<hr>', unsafe_allow_html=True)

    # Alpha comparison
    alpha_scores = hyb.get("alpha_scores", {})
    if alpha_scores:
        st.markdown('<div class="section-label">Alpha Tuning — Precision@10 vs CF Weight</div>', unsafe_allow_html=True)
        alpha_df = pd.DataFrame(
            [(a, p) for a, p in alpha_scores.items()],
            columns=["Alpha (CF Weight)", "Precision@10"]
        ).sort_values("Alpha (CF Weight)")

        # Manual bar chart
        max_p = max(alpha_scores.values()) if alpha_scores else 1
        for _, row in alpha_df.iterrows():
            a_val = row["Alpha (CF Weight)"]
            p_val = row["Precision@10"]
            bar_w = int((p_val / max_p) * 100) if max_p > 0 else 0
            is_best = (a_val == best_alpha)
            bar_color = "var(--green-glow)" if is_best else "var(--green-dim)"
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:8px; font-family:'Cairo',sans-serif;">
              <div style="width:60px; font-size:0.78rem; color:{'var(--green-glow)' if is_best else 'var(--text-muted)'}; font-weight:{'700' if is_best else '400'};">
                α = {a_val}
              </div>
              <div style="flex:1; background:var(--bg-raised); height:18px; border-radius:2px; margin:0 12px; position:relative;">
                <div style="width:{bar_w}%; height:18px; background:{bar_color}; border-radius:2px; transition:width 0.3s;"></div>
              </div>
              <div style="width:60px; font-size:0.78rem; color:var(--text-muted); text-align:right;">
                {p_val:.4f}
              </div>
              {'<span style="margin-left:8px; font-size:0.65rem; color:var(--green-glow); letter-spacing:0.1em; text-transform:uppercase;">Best</span>' if is_best else ''}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    # Architecture overview
    st.markdown('<div class="section-label">System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem;">
      <div class="movie-card">
        <div class="movie-card-body">
          <div style="font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase; color:var(--text-faint); margin-bottom:0.5rem;">1</div>
          <div style="font-family:'Playfair Display',serif; font-size:1rem; color:var(--text-primary); margin-bottom:0.5rem;">Preprocessing</div>
          <div style="font-size:0.78rem; color:var(--text-muted); line-height:1.6;">
            610 users · 9,742 movies · 100,836 ratings<br>
            Year extraction · Genre parsing · TF-IDF strings<br>
            Rating normalization [0,1] · MultiLabelBinarizer
          </div>
        </div>
      </div>
      <div class="movie-card">
        <div class="movie-card-body">
          <div style="font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase; color:var(--text-faint); margin-bottom:0.5rem;">2</div>
          <div style="font-family:'Playfair Display',serif; font-size:1rem; color:var(--text-primary); margin-bottom:0.5rem;">Content-Based</div>
          <div style="font-size:0.78rem; color:var(--text-muted); line-height:1.6;">
            TF-IDF (unigram + bigram) · 1,502 tokens<br>
            Genres + tags signal · On-demand cosine similarity<br>
            User profile = weighted avg of liked movie vectors
          </div>
        </div>
      </div>
      <div class="movie-card">
        <div class="movie-card-body">
          <div style="font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase; color:var(--text-faint); margin-bottom:0.5rem;">3</div>
          <div style="font-family:'Playfair Display',serif; font-size:1rem; color:var(--text-primary); margin-bottom:0.5rem;">Collaborative</div>
          <div style="font-size:0.78rem; color:var(--text-muted); line-height:1.6;">
            Surprise SVD · 100 latent factors · 25 epochs<br>
            GridSearchCV hyperparameter sweep<br>
            Production model trained on all 100K ratings
          </div>
        </div>
      </div>
    </div>
    <div style="margin-top:1rem; display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
      <div class="movie-card">
        <div class="movie-card-body">
          <div style="font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase; color:var(--text-faint); margin-bottom:0.5rem;">4</div>
          <div style="font-family:'Playfair Display',serif; font-size:1rem; color:var(--text-primary); margin-bottom:0.5rem;">Hybrid Engine</div>
          <div style="font-size:0.78rem; color:var(--text-muted); line-height:1.6;">
            score = α × CF + (1-α) × CBF<br>
            Alpha tuned via Precision@10 sweep · Best α = 0.7<br>
            Cold-start fallback → popularity ranking<br>
            Leave-one-out evaluation protocol
          </div>
        </div>
      </div>
      <div class="movie-card">
        <div class="movie-card-body">
          <div style="font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase; color:var(--text-faint); margin-bottom:0.5rem;">5</div>
          <div style="font-family:'Playfair Display',serif; font-size:1rem; color:var(--text-primary); margin-bottom:0.5rem;">Streamlit UI</div>
          <div style="font-size:0.78rem; color:var(--text-muted); line-height:1.6;">
            TMDB API for posters, overview, cast, runtime<br>
            5 pages: Recs · Explorer · Profile · Analytics · Dataset<br>
            Interactive genre filter · alpha override slider
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


elif nav == "Dataset Overview":
    st.markdown("""
    <div class="page-header">
      <div class="page-header-title">Dataset Overview</div>
      <div class="accent-line"></div>
      <div class="page-header-sub">MovieLens Small — Exploratory Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    tab_d1, tab_d2, tab_d3 = st.tabs(["Ratings", "Movies", "Genres"])

    with tab_d1:
        st.markdown('<div class="section-label">Rating Statistics</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Ratings",     f"{len(ratings):,}")
        d2.metric("Unique Users",       f"{ratings['userId'].nunique():,}")
        d3.metric("Unique Movies Rated",f"{ratings['movieId'].nunique():,}")
        d4.metric("Global Mean Rating", f"{ratings['rating'].mean():.4f}")

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Ratings per Value</div>', unsafe_allow_html=True)
        dist = ratings["rating"].value_counts().sort_index()
        total = dist.sum()
        for rv, cnt in dist.items():
            pct = cnt / total * 100
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:7px; font-family:'Cairo',sans-serif;">
              <div style="width:40px; font-size:0.8rem; color:var(--text-muted);">{rv}</div>
              <div style="flex:1; background:var(--bg-raised); height:16px; border-radius:2px; margin:0 12px;">
                <div style="width:{pct:.1f}%; height:16px; background:linear-gradient(90deg,var(--green-dim),var(--green-bright)); border-radius:2px;"></div>
              </div>
              <div style="width:80px; font-size:0.78rem; color:var(--text-muted); text-align:right;">{cnt:,} ({pct:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Top 10 Most Active Users</div>', unsafe_allow_html=True)
        top_users = ratings.groupby("userId")["movieId"].count().sort_values(ascending=False).head(10)
        max_u = top_users.max()
        for uid, cnt in top_users.items():
            pct = cnt / max_u * 100
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:7px; font-family:'Cairo',sans-serif;">
              <div style="width:60px; font-size:0.78rem; color:var(--text-muted);">User {uid}</div>
              <div style="flex:1; background:var(--bg-raised); height:14px; border-radius:2px; margin:0 12px;">
                <div style="width:{pct:.1f}%; height:14px; background:var(--green-dim); border-radius:2px;"></div>
              </div>
              <div style="width:60px; font-size:0.78rem; color:var(--text-muted); text-align:right;">{cnt:,}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab_d2:
        st.markdown('<div class="section-label">Movie Statistics</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Movies",       f"{len(movies):,}")
        d2.metric("With Tags",          f"{(movies['tags_str'] != '').sum():,}")
        d3.metric("Cold Movies (<5 r)", f"{(movies['rating_count'] < 5).sum():,}")
        d4.metric("Year Range",         f"{int(movies['year'].min())} – {int(movies['year'].max())}")

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Movies by Decade</div>', unsafe_allow_html=True)
        movies["decade"] = (movies["year"] // 10 * 10).astype(str) + "s"
        decade_cnt = movies.groupby("decade").size().sort_index()
        max_dc = decade_cnt.max()
        for dc, cnt in decade_cnt.items():
            pct = cnt / max_dc * 100
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:7px; font-family:'Cairo',sans-serif;">
              <div style="width:55px; font-size:0.78rem; color:var(--text-muted);">{dc}</div>
              <div style="flex:1; background:var(--bg-raised); height:14px; border-radius:2px; margin:0 12px;">
                <div style="width:{pct:.1f}%; height:14px; background:linear-gradient(90deg,var(--green-dim),var(--green-mid)); border-radius:2px;"></div>
              </div>
              <div style="width:50px; font-size:0.78rem; color:var(--text-muted); text-align:right;">{cnt:,}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Top 15 Most Rated Movies</div>', unsafe_allow_html=True)
        top_movies = movies.sort_values("rating_count", ascending=False).head(15)
        for _, mr in top_movies.iterrows():
            st.markdown(f"""
            <div style="display:flex; align-items:center; padding:6px 0; border-bottom:1px solid var(--border);
                        font-family:'Cairo',sans-serif;">
              <div style="flex:1; font-size:0.82rem; color:var(--text-primary);">
                {str(mr['title']).rsplit('(',1)[0].strip()}
                <span style="color:var(--text-faint); font-size:0.72rem;"> ({int(mr.get('year',0))})</span>
              </div>
              <div style="font-size:0.75rem; color:var(--green-mid); margin-left:1rem;">{int(mr['rating_count']):,} ratings</div>
              <div style="font-size:0.75rem; color:var(--text-muted); margin-left:1rem; width:40px; text-align:right;">{mr['rating_mean']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab_d3:
        st.markdown('<div class="section-label">Genre Distribution</div>', unsafe_allow_html=True)
        all_genre_counts = pd.Series(
            [g for glist in movies["genres"].dropna().str.split("|")
             for g in glist if g != "(no genres listed)"]
        ).value_counts()

        max_gc = all_genre_counts.max()
        for g, cnt in all_genre_counts.items():
            pct = cnt / max_gc * 100
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:8px; font-family:'Cairo',sans-serif;">
              <div style="width:130px; font-size:0.8rem; color:var(--text-muted);">{g}</div>
              <div style="flex:1; background:var(--bg-raised); height:16px; border-radius:2px; margin:0 12px;">
                <div style="width:{pct:.1f}%; height:16px; background:linear-gradient(90deg,var(--green-dim),var(--green-glow)); border-radius:2px;"></div>
              </div>
              <div style="width:50px; font-size:0.78rem; color:var(--text-muted); text-align:right;">{cnt:,}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Average Rating by Genre</div>', unsafe_allow_html=True)

        genre_rating_rows = []
        for _, mr in movies.iterrows():
            genres_list = str(mr.get("genres","")).split("|")
            for g in genres_list:
                if g and g != "(no genres listed)":
                    genre_rating_rows.append({"genre": g, "rating_mean": mr["rating_mean"],
                                              "rating_count": mr["rating_count"]})

        genre_rt_df = pd.DataFrame(genre_rating_rows)
        genre_avg = (genre_rt_df[genre_rt_df["rating_count"] > 0]
                     .groupby("genre")["rating_mean"]
                     .mean()
                     .sort_values(ascending=False))

        max_gr = genre_avg.max()
        for g, avg_r in genre_avg.items():
            pct = avg_r / max_gr * 100
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:8px; font-family:'Cairo',sans-serif;">
              <div style="width:130px; font-size:0.8rem; color:var(--text-muted);">{g}</div>
              <div style="flex:1; background:var(--bg-raised); height:14px; border-radius:2px; margin:0 12px;">
                <div style="width:{pct:.1f}%; height:14px; background:linear-gradient(90deg,var(--gold),rgba(200,169,110,0.4)); border-radius:2px;"></div>
              </div>
              <div style="width:40px; font-size:0.78rem; color:var(--gold); text-align:right;">{avg_r:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
