"""
app.py — Job Recommendation System
Run with:  streamlit run app.py
Requires:  clean_job_dataset.csv in the same directory

Resume upload supports: PDF, DOCX, TXT
"""

import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# ── Optional resume parsers (graceful fallback if not installed) ───────────────
try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from docx import Document as DocxDocument
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Recommendation System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3347;
    }

    .job-card {
        background: linear-gradient(135deg, #1e2235 0%, #252a3d 100%);
        border: 1px solid #2e3347;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .job-card:hover { border-color: #5c6bc0; }
    .job-card .rank    { font-size: 11px; color: #7986cb; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
    .job-card .title   { font-size: 17px; font-weight: 700; color: #e8eaf6; margin: 4px 0 2px; }
    .job-card .company { font-size: 13px; color: #9fa8da; }

    .score-badge {
        display: inline-block;
        background: #283593;
        color: #c5cae9;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 8px;
    }

    .section-header {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #7986cb;
        text-transform: uppercase;
        margin-bottom: 14px;
        border-bottom: 1px solid #2e3347;
        padding-bottom: 6px;
    }

    .resume-upload-box {
        background: #1a1d27;
        border: 2px dashed #3949ab;
        border-radius: 14px;
        padding: 28px 24px;
        text-align: center;
        margin-bottom: 16px;
    }
    .resume-upload-box .upload-icon  { font-size: 36px; }
    .resume-upload-box .upload-title { font-size: 16px; font-weight: 700; color: #e8eaf6; margin: 8px 0 4px; }
    .resume-upload-box .upload-sub   { font-size: 12px; color: #6c7293; }

    .resume-preview {
        background: #1e2235;
        border: 1px solid #2e3347;
        border-radius: 10px;
        padding: 16px;
        max-height: 220px;
        overflow-y: auto;
        font-size: 13px;
        color: #9fa8da;
        line-height: 1.6;
        white-space: pre-wrap;
    }

    [data-testid="stMetric"] {
        background: #1e2235;
        border: 1px solid #2e3347;
        border-radius: 10px;
        padding: 14px 18px;
    }

    .stTabs [data-baseweb="tab-list"] { background: #1a1d27; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"]      { border-radius: 6px; color: #9fa8da; }
    .stTabs [aria-selected="true"]    { background: #283593 !important; color: #fff !important; }

    .stButton > button {
        background: linear-gradient(135deg, #3949ab, #5c6bc0);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        width: 100%;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #5c6bc0, #7986cb); }

    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    .stTextArea textarea, .stTextInput input {
        background: #1e2235 !important;
        border: 1px solid #2e3347 !important;
        color: #e8eaf6 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] {
        background: #1e2235;
        border: 1px dashed #3949ab;
        border-radius: 10px;
        padding: 10px;
    }

    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── NLTK bootstrap ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _nltk_setup():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

STOP_WORDS, LEMMATIZER = _nltk_setup()


# ── Text preprocessing ─────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        LEMMATIZER.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 1
    ]
    return " ".join(tokens)


# ── Resume text extraction ─────────────────────────────────────────────────────
def extract_text_from_resume(uploaded_file) -> str:
    """Extract raw text from a PDF, DOCX, or TXT uploaded file."""
    name = uploaded_file.name.lower()
    raw  = uploaded_file.read()

    if name.endswith(".pdf"):
        if not PDF_OK:
            st.error("📦 `pdfplumber` not installed. Run:  pip install pdfplumber")
            return ""
        parts = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n".join(parts)

    elif name.endswith(".docx"):
        if not DOCX_OK:
            st.error("📦 `python-docx` not installed. Run:  pip install python-docx")
            return ""
        doc = DocxDocument(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")

    else:
        st.error("Unsupported file type. Please upload PDF, DOCX, or TXT.")
        return ""


# ── Load & index job data ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📊 Loading job dataset & building TF-IDF index…")
def load_and_index():
    try:
        df = pd.read_csv("clean_job_dataset.csv")
    except FileNotFoundError:
        st.error("❌ `clean_job_dataset.csv` not found. Place it in the same folder as `app.py`.")
        st.stop()

    if "processed_text" not in df.columns:
        df["processed_text"] = df["Job Description"].apply(preprocess)

    df = df[df["processed_text"].str.strip().astype(bool)].reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        max_features=5_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf = vectorizer.fit_transform(df["processed_text"])
    return df, tfidf, vectorizer


# ── Phase 1: job-to-job similarity ────────────────────────────────────────────
def recommend_jobs(jobs_df, job_tfidf, job_index: int, top_n: int = 5) -> pd.DataFrame:
    query_vec = job_tfidf[job_index]
    scores    = linear_kernel(query_vec, job_tfidf).flatten()
    k         = min(top_n + 1, len(scores))
    part      = np.argpartition(-scores, k)[:k]
    top_i     = part[np.argsort(-scores[part])]
    top_i     = top_i[top_i != job_index][:top_n]

    cols   = [c for c in ["Job Title", "Company", "location", "Work Type", "Salary Range"] if c in jobs_df.columns]
    result = jobs_df.iloc[top_i][cols].copy()
    result["Similarity"] = scores[top_i].round(4)
    result.index = range(1, len(result) + 1)
    return result


# ── Phase 2: resume → job matching ────────────────────────────────────────────
def match_resume_to_jobs(jobs_df, job_tfidf, vectorizer, resume_text: str, top_n: int = 10) -> pd.DataFrame:
    processed = preprocess(resume_text)
    if not processed:
        return pd.DataFrame()
    resume_vec = vectorizer.transform([processed])
    scores     = cosine_similarity(resume_vec, job_tfidf).flatten()
    k          = min(top_n, len(scores))
    part       = np.argpartition(-scores, k)[:k]
    top_i      = part[np.argsort(-scores[part])]

    cols   = [c for c in ["Job Title", "Company", "location", "Work Type", "Salary Range", "Experience"] if c in jobs_df.columns]
    result = jobs_df.iloc[top_i][cols].copy()
    result["Match Score"] = scores[top_i].round(4)
    result.index = range(1, len(result) + 1)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💼 Job Recommender")
    st.markdown("---")
    st.markdown("#### ⚙️ Settings")
    top_n = st.slider("Results to show", min_value=3, max_value=20, value=5)
    st.markdown("---")
    st.markdown(
        "<small style='color:#4a5080'>"
        "🔍 <b>Tab 1</b> — Job-to-Job similarity<br>"
        "📄 <b>Tab 2</b> — Upload resume → matched jobs<br>"
        "📊 <b>Tab 3</b> — Dataset explorer"
        "</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#4a5080'>"
        "Supported resume formats:<br>"
        "📕 PDF &nbsp;|&nbsp; 📘 DOCX &nbsp;|&nbsp; 📄 TXT"
        "</small>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 💼 Job Recommendation System")
st.markdown("Explore similar job postings or **upload your resume** to find the best-matching roles.")

jobs_df, job_tfidf, vectorizer = load_and_index()

# KPI strip
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Jobs", f"{len(jobs_df):,}")
with c2:
    roles = jobs_df["Role"].nunique() if "Role" in jobs_df.columns else jobs_df["Job Title"].nunique()
    st.metric("Unique Roles", f"{roles:,}")
with c3:
    companies = jobs_df["Company"].nunique() if "Company" in jobs_df.columns else 0
    st.metric("Companies", f"{companies:,}")
with c4:
    st.metric("TF-IDF Features", f"{job_tfidf.shape[1]:,}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Job-to-Job Similarity", "📄 Resume Matching", "📊 Dataset Explorer"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 1 — Job-to-Job
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("### Find similar jobs to any posting")

    title_col   = "Job Title" if "Job Title" in jobs_df.columns else jobs_df.columns[0]
    company_col = "Company"   if "Company"   in jobs_df.columns else None

    search_query = st.text_input("🔎 Search job title", placeholder="e.g. Data Scientist")
    mask     = jobs_df[title_col].str.contains(search_query, case=False, na=False) if search_query else pd.Series([True] * len(jobs_df))
    filtered = jobs_df[mask].head(50)

    if filtered.empty:
        st.warning("No jobs match that search. Try a different keyword.")
    else:
        label_series      = (filtered[title_col] + (" @ " + filtered[company_col] if company_col else "")).reset_index(drop=True)
        label_to_orig_idx = dict(zip(label_series, filtered.index.tolist()))
        selected_label    = st.selectbox("Select seed job", label_series.tolist(), label_visibility="collapsed")
        seed_idx          = label_to_orig_idx[selected_label]

        if st.button("🚀 Find Similar Jobs", key="btn_p1"):
            with st.spinner("Computing similarity…"):
                results = recommend_jobs(jobs_df, job_tfidf, seed_idx, top_n=top_n)

            st.markdown(f"<div class='section-header'>Top {top_n} similar jobs</div>", unsafe_allow_html=True)
            for i, row in results.iterrows():
                score_pct  = int(row["Similarity"] * 100)
                meta_parts = [str(row.get(c, "")) for c in ["Work Type", "location", "Salary Range"]
                              if str(row.get(c, "")) not in ("", "nan")]
                meta = "  ·  ".join(meta_parts)
                st.markdown(f"""
                <div class='job-card'>
                    <div class='rank'>#{i}</div>
                    <div class='title'>{row[title_col]}</div>
                    <div class='company'>{row.get('Company','')}</div>
                    {"<div style='font-size:12px;color:#7986cb;margin-top:4px'>" + meta + "</div>" if meta else ""}
                    <div class='score-badge'>Match {score_pct}%</div>
                </div>""", unsafe_allow_html=True)

            with st.expander("View as table"):
                st.dataframe(results, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 2 — Resume Matching (FILE UPLOAD)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("### Upload your resume to find matching jobs")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("""
        <div class='resume-upload-box'>
            <div class='upload-icon'>📄</div>
            <div class='upload-title'>Drop your resume here</div>
            <div class='upload-sub'>Supports PDF · DOCX · TXT &nbsp;(max 10 MB)</div>
        </div>
        """, unsafe_allow_html=True)

        resume_file = st.file_uploader(
            "Upload resume",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
            key="resume_uploader",
        )

        with st.expander("✏️ Or paste your resume text manually"):
            manual_text = st.text_area(
                "Resume text",
                height=200,
                placeholder=(
                    "e.g. Python developer with 5 years of experience in machine learning, "
                    "NLP, and data pipeline engineering. Proficient in PyTorch, scikit-learn, "
                    "SQL, and AWS. Strong background in model deployment and A/B testing."
                ),
                key="manual_resume",
            )

        run_match = st.button("🎯 Find Matching Jobs", key="btn_p2")

    with right:
        st.markdown("**💡 Quick demo examples**")
        examples = {
            "🤖 ML Engineer":  "Machine learning engineer with Python, TensorFlow, deep learning, model deployment, AWS SageMaker, MLflow, feature engineering",
            "🎨 Frontend Dev": "Frontend developer skilled in React, TypeScript, CSS, Figma, responsive design, Next.js, web accessibility",
            "📊 Data Analyst": "Data analyst with Excel, SQL, Power BI, Tableau, statistical analysis, Python, reporting and business dashboards",
            "☕ Backend Dev":  "Java backend developer with Spring Boot, REST APIs, PostgreSQL, Docker, Kubernetes, microservices architecture",
            "🔒 DevOps":       "DevOps engineer with CI/CD pipelines, Jenkins, GitHub Actions, AWS, Terraform, Linux administration and monitoring",
        }
        for label, text in examples.items():
            if st.button(label, key=f"ex_{label}"):
                # Store example in session state so it survives the rerun
                st.session_state["example_text"] = text

    # ── Resolve resume text ──
    resume_text = ""

    # 1. Uploaded file takes priority
    if resume_file is not None:
        with st.spinner(f"Reading {resume_file.name}…"):
            resume_text = extract_text_from_resume(resume_file)

        if resume_text.strip():
            st.success(f"✅ Extracted **{len(resume_text.split()):,} words** from `{resume_file.name}`")
            with st.expander("👁️ Preview extracted text"):
                preview = resume_text[:3000] + ("…" if len(resume_text) > 3000 else "")
                st.markdown(f"<div class='resume-preview'>{preview}</div>", unsafe_allow_html=True)
        else:
            st.warning("Could not extract text from the file. Try pasting manually.")

    # 2. Manual text if no file
    if not resume_text.strip():
        manual = st.session_state.get("example_text", "") or st.session_state.get("manual_resume", "")
        resume_text = manual

    # ── Run matching ──
    if run_match:
        if not resume_text.strip():
            st.warning("⚠️ Please upload a resume file or paste your resume text first.")
        else:
            with st.spinner("🔍 Scanning all job postings…"):
                results = match_resume_to_jobs(jobs_df, job_tfidf, vectorizer, resume_text, top_n=top_n)

            if results.empty:
                st.warning("Could not extract meaningful tokens. Try a more detailed resume.")
            else:
                st.markdown("---")
                st.markdown(
                    f"<div class='section-header'>Top {top_n} matching jobs for your resume</div>",
                    unsafe_allow_html=True,
                )

                for i, row in results.iterrows():
                    score_pct  = int(row["Match Score"] * 100)
                    meta_parts = [str(row.get(c, "")) for c in ["Experience", "Work Type", "location", "Salary Range"]
                                  if str(row.get(c, "")) not in ("", "nan")]
                    meta = "  ·  ".join(meta_parts)

                    if score_pct >= 60:
                        badge_bg, badge_fg = "#1b5e20", "#a5d6a7"
                    elif score_pct >= 30:
                        badge_bg, badge_fg = "#e65100", "#ffe0b2"
                    else:
                        badge_bg, badge_fg = "#283593", "#c5cae9"

                    st.markdown(f"""
                    <div class='job-card'>
                        <div class='rank'>#{i}</div>
                        <div class='title'>{row.get('Job Title','')}</div>
                        <div class='company'>{row.get('Company','')}</div>
                        {"<div style='font-size:12px;color:#7986cb;margin-top:4px'>" + meta + "</div>" if meta else ""}
                        <span style='display:inline-block;background:{badge_bg};color:{badge_fg};
                                     font-size:12px;font-weight:600;padding:3px 10px;
                                     border-radius:20px;margin-top:8px;'>
                            Match {score_pct}%
                        </span>
                    </div>""", unsafe_allow_html=True)

                with st.expander("📋 View as table"):
                    st.dataframe(results, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 3 — Dataset Explorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("### Explore the job dataset")

    c1, c2, c3 = st.columns(3)
    with c1:
        wt_filter = st.multiselect("Work Type", jobs_df["Work Type"].dropna().unique().tolist()) if "Work Type" in jobs_df.columns else []
    with c2:
        country_filter = st.multiselect("Country", sorted(jobs_df["Country"].dropna().unique().tolist())) if "Country" in jobs_df.columns else []
    with c3:
        title_search = st.text_input("Job title contains", placeholder="e.g. Engineer")

    view_df = jobs_df.copy()
    if wt_filter:
        view_df = view_df[view_df["Work Type"].isin(wt_filter)]
    if country_filter:
        view_df = view_df[view_df["Country"].isin(country_filter)]
    if title_search:
        view_df = view_df[view_df["Job Title"].str.contains(title_search, case=False, na=False)]

    st.markdown(f"**{len(view_df):,} rows** match the current filters.")
    show_cols = [c for c in ["Job Title", "Company", "location", "Country", "Work Type",
                             "Salary Range", "Experience", "Qualifications"] if c in view_df.columns]
    st.dataframe(view_df[show_cols].head(500), use_container_width=True, height=420)

    if len(view_df) > 0:
        st.markdown("---")
        ch1, ch2 = st.columns(2)
        with ch1:
            if "Work Type" in view_df.columns:
                st.markdown("**Work Type distribution**")
                st.bar_chart(view_df["Work Type"].value_counts().head(10))
        with ch2:
            if "Qualifications" in view_df.columns:
                st.markdown("**Qualification distribution**")
                st.bar_chart(view_df["Qualifications"].value_counts().head(10))
