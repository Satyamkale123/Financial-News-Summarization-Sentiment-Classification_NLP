import streamlit as st
import torch
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import time

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight NLP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e8eaf0;
}
.stApp { background-color: #0a0e1a; }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    background: linear-gradient(135deg, #00d4ff 0%, #7b61ff 50%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    line-height: 1.1;
}
.hero-sub {
    color: #8892a4;
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.team-badge {
    display: inline-block;
    background: rgba(123,97,255,0.12);
    border: 1px solid rgba(123,97,255,0.3);
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.82rem;
    color: #a89fff;
    margin-top: 1rem;
    letter-spacing: 0.05em;
}

/* Cards */
.card {
    background: #131929;
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #c8d0e0;
    margin-bottom: 0.3rem;
}
.card-sub {
    font-size: 0.8rem;
    color: #5a6a80;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Sentiment badge */
.badge-positive {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,210,100,0.12);
    border: 1px solid rgba(0,210,100,0.35);
    color: #00d264;
    border-radius: 30px;
    padding: 0.4rem 1.1rem;
    font-size: 1rem;
    font-weight: 600;
}
.badge-negative {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,80,80,0.12);
    border: 1px solid rgba(255,80,80,0.35);
    color: #ff5050;
    border-radius: 30px;
    padding: 0.4rem 1.1rem;
    font-size: 1rem;
    font-weight: 600;
}
.badge-neutral {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(150,160,180,0.12);
    border: 1px solid rgba(150,160,180,0.35);
    color: #96a0b4;
    border-radius: 30px;
    padding: 0.4rem 1.1rem;
    font-size: 1rem;
    font-weight: 600;
}

/* Confidence bar */
.conf-bar-bg {
    background: #1e2d45;
    border-radius: 8px;
    height: 8px;
    margin-top: 0.8rem;
    overflow: hidden;
}
.conf-bar-fill-pos { background: linear-gradient(90deg,#00d264,#00ffaa); height:8px; border-radius:8px; transition: width 1s ease; }
.conf-bar-fill-neg { background: linear-gradient(90deg,#ff5050,#ff9f43); height:8px; border-radius:8px; transition: width 1s ease; }
.conf-bar-fill-neu { background: linear-gradient(90deg,#96a0b4,#c8d0e0); height:8px; border-radius:8px; transition: width 1s ease; }

/* Summary box */
.summary-box {
    background: #0d1520;
    border-left: 3px solid #7b61ff;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #c0ccde;
    margin-top: 0.5rem;
    font-style: italic;
}

/* Metric pills */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-pill {
    background: #0d1520;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    text-align: center;
    flex: 1;
    min-width: 80px;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #00d4ff;
}
.metric-label {
    font-size: 0.72rem;
    color: #5a6a80;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e2d45;
    margin: 2rem 0;
}

/* Textarea */
.stTextArea textarea {
    background: #0d1520 !important;
    border: 1px solid #1e2d45 !important;
    color: #e8eaf0 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
}
.stTextArea textarea:focus {
    border-color: #7b61ff !important;
    box-shadow: 0 0 0 2px rgba(123,97,255,0.2) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7b61ff, #00d4ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Examples */
.example-chip {
    display: inline-block;
    background: #131929;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    font-size: 0.78rem;
    color: #8892a4;
    margin: 0.2rem;
    cursor: pointer;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #131929;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #5a6a80;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: #1e2d45 !important;
    color: #e8eaf0 !important;
}

/* Results appear */
.result-appear {
    animation: fadeUp 0.4s ease forwards;
}
@keyframes fadeUp {
    from { opacity:0; transform: translateY(12px); }
    to   { opacity:1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ─── Model loading (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    device = 0 if torch.cuda.is_available() else -1
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=device,
        truncation=True,
        max_length=512
    )
    t5_tok = T5Tokenizer.from_pretrained("t5-small")
    t5_mod = T5ForConditionalGeneration.from_pretrained("t5-small")
    if torch.cuda.is_available():
        t5_mod = t5_mod.cuda()
    t5_mod.eval()
    return finbert, t5_tok, t5_mod


def summarize(text, t5_tok, t5_mod):
    inp = t5_tok(
        "summarize: " + text.replace("\n", " "),
        return_tensors="pt", max_length=512, truncation=True
    )
    if torch.cuda.is_available():
        inp = {k: v.cuda() for k, v in inp.items()}
    with torch.no_grad():
        out = t5_mod.generate(
            **inp, max_length=80, min_length=25,
            num_beams=4, early_stopping=True, no_repeat_ngram_size=2
        )
    return t5_tok.decode(out[0], skip_special_tokens=True)


def sentiment_badge(label, score):
    pct = int(score * 100)
    if label.lower() == "positive":
        badge = f'<span class="badge-positive">📈 Positive</span>'
        bar   = f'<div class="conf-bar-bg"><div class="conf-bar-fill-pos" style="width:{pct}%"></div></div>'
    elif label.lower() == "negative":
        badge = f'<span class="badge-negative">📉 Negative</span>'
        bar   = f'<div class="conf-bar-bg"><div class="conf-bar-fill-neg" style="width:{pct}%"></div></div>'
    else:
        badge = f'<span class="badge-neutral">➡️ Neutral</span>'
        bar   = f'<div class="conf-bar-bg"><div class="conf-bar-fill-neu" style="width:{pct}%"></div></div>'
    return badge, bar, pct


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">FinSight NLP</div>
    <div class="hero-sub">Financial News Summarization & Sentiment Intelligence</div>
    <div class="team-badge">DATA 641 · NLP · Team 20 — Satyam Kale · Mayur Sangle · Adwait Gaur</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── Load models ─────────────────────────────────────────────────────────────
with st.spinner("Loading FinBERT & T5 models..."):
    finbert, t5_tok, t5_mod = load_models()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Article", "📰 Headline Screener", "📊 Model Performance"])

# ══════════════════════════════════════════════════════
# TAB 1 — Full Article Analysis
# ══════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown('<div class="card-title">Paste a Financial Article</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Full article text · earnings reports · press releases · news stories</div>', unsafe_allow_html=True)

        # Example buttons
        examples = {
            "Apple Earnings": "Apple Inc. reported fourth-quarter earnings that exceeded Wall Street expectations, driven by strong iPhone sales and record-breaking services revenue. The company reported earnings per share of $1.46, beating analyst estimates of $1.39. Revenue came in at $89.5 billion, up 8% year-over-year. Services revenue hit an all-time high of $22.3 billion, up 16% from the previous year. The company returned $25 billion to shareholders through buybacks and dividends during the quarter.",
            "Fed Rate Hike": "The Federal Reserve raised its benchmark interest rate by 25 basis points to a target range of 5.25% to 5.50%, the highest level in 22 years. Fed Chair Jerome Powell signaled that future hikes would depend on incoming economic data. Markets reacted with volatility, with the S&P 500 falling 1.2% on the announcement. Analysts warned that prolonged high rates could increase recession risks in 2024.",
            "Tesla Miss": "Tesla Inc. reported lower-than-expected vehicle deliveries for the third quarter, delivering 435,059 vehicles against analyst estimates of 470,000. The shortfall was attributed to planned factory shutdowns. Despite the delivery miss, Tesla maintained its full-year guidance of 1.8 million vehicles. Shares fell 5.7% in after-hours trading.",
            "Amazon AWS": "Amazon Web Services reported revenue of $23.8 billion, representing 17% year-over-year growth and beating consensus estimates. The cloud division's operating income reached $7.2 billion with margins expanding to 30.3%. CEO Andy Jassy emphasized continued investment in generative AI tools.",
        }

        ex_cols = st.columns(4)
        selected_example = None
        for i, (name, text) in enumerate(examples.items()):
            with ex_cols[i]:
                if st.button(name, key=f"ex_{i}"):
                    selected_example = text

        article_text = st.text_area(
            "",
            value=selected_example if selected_example else "",
            height=220,
            placeholder="Paste any financial news article here...",
            label_visibility="collapsed"
        )

        analyze_btn = st.button("⚡ Analyze Article", key="analyze")

    with col2:
        st.markdown('<div class="card-title">Analysis Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">T5 Summary · FinBERT Sentiment · Confidence</div>', unsafe_allow_html=True)

        if analyze_btn and article_text.strip():
            with st.spinner("Running pipeline..."):
                t0 = time.time()
                summary = summarize(article_text, t5_tok, t5_mod)
                sent_result = finbert(summary)[0]
                elapsed = time.time() - t0

            label = sent_result['label']
            score = sent_result['score']
            badge_html, bar_html, pct = sentiment_badge(label, score)

            st.markdown(f"""
            <div class="result-appear">
                <div style="margin-bottom:1rem;">
                    <div style="font-size:0.78rem;color:#5a6a80;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">Sentiment</div>
                    {badge_html}
                    <div style="font-size:0.82rem;color:#5a6a80;margin-top:0.5rem;">Confidence: <span style="color:#c0ccde;font-weight:600;">{pct}%</span></div>
                    {bar_html}
                </div>
                <div style="margin-top:1.5rem;">
                    <div style="font-size:0.78rem;color:#5a6a80;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">T5 Summary</div>
                    <div class="summary-box">{summary.capitalize()}</div>
                </div>
                <div style="margin-top:1.2rem;">
                    <div class="metric-row">
                        <div class="metric-pill">
                            <div class="metric-value">{pct}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-value">{len(article_text.split())}</div>
                            <div class="metric-label">Input Words</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-value">{len(summary.split())}</div>
                            <div class="metric-label">Summary Words</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-value">{elapsed:.1f}s</div>
                            <div class="metric-label">Runtime</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif analyze_btn:
            st.warning("Please paste an article first.")
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#2a3a55;">
                <div style="font-size:2.5rem;margin-bottom:1rem;">📋</div>
                <div style="font-size:0.9rem;">Paste an article and click Analyze</div>
                <div style="font-size:0.78rem;margin-top:0.5rem;">or pick an example above</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 2 — Headline Screener
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="card-title">Bulk Headline Sentiment Screener</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Enter one headline per line · FinBERT classifies each instantly</div>', unsafe_allow_html=True)

    default_headlines = """Nvidia stock surges 10% after blowout earnings beat expectations
Bank of America reports massive losses amid credit crisis
Oil prices remain stable as OPEC maintains output levels
Microsoft Azure cloud revenue grows 29% beating all forecasts
Fed signals possible recession warning, markets tumble
Goldman Sachs upgrades Tesla to buy with $300 price target
Crypto markets crash as SEC announces new regulations
Apple launches new AI features driving record app store revenue"""

    headlines_input = st.text_area(
        "",
        value=default_headlines,
        height=200,
        placeholder="One headline per line...",
        label_visibility="collapsed"
    )

    if st.button("📊 Screen All Headlines", key="screen"):
        headlines = [h.strip() for h in headlines_input.strip().split("\n") if h.strip()]
        if headlines:
            with st.spinner(f"Analyzing {len(headlines)} headlines..."):
                results = finbert(headlines)

            pos = sum(1 for r in results if r['label'].lower() == 'positive')
            neg = sum(1 for r in results if r['label'].lower() == 'negative')
            neu = sum(1 for r in results if r['label'].lower() == 'neutral')

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-pill"><div class="metric-value" style="color:#00d264">{pos}</div><div class="metric-label">Positive</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-pill"><div class="metric-value" style="color:#ff5050">{neg}</div><div class="metric-label">Negative</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-pill"><div class="metric-value" style="color:#96a0b4">{neu}</div><div class="metric-label">Neutral</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-pill"><div class="metric-value" style="color:#00d4ff">{len(headlines)}</div><div class="metric-label">Total</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            for headline, result in zip(headlines, results):
                label = result['label']
                score = result['score']
                badge_html, bar_html, pct = sentiment_badge(label, score)
                color = "#00d264" if label.lower()=="positive" else "#ff5050" if label.lower()=="negative" else "#96a0b4"
                st.markdown(f"""
                <div class="card result-appear" style="padding:1rem 1.2rem;margin-bottom:0.6rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
                        <div style="font-size:0.92rem;color:#c0ccde;flex:1;min-width:200px;">{headline}</div>
                        <div style="display:flex;align-items:center;gap:0.8rem;">
                            {badge_html}
                            <span style="font-size:0.82rem;color:{color};font-weight:600;">{pct}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="card-title">Experiment Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Financial PhraseBank · 98 samples (sentences_allagree) · 80/20 split</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-row">
        <div class="metric-pill">
            <div class="metric-value" style="color:#00d264">85%</div>
            <div class="metric-label">FinBERT Accuracy</div>
        </div>
        <div class="metric-pill">
            <div class="metric-value" style="color:#00d4ff">0.849</div>
            <div class="metric-label">FinBERT F1</div>
        </div>
        <div class="metric-pill">
            <div class="metric-value" style="color:#7b61ff">+117%</div>
            <div class="metric-label">vs Baseline F1</div>
        </div>
        <div class="metric-pill">
            <div class="metric-value" style="color:#ff9f43">0.205</div>
            <div class="metric-label">ROUGE-1 (T5)</div>
        </div>
        <div class="metric-pill">
            <div class="metric-value" style="color:#ff6b6b">95%</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Sentiment Classification</div>
            <div class="card-sub">Task 1 Results</div>
            <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
                <tr style="color:#5a6a80;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;">
                    <td style="padding:0.4rem 0;">Model</td>
                    <td style="padding:0.4rem 0;text-align:right;">Accuracy</td>
                    <td style="padding:0.4rem 0;text-align:right;">F1</td>
                </tr>
                <tr style="border-top:1px solid #1e2d45;color:#8892a4;">
                    <td style="padding:0.5rem 0;">Logistic Regression</td>
                    <td style="padding:0.5rem 0;text-align:right;">55.0%</td>
                    <td style="padding:0.5rem 0;text-align:right;">0.390</td>
                </tr>
                <tr style="border-top:1px solid #1e2d45;color:#8892a4;">
                    <td style="padding:0.5rem 0;">Naive Bayes</td>
                    <td style="padding:0.5rem 0;text-align:right;">55.0%</td>
                    <td style="padding:0.5rem 0;text-align:right;">0.390</td>
                </tr>
                <tr style="border-top:1px solid #1e2d45;color:#00d264;font-weight:600;">
                    <td style="padding:0.5rem 0;">FinBERT (Ours) ✓</td>
                    <td style="padding:0.5rem 0;text-align:right;">85.0%</td>
                    <td style="padding:0.5rem 0;text-align:right;">0.849</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Summarization — T5-small</div>
            <div class="card-sub">Task 2 ROUGE Scores</div>
            <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
                <tr style="color:#5a6a80;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;">
                    <td style="padding:0.4rem 0;">Metric</td>
                    <td style="padding:0.4rem 0;text-align:right;">Score</td>
                    <td style="padding:0.4rem 0;text-align:right;">Interpretation</td>
                </tr>
                <tr style="border-top:1px solid #1e2d45;color:#8892a4;">
                    <td style="padding:0.5rem 0;">ROUGE-1</td>
                    <td style="padding:0.5rem 0;text-align:right;">0.205</td>
                    <td style="padding:0.5rem 0;text-align:right;color:#7b61ff;">Unigram overlap</td>
                </tr>
                <tr style="border-top:1px solid #1e2d45;color:#8892a4;">
                    <td style="padding:0.5rem 0;">ROUGE-2</td>
                    <td style="padding:0.5rem 0;text-align:right;">0.015</td>
                    <td style="padding:0.5rem 0;text-align:right;color:#7b61ff;">Bigram overlap</td>
                </tr>
                <tr style="border-top:1px solid #1e2d45;color:#8892a4;">
                    <td style="padding:0.5rem 0;">ROUGE-L</td>
                    <td style="padding:0.5rem 0;text-align:right;">0.138</td>
                    <td style="padding:0.5rem 0;text-align:right;color:#7b61ff;">Longest match</td>
                </tr>
            </table>
            <div style="margin-top:0.8rem;font-size:0.78rem;color:#5a6a80;">
                Standard range for T5-small: ROUGE-1 0.18–0.28 ✓
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:0.5rem;">
        <div class="card-title">Pipeline — End-to-End Results</div>
        <div class="card-sub">Task 3 · Article → T5 Summary → FinBERT Sentiment</div>
        <div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.5rem;">
            <span class="badge-positive">📈 Apple Q4 · 95%</span>
            <span class="badge-negative">📉 Fed Hike · 97%</span>
            <span class="badge-negative">📉 Tesla Miss · 97%</span>
            <span class="badge-positive">📈 Amazon AWS · 96%</span>
            <span class="badge-positive">📈 JPMorgan · 96%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:0.5rem;">
        <div class="card-title">Architecture</div>
        <div class="card-sub">Models & Framework</div>
        <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:0.5rem;">
            <span style="background:#1e2d45;border-radius:6px;padding:0.3rem 0.7rem;font-size:0.8rem;color:#c0ccde;">ProsusAI/finbert</span>
            <span style="background:#1e2d45;border-radius:6px;padding:0.3rem 0.7rem;font-size:0.8rem;color:#c0ccde;">t5-small</span>
            <span style="background:#1e2d45;border-radius:6px;padding:0.3rem 0.7rem;font-size:0.8rem;color:#c0ccde;">HuggingFace Transformers</span>
            <span style="background:#1e2d45;border-radius:6px;padding:0.3rem 0.7rem;font-size:0.8rem;color:#c0ccde;">PyTorch</span>
            <span style="background:#1e2d45;border-radius:6px;padding:0.3rem 0.7rem;font-size:0.8rem;color:#c0ccde;">scikit-learn</span>
            <span style="background:#1e2d45;border-radius:6px;padding:0.3rem 0.7rem;font-size:0.8rem;color:#c0ccde;">Streamlit</span>
            <span style="background:#1e2d45;border-radius:6px;padding:0.3rem 0.7rem;font-size:0.8rem;color:#c0ccde;">Financial PhraseBank</span>
            <span style="background:#1e2d45;border-radius:6px;padding:0.3rem 0.7rem;font-size:0.8rem;color:#c0ccde;">Google Colab T4</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
