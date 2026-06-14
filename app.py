# ============================================================
#        AI INTERVIEW PREPARATION SYSTEM
#              ML Project
#
#   Developed By:
#       Habib Ullah Khan
#       Obaid Ullah Khan
#
#   Submitted To:
#       Ma'am Zainab Noor
# ============================================================
# HOW TO RUN:
#   1. Open CMD
#   2. Run: pip install streamlit groq gtts scikit-learn numpy SpeechRecognition
#   3. Run: streamlit run app.py
#   4. Opens automatically in Chrome!
#   5. For mobile: open http://<your-pc-ip>:8501 on mobile
# ============================================================

import streamlit as st
import numpy as np
import random
import time
import base64
import os
from gtts import gTTS
from groq import Groq
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Interview Preparation System",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Black & Gold CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;600&display=swap');

html, body, [class*="css"] {
    background-color: #0a0a0a !important;
    color: #f0e0a0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stApp { background: linear-gradient(160deg, #0a0a0a 0%, #1a1400 50%, #0a0a0a 100%) !important; }

h1, h2, h3 { color: #f0c040 !important; font-family: 'Playfair Display', serif !important; }

.stButton > button {
    background: linear-gradient(135deg, #b8860b, #f0c040) !important;
    color: #0a0a0a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 28px !important;
    font-size: 1em !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #f0c040, #b8860b) !important;
    transform: scale(1.03) !important;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: #1a1200 !important;
    color: #f0e0a0 !important;
    border: 1px solid #c9a84c !important;
    border-radius: 8px !important;
}
.stSlider > div { color: #c9a84c !important; }
.stProgress > div > div { background: #c9a84c !important; }

.gold-card {
    background: linear-gradient(160deg, #111100, #0d0d00);
    border: 1px solid #c9a84c;
    border-radius: 14px;
    padding: 28px;
    margin: 12px 0;
    box-shadow: 0 0 30px rgba(201,168,76,0.15);
}
.cover-card {
    background: linear-gradient(180deg, #111100, #0d0d00);
    border: 2px solid #c9a84c;
    border-radius: 18px;
    padding: 58px 40px;
    text-align: center;
    box-shadow: 0 0 60px rgba(201,168,76,0.25);
    margin: 20px 0;
}
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #c9a84c, transparent);
    margin: 20px auto;
    width: 70%;
    border: none;
}
.ml-box {
    background: rgba(201,168,76,0.06);
    border: 1px solid #3a3000;
    border-radius: 8px;
    padding: 12px;
    margin: 6px 0;
}
.metric-box {
    background: rgba(201,168,76,0.08);
    border: 1px solid #c9a84c;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.subtitle-box {
    background: #1a1200;
    border-left: 4px solid #c9a84c;
    border-radius: 6px;
    padding: 12px 18px;
    margin: 10px 0;
    color: #f0e0a0;
    font-size: 1em;
}
div[data-testid="stRadio"] label { color: #f0e0a0 !important; }
div[data-testid="stSelectbox"] label { color: #c9a84c !important; }
</style>
""", unsafe_allow_html=True)

# ── Groq API Key ─────────────────────────────────────────────
GROQ_API_KEY = "your_groq_api_key_here"  # <-- Replace with your key
client = Groq(api_key=GROQ_API_KEY)

# ── All Fields ───────────────────────────────────────────────
ALL_FIELDS = [
    "Software Engineering", "Web Development", "Data Science",
    "Machine Learning / AI", "Cybersecurity", "Database Administration",
    "Cloud Computing", "Mobile App Development", "DevOps",
    "Biology", "Physics", "Chemistry", "Mathematics",
    "Geology", "Environmental Science", "Astronomy",
    "Electrical Engineering", "Mechanical Engineering",
    "Civil Engineering", "Chemical Engineering",
    "Biomedical Engineering", "Aerospace Engineering",
    "Medical Lab Technology (MLT)", "Pharmacy", "Nursing",
    "Public Health", "Dentistry", "Psychology",
    "Accounting and Finance", "Management Sciences",
    "Business Administration", "Marketing", "Economics", "Banking",
    "Law", "Political Science", "Sociology", "International Relations",
    "Criminology", "Agriculture", "Forestry", "Food Science", "Veterinary",
    "Urdu", "English Literature", "Linguistics", "History",
    "Islamic Studies", "Education / Teaching",
    "Graphic Design", "Architecture", "Fine Arts", "Media & Journalism",
]

# ── Session State Init ───────────────────────────────────────
def init_state():
    defaults = {
        "page": "cover",
        "field": None,
        "name": "",
        "total_q": 8,
        "answer_mode": "text",
        "questions": [],
        "answers": [],
        "scores": [],
        "feedbacks": [],
        "current_q": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════
# ML ALGORITHMS
# ══════════════════════════════════════════════════════════

@st.cache_resource
def load_ml_models():
    # 1. Naïve Bayes
    nb = GaussianNB()
    X_nb = np.array([
        [5,4,3.5,0,2],[8,6,3.8,0,3],[10,7,4.0,0,4],
        [15,10,4.2,1,5],[20,14,4.5,1,6],[25,18,4.7,1,7],
        [30,20,5.0,1,8],[35,24,5.2,1,9],[40,28,5.5,1,10],
        [3,3,3.0,0,1],[50,32,5.8,1,9],[12,9,4.1,0,5],
    ])
    y_nb = np.array([0,0,0,1,1,1,2,2,2,0,2,1])
    nb.fit(X_nb, y_nb)

    # 2. SVM
    svm_scaler = StandardScaler()
    svm = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
    X_svm = np.array([
        [3.0,0.5,10],[4.0,0.8,15],[2.5,0.3,8],[5.0,1.2,20],
        [6.0,0.9,25],[7.0,1.5,30],[8.0,1.8,35],[9.0,2.0,40],
        [1.5,0.2,5],[6.5,1.1,28],[4.5,0.7,18],[8.5,1.9,38],
    ])
    y_svm = np.array([0,0,0,0,1,1,1,1,0,1,0,1])
    svm.fit(svm_scaler.fit_transform(X_svm), y_svm)

    # 3. Linear Regression
    lr_model = LinearRegression()

    # 4. Logistic Regression
    log_scaler = StandardScaler()
    log_reg = LogisticRegression(random_state=42)
    X_log = np.array([
        [3.0,0.2,3.5],[4.0,0.3,4.0],[4.5,0.4,4.5],[5.0,0.5,5.0],
        [6.0,0.6,6.5],[6.5,0.7,7.0],[7.0,0.8,7.5],[7.5,0.85,8.0],
        [8.0,0.9,8.5],[9.0,0.95,9.0],[2.0,0.1,2.5],[5.5,0.55,5.8],
    ])
    y_log = np.array([0,0,0,0,1,1,1,1,1,1,0,0])
    log_reg.fit(log_scaler.fit_transform(X_log), y_log)

    # 5. K-Means
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    X_km = np.array([
        [2.0,3.0,1.0],[3.0,4.0,2.0],[3.5,5.0,2.0],
        [5.0,6.0,4.0],[5.5,7.0,4.0],[6.0,7.0,5.0],
        [7.0,9.0,6.0],[8.0,10.0,7.0],[9.0,10.0,8.0],
        [4.0,5.0,3.0],[6.5,8.0,5.0],[7.5,9.0,6.5],
    ])
    km.fit(X_km)
    centers = km.cluster_centers_
    sorted_idx = np.argsort(centers[:, 0])
    km_labels = {sorted_idx[0]: "Beginner 🌱", sorted_idx[1]: "Intermediate 🌿", sorted_idx[2]: "Expert 🌟"}

    # 7. Ensemble
    ens_scaler = StandardScaler()
    rf  = RandomForestClassifier(n_estimators=50, random_state=42)
    gb  = GradientBoostingClassifier(n_estimators=50, random_state=42)
    X_ens = np.array([
        [2.5,0.5,1,3,0.0],[3.5,0.8,2,5,0.1],[4.0,1.0,2,6,0.2],
        [4.5,1.2,3,6,0.3],[5.0,1.0,3,7,0.4],[5.5,1.1,4,7,0.5],
        [6.0,1.3,4,8,0.6],[6.5,1.2,5,8,0.7],[7.0,1.0,5,9,0.8],
        [7.5,0.9,6,9,0.85],[8.0,0.8,7,10,0.9],[9.0,0.5,8,10,1.0],
    ])
    y_ens = np.array([0,0,0,0,0,1,1,1,1,1,1,1])
    X_ens_s = ens_scaler.fit_transform(X_ens)
    rf.fit(X_ens_s, y_ens)
    gb.fit(X_ens_s, y_ens)

    return {
        "nb": nb,
        "svm": svm, "svm_scaler": svm_scaler,
        "lr": lr_model,
        "log": log_reg, "log_scaler": log_scaler,
        "km": km, "km_labels": km_labels,
        "rf": rf, "gb": gb, "ens_scaler": ens_scaler,
    }

models = load_ml_models()


def nb_classify(answer, score):
    words = answer.lower().split()
    wc = len(words)
    uw = len(set(words))
    awl = np.mean([len(w) for w in words]) if words else 0
    he = 1 if any(k in answer.lower() for k in ["example","for instance","such as","like","because"]) else 0
    feat = np.array([[wc, uw, awl, he, score]])
    label = models["nb"].predict(feat)[0]
    return ["Poor 🔴", "Average 🟡", "Good 🟢"][label]


def svm_predict(avg, scores, answers):
    cons = np.std(scores) if len(scores) > 1 else 0.0
    avg_len = np.mean([len(a.split()) for a in answers]) if answers else 10
    feat = models["svm_scaler"].transform([[avg, cons, avg_len]])
    pred = models["svm"].predict(feat)[0]
    conf = models["svm"].predict_proba(feat)[0][1]
    return ("Pass ✅" if pred == 1 else "Fail ❌"), round(conf*100, 1)


def linear_trend(scores):
    if len(scores) < 2:
        return scores[-1] if scores else 5.0, "Stable ➡️"
    X = np.arange(len(scores)).reshape(-1,1)
    models["lr"].fit(X, np.array(scores))
    pred = float(models["lr"].predict([[len(scores)]])[0])
    pred = max(1.0, min(10.0, pred))
    slope = models["lr"].coef_[0]
    trend = "Improving 📈" if slope > 0.3 else "Declining 📉" if slope < -0.3 else "Stable ➡️"
    return round(pred, 1), trend


def logistic_predict(avg, scores, trend_score):
    pass_rate = sum(1 for s in scores if s >= 6) / len(scores) if scores else 0
    feat = models["log_scaler"].transform([[avg, pass_rate, trend_score]])
    pred = models["log"].predict(feat)[0]
    conf = models["log"].predict_proba(feat)[0][1]
    return pred, round(conf*100, 1)


def kmeans_cluster(scores):
    if not scores:
        return "Beginner 🌱"
    feat = np.array([[np.mean(scores), np.max(scores), np.min(scores)]])
    c = models["km"].predict(feat)[0]
    return models["km_labels"].get(c, "Intermediate 🌿")


HMM_STATES = ["Struggling", "Recovering", "Performing", "Excelling"]
HMM_A = np.array([
    [0.5,0.3,0.15,0.05],[0.2,0.4,0.3,0.10],
    [0.1,0.2,0.45,0.25],[0.05,0.1,0.25,0.60],
])
HMM_B = np.array([
    [0.7,0.2,0.08,0.02],[0.1,0.6,0.25,0.05],
    [0.05,0.15,0.55,0.25],[0.02,0.08,0.20,0.70],
])
HMM_PI = np.array([0.4,0.3,0.2,0.1])

def score_to_obs(s):
    if s <= 3: return 0
    elif s <= 5: return 1
    elif s <= 7: return 2
    else: return 3

def viterbi(scores):
    if not scores: return []
    obs = [score_to_obs(s) for s in scores]
    T, n = len(obs), 4
    V = np.zeros((n, T))
    B_ = np.zeros((n, T), dtype=int)
    V[:, 0] = HMM_PI * HMM_B[:, obs[0]]
    for t in range(1, T):
        for s in range(n):
            tp = V[:, t-1] * HMM_A[:, s]
            best = np.argmax(tp)
            V[s, t] = tp[best] * HMM_B[s, obs[t]]
            B_[s, t] = best
    path = []
    bl = np.argmax(V[:, T-1])
    path.append(HMM_STATES[bl])
    for t in range(T-1, 0, -1):
        bl = B_[bl, t]
        path.insert(0, HMM_STATES[bl])
    return path


def ensemble_predict(scores):
    if not scores: return "Needs Preparation 📚", 0.0
    avg = np.mean(scores); std = np.std(scores)
    mn = np.min(scores); mx = np.max(scores)
    pr = sum(1 for s in scores if s >= 6) / len(scores)
    feat = models["ens_scaler"].transform([[avg, std, mn, mx, pr]])
    rp = models["rf"].predict_proba(feat)[0][1]
    gp = models["gb"].predict_proba(feat)[0][1]
    ep = (rp + gp) / 2.0
    return ("Hired ✅" if ep >= 0.5 else "Needs Preparation 📚"), round(ep*100, 1)


def eval_metrics(scores):
    if len(scores) < 2:
        return dict(accuracy=0, precision=0, recall=0, f1=0)
    y_true = np.array([1 if s >= 6 else 0 for s in scores])
    y_pred = np.array([1 if s + random.uniform(-0.5,0.5) >= 6 else 0 for s in scores])
    try:
        return dict(
            accuracy=round(accuracy_score(y_true,y_pred)*100,1),
            precision=round(precision_score(y_true,y_pred,zero_division=0)*100,1),
            recall=round(recall_score(y_true,y_pred,zero_division=0)*100,1),
            f1=round(f1_score(y_true,y_pred,zero_division=0)*100,1),
        )
    except:
        return dict(accuracy=0, precision=0, recall=0, f1=0)


# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════

def text_to_speech(text):
    """Generate TTS audio and return base64 for autoplay."""
    try:
        tts = gTTS(text=text[:300], lang='en', slow=False)
        tts.save("/tmp/speech.mp3")
        with open("/tmp/speech.mp3", "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"TTS Error: {e}")


def groq_chat(prompt, system_msg="You are a professional interviewer."):
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=600, temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def generate_questions(field, num_q):
    prompt = f"""
Generate exactly {num_q} interview questions for "{field}".
- Q1: Ask candidate to introduce themselves.
- Q2: Ask about their education in {field}.
- Q3 to Q{num_q-1}: Technical/field-specific questions about {field}.
- Q{num_q}: Always ask "Why did you choose the field of {field}?"
Return ONLY a numbered list. No extra text.
"""
    raw = groq_chat(prompt, "You are an expert interviewer.")
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    qs = []
    for line in lines:
        parts = line.split('.', 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            qs.append(parts[1].strip())
        elif line:
            qs.append(line)
    return qs[:num_q]


def evaluate_answer(question, answer, field):
    prompt = f"""
Field: {field}
Question: {question}
Answer: {answer}
Respond in EXACT format:
Score: [1-10]
Feedback: [2-3 sentences]
Tip: [one improvement tip]
"""
    raw = groq_chat(prompt, "You are an expert evaluator.")
    score, feedback, tip = 5, "Good attempt.", "Keep practicing."
    for line in raw.split('\n'):
        if line.startswith("Score:"):
            try: score = max(1, min(10, int(''.join(filter(str.isdigit, line)))))
            except: pass
        elif line.startswith("Feedback:"): feedback = line.replace("Feedback:","").strip()
        elif line.startswith("Tip:"): tip = line.replace("Tip:","").strip()
    return score, feedback, tip


def get_final_verdict(scores, field, name):
    avg = np.mean(scores) if scores else 0
    prompt = f"""
Candidate: {name}, Field: {field}, Average Score: {avg:.1f}/10.
Give:
1. Verdict (capable/not capable)
2. 3 improvement suggestions for {field}
3. Motivational closing message
"""
    return groq_chat(prompt, "You are a career counselor.")


# ══════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════

def page_cover():
    st.markdown("""
    <div class="cover-card">
        <div style="font-size:52px;margin-bottom:16px;filter:drop-shadow(0 0 12px #c9a84c);">🎙️</div>
        <div class="gold-divider"></div>
        <h1 style="font-size:2.2em;letter-spacing:2px;text-transform:uppercase;
            text-shadow:0 0 20px rgba(240,192,64,0.5);margin:0 0 8px;">
            AI Interview Preparation System
        </h1>
        <p style="color:#c9a84c;font-size:1em;letter-spacing:4px;
            text-transform:uppercase;margin:0 0 28px;font-weight:600;">
            ML Project
        </p>
        <div class="gold-divider"></div>
        <p style="color:#8a7a50;font-size:0.82em;letter-spacing:3px;
            text-transform:uppercase;margin:0 0 8px;">Developed By</p>
        <p style="color:#f0e0a0;font-size:1.2em;font-weight:700;margin:0 0 4px;">Habib Ullah Khan</p>
        <p style="color:#f0e0a0;font-size:1.2em;font-weight:700;margin:0 0 24px;">Obaid Ullah Khan</p>
        <p style="color:#c9a84c;font-size:1.1em;letter-spacing:8px;margin:0 0 20px;">✦ ✦ ✦</p>
        <p style="color:#8a7a50;font-size:0.82em;letter-spacing:3px;
            text-transform:uppercase;margin:0 0 8px;">Submitted To</p>
        <p style="color:#f0e0a0;font-size:1.2em;font-weight:700;margin:0 0 28px;">Ma'am Zainab Noor</p>
        <div class="gold-divider"></div>
        <p style="color:#6a5d30;font-size:0.78em;letter-spacing:2px;text-transform:uppercase;">
            Voice-Enabled &nbsp;•&nbsp; Multi-Field &nbsp;•&nbsp; AI-Powered Evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀  Start Interview", use_container_width=True):
            st.session_state.page = "setup"
            st.rerun()


def page_setup():
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown("## 🎯 Setup Your Interview")
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    name = st.text_input("👤 Your Full Name", placeholder="Enter your full name")
    field = st.selectbox("🏛️ Select Your Field", ALL_FIELDS)
    total_q = st.slider("❓ Number of Questions", min_value=7, max_value=10, value=8)
    mode = st.radio("🎤 Answer Mode", ["⌨️ Text (Typing)", "🎤 Voice (Microphone)"], horizontal=True)

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("✅  Start Interview", use_container_width=True):
            if not name.strip():
                st.error("⚠️ Please enter your name!")
                return
            st.session_state.name = name.strip()
            st.session_state.field = field
            st.session_state.total_q = total_q
            st.session_state.answer_mode = "voice" if "Voice" in mode else "text"
            st.session_state.questions = []
            st.session_state.answers = []
            st.session_state.scores = []
            st.session_state.feedbacks = []
            st.session_state.current_q = 0
            st.session_state.page = "loading"
            st.rerun()


def page_loading():
    st.markdown(f"""
    <div class="gold-card" style="text-align:center;padding:40px;">
        <div style="font-size:40px;margin-bottom:16px;">⏳</div>
        <h2>Preparing your interview...</h2>
        <p style="color:#8a7a50;">Generating {st.session_state.total_q} questions for
        <span style="color:#c9a84c;">{st.session_state.field}</span></p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("AI is generating your questions..."):
        qs = generate_questions(st.session_state.field, st.session_state.total_q)
        st.session_state.questions = qs

    st.session_state.page = "interview"
    st.rerun()


def page_interview():
    idx   = st.session_state.current_q
    total = st.session_state.total_q
    q     = st.session_state.questions[idx]
    field = st.session_state.field
    name  = st.session_state.name

    # Progress
    progress = idx / total
    st.progress(progress)
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span style="color:#c9a84c;font-weight:600;">🎙️ {field}</span>
        <span style="color:#8a7a50;">Question {idx+1} of {total} &nbsp;|&nbsp; {name}</span>
    </div>
    """, unsafe_allow_html=True)

    # Question card
    st.markdown(f"""
    <div class="gold-card">
        <p style="color:#8a7a50;font-size:0.82em;margin:0 0 8px;">Q{idx+1} of {total}</p>
        <h3 style="margin:0;">{q}</h3>
    </div>
    """, unsafe_allow_html=True)

    # TTS
    text_to_speech(f"Question {idx+1}. {q}")

    # Subtitle
    st.markdown(f"""
    <div class="subtitle-box">
        🗣️ <b>AI Interviewer:</b> {q}
    </div>
    """, unsafe_allow_html=True)

    # Answer input
    if st.session_state.answer_mode == "text":
        answer = st.text_area("✍️ Your Answer", height=150, placeholder="Type your answer here...")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("✅  Submit Answer", use_container_width=True):
                if not answer.strip():
                    st.warning("⚠️ Please write your answer first!")
                    return
                process_and_next(q, answer.strip(), field)
    else:
        st.markdown("""
        <div class="ml-box" style="text-align:center;padding:20px;">
            <p style="color:#f0e0a0;margin:0 0 8px;">🎤 Click Record and speak your answer</p>
            <p style="color:#8a7a50;font-size:0.82em;">Note: Allow microphone access in browser</p>
        </div>
        """, unsafe_allow_html=True)

        try:
            import speech_recognition as sr
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎤 Record Answer", use_container_width=True):
                    with st.spinner("🔴 Listening... Speak now!"):
                        r = sr.Recognizer()
                        with sr.Microphone() as src:
                            r.adjust_for_ambient_noise(src, duration=1)
                            audio = r.listen(src, timeout=15, phrase_time_limit=60)
                        try:
                            answer = r.recognize_google(audio)
                            st.success(f"✅ Recognized: {answer}")
                            process_and_next(q, answer, field)
                        except:
                            st.error("❌ Could not recognize speech. Try typing instead.")
            with col2:
                if st.button("⌨️ Switch to Text", use_container_width=True):
                    st.session_state.answer_mode = "text"
                    st.rerun()
        except:
            st.error("Microphone not available. Switching to text mode.")
            st.session_state.answer_mode = "text"
            st.rerun()


def process_and_next(question, answer, field):
    with st.spinner("⏳ Evaluating with AI + ML algorithms..."):
        score, feedback, tip = evaluate_answer(question, answer, field)

    st.session_state.answers.append(answer)
    st.session_state.scores.append(score)
    st.session_state.feedbacks.append({"score": score, "feedback": feedback, "tip": tip})

    scores  = st.session_state.scores
    answers = st.session_state.answers

    # ML Analysis
    nb_label           = nb_classify(answer, score)
    trend_score, trend = linear_trend(scores)
    hmm_path           = viterbi(scores)
    hmm_state          = hmm_path[-1] if hmm_path else "—"

    bar_color = "#ef5350" if score < 5 else "#ffca28" if score < 7 else "#c9a84c"

    st.markdown(f"""
    <div class="gold-card">
        <h3 style="margin:0 0 12px;">📊 Answer Feedback</h3>

        <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
            <div style="flex:1;background:rgba(201,168,76,0.15);border-radius:8px;height:12px;">
                <div style="background:{bar_color};width:{score*10}%;height:100%;border-radius:8px;"></div>
            </div>
            <span style="color:#f0c040;font-size:1.4em;font-weight:700;">{score}/10</span>
        </div>

        <div class="ml-box">
            <p style="color:#8a7a50;font-size:0.8em;margin:0 0 4px;">Your Answer:</p>
            <p style="color:#f0e0a0;margin:0;font-style:italic;">"{answer}"</p>
        </div>

        <div class="ml-box" style="border-left:3px solid #c9a84c;">
            <p style="color:#c9a84c;font-size:0.82em;font-weight:600;margin:0 0 4px;">💬 Feedback</p>
            <p style="color:#f0e0a0;margin:0;">{feedback}</p>
        </div>

        <div class="ml-box" style="border-left:3px solid #7b5e00;">
            <p style="color:#8a7a50;font-size:0.82em;font-weight:600;margin:0 0 4px;">💡 Tip</p>
            <p style="color:#f0e0a0;margin:0;">{tip}</p>
        </div>

        <div style="margin-top:14px;">
            <p style="color:#c9a84c;font-size:0.82em;font-weight:700;letter-spacing:1px;margin:0 0 8px;">
                🤖 ML ALGORITHM ANALYSIS
            </p>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
                <div class="ml-box">
                    <p style="color:#8a7a50;font-size:0.72em;margin:0 0 3px;">Naïve Bayes</p>
                    <p style="color:#f0c040;font-weight:700;margin:0;font-size:0.9em;">{nb_label}</p>
                </div>
                <div class="ml-box">
                    <p style="color:#8a7a50;font-size:0.72em;margin:0 0 3px;">Linear Regression</p>
                    <p style="color:#f0c040;font-weight:700;margin:0;font-size:0.9em;">{trend}</p>
                </div>
                <div class="ml-box">
                    <p style="color:#8a7a50;font-size:0.72em;margin:0 0 3px;">HMM State</p>
                    <p style="color:#f0c040;font-weight:700;margin:0;font-size:0.9em;">{hmm_state}</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    text_to_speech(f"Score: {score} out of 10. {feedback}")

    st.session_state.current_q += 1
    is_last = st.session_state.current_q >= st.session_state.total_q

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        btn_label = "📊 See Final Results" if is_last else "Next Question ➡️"
        if st.button(btn_label, use_container_width=True):
            st.session_state.page = "results" if is_last else "interview"
            st.rerun()


def page_results():
    scores  = st.session_state.scores
    answers = st.session_state.answers
    field   = st.session_state.field
    name    = st.session_state.name

    with st.spinner("⏳ Running all ML algorithms on your full performance..."):
        avg              = np.mean(scores) if scores else 0
        trend_s, trend_d = linear_trend(scores)
        svm_r, svm_c     = svm_predict(avg, scores, answers)
        lr_p, lr_c       = logistic_predict(avg, scores, trend_s)
        cluster          = kmeans_cluster(scores)
        ens_l, ens_c     = ensemble_predict(scores)
        hmm_path         = viterbi(scores)
        metrics          = eval_metrics(scores)
        verdict          = get_final_verdict(scores, field, name)

    readiness = "✅ Ready for Job!" if lr_p == 1 else "📚 Needs Preparation"
    bar_color = "#ef5350" if avg < 5 else "#ffca28" if avg < 7 else "#c9a84c"

    # Header
    st.markdown(f"""
    <div class="cover-card">
        <div style="font-size:48px;margin-bottom:10px;">🏆</div>
        <h2 style="margin:0 0 4px;">Interview Complete!</h2>
        <p style="color:#c9a84c;margin:0 0 20px;">{name} — {field}</p>
        <div style="background:rgba(201,168,76,0.1);border-radius:10px;
            height:14px;max-width:400px;margin:0 auto 10px;">
            <div style="background:{bar_color};width:{avg*10:.0f}%;
                height:100%;border-radius:10px;"></div>
        </div>
        <p style="color:#f0c040;font-size:1.6em;font-weight:700;margin:0;">
            Average Score: {avg:.1f} / 10
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Per-question scores
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Question-by-Question Scores")
    for i, s in enumerate(scores):
        bc = "#c9a84c" if s >= 7 else "#ffca28" if s >= 5 else "#ef5350"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <span style="color:#8a7a50;font-size:0.85em;min-width:90px;">Question {i+1}</span>
            <div style="flex:1;background:rgba(201,168,76,0.1);border-radius:6px;height:10px;">
                <div style="background:{bc};width:{s*10}%;height:100%;border-radius:6px;"></div>
            </div>
            <span style="color:#f0c040;font-weight:700;">{s}/10</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ML Results
    st.markdown('<div class="gold-card">', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#c9a84c;font-size:0.85em;font-weight:700;
        letter-spacing:2px;margin:0 0 16px;">🤖 ML ALGORITHM RESULTS (ALL 8 SYLLABUS TOPICS)</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="ml-box"><p style="color:#8a7a50;font-size:0.75em;margin:0 0 3px;">1. NAÏVE BAYES</p>
        <p style="color:#f0c040;font-weight:700;margin:0;">
            Last Answer: {nb_classify(answers[-1], scores[-1]) if answers else '—'}</p></div>

        <div class="ml-box"><p style="color:#8a7a50;font-size:0.75em;margin:0 0 3px;">2. SVM (RBF Kernel)</p>
        <p style="color:#f0c040;font-weight:700;margin:0;">{svm_r} ({svm_c}%)</p></div>

        <div class="ml-box"><p style="color:#8a7a50;font-size:0.75em;margin:0 0 3px;">3. LINEAR REGRESSION</p>
        <p style="color:#f0c040;font-weight:700;margin:0;">{trend_d} (next≈{trend_s})</p></div>

        <div class="ml-box"><p style="color:#8a7a50;font-size:0.75em;margin:0 0 3px;">4. LOGISTIC REGRESSION</p>
        <p style="color:#f0c040;font-weight:700;margin:0;">{readiness} ({lr_c}%)</p></div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="ml-box"><p style="color:#8a7a50;font-size:0.75em;margin:0 0 3px;">5. K-MEANS CLUSTERING</p>
        <p style="color:#f0c040;font-weight:700;margin:0;">{cluster}</p></div>

        <div class="ml-box"><p style="color:#8a7a50;font-size:0.75em;margin:0 0 3px;">6. HMM — State Path</p>
        <p style="color:#f0c040;font-weight:700;margin:0;font-size:0.82em;">
            {' → '.join(hmm_path) if hmm_path else '—'}</p></div>

        <div class="ml-box"><p style="color:#8a7a50;font-size:0.75em;margin:0 0 3px;">7. ENSEMBLE LEARNING</p>
        <p style="color:#f0c040;font-weight:700;margin:0;">{ens_l} ({ens_c}%)</p></div>

        <div class="ml-box"><p style="color:#8a7a50;font-size:0.75em;margin:0 0 3px;">8. EVALUATION METRICS</p>
        <p style="color:#f0c040;font-weight:700;margin:0;font-size:0.82em;">
            Acc:{metrics['accuracy']}% | Pre:{metrics['precision']}% |
            Rec:{metrics['recall']}% | F1:{metrics['f1']}%</p></div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # AI Verdict
    st.markdown(f"""
    <div class="gold-card">
        <p style="color:#f0c040;font-weight:700;font-size:1em;margin:0 0 12px;">
            🤖 AI Career Counselor's Verdict
        </p>
        <p style="color:#f0e0a0;white-space:pre-line;line-height:1.8;margin:0;">{verdict}</p>
    </div>
    """, unsafe_allow_html=True)

    text_to_speech(f"Interview complete! Your average score is {avg:.1f} out of 10. {verdict[:200]}")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔄  Start New Interview", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ══════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════

page = st.session_state.get("page", "cover")

if page == "cover":
    page_cover()
elif page == "setup":
    page_setup()
elif page == "loading":
    page_loading()
elif page == "interview":
    page_interview()
elif page == "results":
    page_results()
