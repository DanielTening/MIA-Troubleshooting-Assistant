import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import io
import re
import json
import hmac
import hashlib
import unicodedata
from collections import defaultdict, Counter

import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

ART_DIR = "artifacts"

SVM_PIPE_PATH = os.path.join(ART_DIR, "svm_tfidf_pipeline.joblib")
SVM_LABELS_PATH = os.path.join(ART_DIR, "svm_label_classes.npy")
RF_LABELS_PATH = os.path.join(ART_DIR, "rf_label_classes.npy")

GRAPH_JSON_PATH = os.path.join(ART_DIR, "train_graph_indices.json")
RETRIEVAL_CSV_PATH = os.path.join(ART_DIR, "train_retrieval_corpus.csv")
TRAIN_EMB_PATH = os.path.join(ART_DIR, "train_symptom_embeddings.npy")
ACTION_EMB_PATH = os.path.join(ART_DIR, "action_label_embeddings.npy")

USERS_JSON_PATH = "users.json"

RF_FILE_ID = "17ERcZrPGbODL2mMipshdTAtl7fXrL6Pj"

# Fusion weights
W_SVM = 0.35
W_RF = 0.35
W_KG = 0.30

def pbkdf2_hash(password, salt, iterations=200000):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        iterations
    ).hex()

def verify_password(username, password, users):
    user = users.get(username)
    if not user:
        return False
    calc = pbkdf2_hash(password, user["salt"], user["iterations"])
    return hmac.compare_digest(calc, user["password_hash"])

def load_users():
    if os.path.exists(USERS_JSON_PATH):
        return json.load(open(USERS_JSON_PATH))
    return {}

def download_from_gdrive(file_id, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)

@st.cache_resource
def load_models():
    rf_path = os.path.join(ART_DIR, "rf_model.joblib")
    download_from_gdrive(RF_FILE_ID, rf_path)

    svm = joblib.load(SVM_PIPE_PATH)
    rf = joblib.load(rf_path)

    svm_labels = np.load(SVM_LABELS_PATH, allow_pickle=True)
    rf_labels = np.load(RF_LABELS_PATH, allow_pickle=True)

    return svm, rf, svm_labels, rf_labels

@st.cache_resource
def load_kg():
    graph = json.load(open(GRAPH_JSON_PATH))
    corpus = pd.read_csv(RETRIEVAL_CSV_PATH)

    train_emb = np.load(TRAIN_EMB_PATH)
    action_emb = np.load(ACTION_EMB_PATH)

    return graph, corpus, train_emb, action_emb

@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def clean_text(text):
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    return text

def kg_retrieval(query, graph, corpus, train_emb, action_emb, emb_model):
    q = clean_text(query)
    q_emb = emb_model.encode([q], normalize_embeddings=True)

    sims = cosine_similarity(q_emb, train_emb)[0]
    idx = np.argsort(-sims)[:50]

    rows = corpus.iloc[idx]

    action_scores = Counter()

    for _, row in rows.iterrows():
        action = row["action_text"]
        action_scores[action] += 1

    # Convert to vector
    actions = list(action_scores.keys())
    scores = np.array(list(action_scores.values()), dtype=float)

    if scores.sum() == 0:
        return np.ones(len(action_emb)) / len(action_emb)

    scores = scores / scores.sum()

    vec = np.zeros(len(action_emb))
    for i, a in enumerate(actions):
        if i < len(vec):
            vec[i] = scores[i]

    return vec

def predict(query):
    svm, rf, svm_labels, rf_labels = load_models()
    graph, corpus, train_emb, action_emb = load_kg()
    emb_model = load_embedding_model()

    svm_scores = svm.decision_function([query])
    svm_scores = np.exp(svm_scores) / np.sum(np.exp(svm_scores))

    rf_scores = rf.predict_proba([query])
    rf_scores = rf_scores / rf_scores.sum()

    kg_vec = kg_retrieval(query, graph, corpus, train_emb, action_emb, emb_model)

    fused = W_SVM * svm_scores + W_RF * rf_scores + W_KG * kg_vec
    fused = fused / fused.sum()

    idx = np.argsort(-fused[0])[:5]

    results = []
    for i in idx:
        results.append({
            "action": str(svm_labels[i]),
            "score": round(float(fused[0][i]), 4)
        })

    return results

def build_pdf(symptom, results):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    elements = []
    elements.append(Paragraph(f"<b>Symptom:</b> {symptom}", None))
    elements.append(Spacer(1, 10))

    for r in results:
        elements.append(Paragraph(f"{r['action']} - {r['score']}", None))
        elements.append(Spacer(1, 5))

    doc.build(elements)
    return buffer.getvalue()

st.set_page_config(layout="wide")

if "auth" not in st.session_state:
    st.session_state.auth = False

users = load_users()

# LOGIN
if not st.session_state.auth:
    st.title("Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_password(u, p, users):
            st.session_state.auth = True
            st.success("Login OK")
            st.rerun()
        else:
            st.error("Invalid login")

    st.stop()

# MAIN APP
st.title("Troubleshooting Assistant")

query = st.text_area("Enter symptom")

if st.button("Analyze"):
    if not query:
        st.warning("Enter symptom")
    else:
        results = predict(query)

        st.dataframe(pd.DataFrame(results))

        pdf = build_pdf(query, results)

        st.download_button(
            "Download PDF",
            pdf,
            file_name="report.pdf"
        )