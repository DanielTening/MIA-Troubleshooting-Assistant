import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import re
import json
import hmac
import hashlib
import unicodedata
from collections import defaultdict, Counter
from deep_translator import GoogleTranslator
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
ART_DIR = "artifacts"

SVM_PIPE_PATH = os.path.join(ART_DIR, "svm_tfidf_pipeline.joblib")
SVM_LABELS_PATH = os.path.join(ART_DIR, "svm_label_classes.npy")
RF_LABELS_PATH = os.path.join(ART_DIR, "rf_label_classes.npy")
RF_PIPE_PATH = os.path.join(ART_DIR, "rf_tfidf_pipeline.joblib")

GRAPH_JSON_PATH = os.path.join(ART_DIR, "train_graph_indices.json")
RETRIEVAL_CSV_PATH = os.path.join(ART_DIR, "train_retrieval_corpus.csv")
TRAIN_EMB_PATH = os.path.join(ART_DIR, "train_symptom_embeddings.npy")
ACTION_EMB_PATH = os.path.join(ART_DIR, "action_label_embeddings.npy")
DATASET_XLSX_PATH = os.path.join(ART_DIR, "Annotated_Data_cleaned.xlsx")

USERS_JSON_PATH = "users.json"
RF_FILE_ID = "17ERcZrPGbODL2mMipshdTAtl7fXrL6Pj"
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 5
TOPN_RETRIEVAL = 50

# True three-way fusion
W_SVM = 0.35
W_RF = 0.35
W_KG = 0.30

INPUT_COL = "symptom_text"
ACTION_COL = "action_text"
CAUSE_COL = "cause_text"
ASSET_COL = "asset_text"
FAULT_COL = "fault_class_text"
RECORD_ID_COL = "record_id"
TEXT = {
    "English": {
        "app_title": "Troubleshooting Assistant",
        "app_subtitle": "Bilingual industrial maintenance recommendation assistant",
        "login_title": "User Login",
        "username": "Username",
        "password": "Password",
        "login_btn": "Login",
        "logout_btn": "Logout",
        "login_error": "Invalid username or password.",
        "login_ok": "Login successful.",
        "lang": "Language",
        "symptom": "Describe the symptom",
        "asset": "Asset filter (optional)",
        "fault": "Fault class filter (optional)",
        "analyze": "Analyze",
        "clear": "Clear",
        "asset_all": "All assets",
        "fault_all": "All fault classes",
        "best_action": "Best action to implement first",
        "best_reason": "Why this is recommended",
        "top_actions": "Top 5 action recommendations",
        "probable_causes": "Probable causes",
        "similar_cases": "Most relevant similar incidents",
        "score": "Score",
        "action": "Action",
        "cause": "Cause",
        "asset_col": "Asset",
        "fault_col": "Fault class",
        "incident": "Incident",
        "reason_1": "It received the highest fused score from the SVM, RF, and KG-enhanced retrieval components.",
        "reason_2": "It is supported by semantically similar historical incidents.",
        "reason_3": "It remains consistent with the filtered asset/fault context.",
        "reason_4": "Probable causes from related cases also support this recommendation.",
        "empty_query": "Please enter a symptom description.",
        "no_results": "No recommendation could be generated from the current input and filters.",
        "warning_filter": "The selected filters are restrictive. Results are based on a reduced evidence pool.",
        "system_note": "The assistant uses multilingual semantic retrieval to improve tolerance to noisy wording and spelling variation.",
        "fusion_note": "Fusion mode: SVM + RF + KG retrieval",
        "likely_causes_note": "Causes are inferred from similar historical incidents and graph-style evidence, when available."
    },
    "Français": {
        "app_title": "Assistant de dépannage",
        "app_subtitle": "Assistant bilingue de recommandation en maintenance industrielle",
        "login_title": "Connexion utilisateur",
        "username": "Nom d'utilisateur",
        "password": "Mot de passe",
        "login_btn": "Se connecter",
        "logout_btn": "Se déconnecter",
        "login_error": "Nom d'utilisateur ou mot de passe invalide.",
        "login_ok": "Connexion réussie.",
        "lang": "Langue",
        "symptom": "Décrivez le symptôme",
        "asset": "Filtre actif/équipement (optionnel)",
        "fault": "Filtre classe de défaut (optionnel)",
        "analyze": "Analyser",
        "clear": "Effacer",
        "asset_all": "Tous les actifs",
        "fault_all": "Toutes les classes de défaut",
        "best_action": "Meilleure action à mettre en œuvre en premier",
        "best_reason": "Pourquoi cette recommandation",
        "top_actions": "Top 5 des actions recommandées",
        "probable_causes": "Causes probables",
        "similar_cases": "Incidents historiques les plus pertinents",
        "score": "Score",
        "action": "Action",
        "cause": "Cause",
        "asset_col": "Actif",
        "fault_col": "Classe de défaut",
        "incident": "Incident",
        "reason_1": "Elle a obtenu le score fusionné le plus élevé à partir des composantes SVM, RF et récupération enrichie par KG.",
        "reason_2": "Elle est soutenue par des incidents historiques sémantiquement similaires.",
        "reason_3": "Elle reste cohérente avec le contexte filtré actif/classe de défaut.",
        "reason_4": "Les causes probables issues des cas liés soutiennent également cette recommandation.",
        "empty_query": "Veuillez saisir une description du symptôme.",
        "no_results": "Aucune recommandation n’a pu être générée à partir de la saisie et des filtres actuels.",
        "warning_filter": "Les filtres sélectionnés sont restrictifs. Les résultats reposent sur un ensemble réduit d’évidences.",
        "system_note": "L’assistant utilise une récupération sémantique multilingue pour mieux tolérer les formulations bruitées et les erreurs d’orthographe.",
        "fusion_note": "Mode de fusion : SVM + RF + récupération KG",
        "likely_causes_note": "Les causes sont inférées à partir d’incidents historiques similaires et d’évidences de type graphe, lorsqu’elles sont disponibles."
    }
}
def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = strip_accents(text)
    text = re.sub(r"[^\w\s;/\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_semicolon_values(value: str):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = [p.strip() for p in str(value).split(";")]
    parts = [p for p in parts if p]
    return parts

def scores_to_proba(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    vec[vec < 0] = 0.0
    s = vec.sum()
    if s <= 0:
        return np.ones_like(vec) / len(vec)
    return vec / s

def norm01_map(d: dict) -> dict:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin:
        return {k: 0.0 for k in d}
    return {k: (float(v) - vmin) / (vmax - vmin) for k, v in d.items()}
def pbkdf2_hash(password: str, salt: str, iterations: int = 200_000) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return dk.hex()

def verify_password(username: str, password: str, users_db: dict) -> bool:
    user = users_db.get(username)
    if not user:
        return False
    calc = pbkdf2_hash(password, user["salt"], user.get("iterations", 200_000))
    return hmac.compare_digest(calc, user["password_hash"])

def load_users():
    if os.path.exists(USERS_JSON_PATH):
        with open(USERS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMB_MODEL_NAME)

def download_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

@st.cache_resource
def load_models():
    download_from_gdrive(RF_FILE_ID, RF_PIPE_PATH)
    svm_pipe = joblib.load(SVM_PIPE_PATH)
    rf_pipe = joblib.load(RF_PIPE_PATH)
    svm_labels = np.load(SVM_LABELS_PATH, allow_pickle=True)
    rf_labels = np.load(RF_LABELS_PATH, allow_pickle=True)
    return svm_pipe, rf_pipe, svm_labels, rf_labels

@st.cache_data
def load_graph_and_corpus():
    with open(GRAPH_JSON_PATH, "r", encoding="utf-8") as f:
        graph_idx = json.load(f)

    corpus = pd.read_csv(RETRIEVAL_CSV_PATH)

    # normalize missing columns safely
    for col in [INPUT_COL, ACTION_COL, CAUSE_COL, ASSET_COL, FAULT_COL]:
        if col not in corpus.columns:
            corpus[col] = ""
        corpus[col] = corpus[col].fillna("").astype(str)

    # asset split list per row
    corpus["_asset_list"] = corpus[ASSET_COL].apply(split_semicolon_values)
    corpus["_asset_list_clean"] = corpus["_asset_list"].apply(lambda xs: [clean_text(x) for x in xs])
    corpus["_fault_clean"] = corpus[FAULT_COL].apply(clean_text)
    corpus["_symptom_clean"] = corpus[INPUT_COL].apply(clean_text)

    return graph_idx, corpus

@st.cache_data
def load_embeddings():
    train_symptom_emb = np.load(TRAIN_EMB_PATH)
    action_label_emb = np.load(ACTION_EMB_PATH)
    return train_symptom_emb, action_label_emb

@st.cache_data
def load_dropdown_source():
    df = pd.read_excel(DATASET_XLSX_PATH)
    for col in [ASSET_COL, FAULT_COL]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    all_assets = []
    for val in df[ASSET_COL]:
        all_assets.extend(split_semicolon_values(val))

    # unique cleaned mapping while preserving readable form
    asset_map = {}
    for a in all_assets:
        key = clean_text(a)
        if key and key not in asset_map:
            asset_map[key] = a.strip()

    all_faults = []
    for val in df[FAULT_COL]:
        txt = str(val).strip()
        if txt:
            all_faults.append(txt)

    fault_map = {}
    for f in all_faults:
        key = clean_text(f)
        if key and key not in fault_map:
            fault_map[key] = f.strip()

    asset_options = [asset_map[k] for k in sorted(asset_map.keys())]
    fault_options = [fault_map[k] for k in sorted(fault_map.keys())]
    return asset_options, fault_options
def align_rf_proba_to_svm_labels(rf_proba: np.ndarray, rf_pipe, rf_labels, svm_labels):
    clf_classes = rf_pipe.named_steps["clf"].classes_

    # map RF class id -> action label text
    mapped_rf_labels = []
    for cid in clf_classes:
        cid_int = int(cid)
        mapped_rf_labels.append(str(rf_labels[cid_int]))

    rf_pos = {lab: j for j, lab in enumerate(mapped_rf_labels)}
    aligned = np.zeros((rf_proba.shape[0], len(svm_labels)), dtype=np.float64)

    for i, lab in enumerate(svm_labels):
        lab = str(lab)
        if lab in rf_pos:
            aligned[:, i] = rf_proba[:, rf_pos[lab]]

    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    aligned = aligned / row_sums
    return aligned

def dense_action_vector(action_score_dict: dict, svm_labels) -> np.ndarray:
    idx = {str(a): i for i, a in enumerate(svm_labels)}
    vec = np.zeros(len(svm_labels), dtype=np.float64)
    for act, sc in action_score_dict.items():
        j = idx.get(str(act))
        if j is not None:
            vec[j] += float(sc)
    return normalize_vector(vec)

def get_filtered_subset(corpus: pd.DataFrame, embeddings: np.ndarray, asset_filter: str, fault_filter: str):
    mask = np.ones(len(corpus), dtype=bool)

    if asset_filter:
        asset_clean = clean_text(asset_filter)
        mask = mask & corpus["_asset_list_clean"].apply(lambda xs: asset_clean in xs).values

    if fault_filter:
        fault_clean = clean_text(fault_filter)
        mask = mask & (corpus["_fault_clean"].values == fault_clean)

    subset = corpus.loc[mask].copy()
    subset_idx = np.where(mask)[0]
    subset_emb = embeddings[subset_idx] if len(subset_idx) > 0 else np.empty((0, embeddings.shape[1]))
    return subset, subset_emb, mask.sum()

def build_kg_distribution(
    query_text: str,
    subset: pd.DataFrame,
    subset_emb: np.ndarray,
    emb_model,
    graph_idx: dict,
    svm_labels,
    action_emb_lookup: pd.DataFrame,
    asset_filter: str = "",
    topn: int = 50
):
    q = clean_text(query_text)
    if not q or len(subset) == 0:
        return (
            np.ones(len(svm_labels), dtype=np.float64) / len(svm_labels),
            [],
            [],
            []
        )

    q_emb = emb_model.encode([q], normalize_embeddings=True).astype("float32")
    sims = cosine_similarity(q_emb, subset_emb)[0]
    top_idx = np.argsort(-sims)[: min(topn, len(sims))]
    top_rows = subset.iloc[top_idx].copy()
    top_rows["_retrieval_score"] = [float(sims[i]) for i in top_idx]

    symptom_to_actions = graph_idx.get("symptom_to_actions", {})
    symptom_to_causes = graph_idx.get("symptom_to_causes", {})
    cause_to_actions = graph_idx.get("cause_to_actions", {})
    asset_to_actions = graph_idx.get("asset_to_actions", {})

    sim_support = defaultdict(float)
    freq_support = Counter()
    graph_support = Counter()
    asset_support = Counter()
    cause_counter = Counter()
    action_to_causes = defaultdict(Counter)

    asset_clean = clean_text(asset_filter) if asset_filter else ""

    candidate_actions = set()

    for _, row in top_rows.iterrows():
        sim = float(row["_retrieval_score"])
        sym = clean_text(row.get(INPUT_COL, ""))
        act = str(row.get(ACTION_COL, "")).strip()
        cau = str(row.get(CAUSE_COL, "")).strip()
        row_assets = row.get("_asset_list_clean", [])

        if act:
            candidate_actions.add(act)
            sim_support[act] += sim
            freq_support[act] += 1
            if cau:
                action_to_causes[act][cau] += 1
                cause_counter[cau] += 1

        related_causes = symptom_to_causes.get(sym, [])
        related_actions = set(symptom_to_actions.get(sym, []))

        for c in related_causes:
            if c:
                cause_counter[c] += 1
            related_actions.update(cause_to_actions.get(c, []))

        if asset_clean:
            for a in asset_to_actions.get(asset_clean, []):
                related_actions.add(a)

        for a in related_actions:
            candidate_actions.add(a)
            graph_support[a] += 1
            if asset_clean and asset_clean in row_assets:
                asset_support[a] += 1

    if not candidate_actions:
        return (
            np.ones(len(svm_labels), dtype=np.float64) / len(svm_labels),
            [],
            [],
            []
        )

    cand_actions = sorted([a for a in candidate_actions if a in action_emb_lookup.index])
    if len(cand_actions) > 0:
        cand_emb = action_emb_lookup.loc[cand_actions].values
        sem = cosine_similarity(q_emb, cand_emb)[0]
        sem_map = {a: float(s) for a, s in zip(cand_actions, sem)}
    else:
        sem_map = {}

    sim_n = norm01_map(sim_support)
    freq_n = norm01_map({a: float(freq_support[a]) for a in candidate_actions})
    graph_n = norm01_map({a: float(graph_support[a]) for a in candidate_actions})
    asset_n = norm01_map({a: float(asset_support[a]) for a in candidate_actions})
    sem_n = norm01_map(sem_map)

    final_action_scores = {}
    for a in candidate_actions:
        final_action_scores[a] = (
            0.35 * sim_n.get(a, 0.0)
            + 0.20 * freq_n.get(a, 0.0)
            + 0.20 * graph_n.get(a, 0.0)
            + 0.10 * asset_n.get(a, 0.0)
            + 0.15 * sem_n.get(a, 0.0)
        )

    kg_vec = dense_action_vector(final_action_scores, svm_labels)

    top_causes = [
        {"cause": c, "support": int(n)}
        for c, n in cause_counter.most_common(5)
        if str(c).strip()
    ]

    evidence_rows = []
    for _, r in top_rows.head(5).iterrows():
        evidence_rows.append({
            "record_id": str(r.get(RECORD_ID_COL, "")),
            "symptom": str(r.get(INPUT_COL, "")),
            "action": str(r.get(ACTION_COL, "")),
            "cause": str(r.get(CAUSE_COL, "")),
            "asset": str(r.get(ASSET_COL, "")),
            "fault_class": str(r.get(FAULT_COL, "")),
            "score": round(float(r.get("_retrieval_score", 0.0)), 4),
        })

    top_action_causes = {
        act: [c for c, _ in action_to_causes[act].most_common(3)]
        for act in action_to_causes
    }

    return kg_vec, top_causes, evidence_rows, top_action_causes

def is_english(text):
    text = text.lower()
    # simple heuristic
    english_keywords = ["the", "pump", "flow", "error", "low", "high", "leak"]
    return any(w in text for w in english_keywords)


def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text  

def predict_all(query_text: str, asset_filter: str = "", fault_filter: str = ""):
    svm_pipe, rf_pipe, svm_labels, rf_labels = load_models()
    graph_idx, corpus = load_graph_and_corpus()
    train_symptom_emb, action_label_emb = load_embeddings()
    emb_model = load_embedding_model()

    subset, subset_emb, subset_count = get_filtered_subset(
        corpus=corpus,
        embeddings=train_symptom_emb,
        asset_filter=asset_filter,
        fault_filter=fault_filter
    )

    # fallback if filters empty out the evidence pool
    used_restrictive_subset = True
    if len(subset) == 0:
        subset = corpus.copy()
        subset_emb = train_symptom_emb
        subset_count = len(subset)
        used_restrictive_subset = False

    action_emb_lookup = pd.DataFrame(action_label_emb, index=[str(x) for x in svm_labels])
    action_emb_lookup = action_emb_lookup.groupby(level=0).first()

    # SVM
    svm_scores = svm_pipe.decision_function([query_text])
    svm_proba = scores_to_proba(svm_scores)[0]

    # RF
    rf_proba_raw = rf_pipe.predict_proba([query_text])
    rf_proba = align_rf_proba_to_svm_labels(rf_proba_raw, rf_pipe, rf_labels, svm_labels)[0]

    # KG
    kg_vec, top_causes, evidence_rows, top_action_causes = build_kg_distribution(
        query_text=query_text,
        subset=subset,
        subset_emb=subset_emb,
        emb_model=emb_model,
        graph_idx=graph_idx,
        svm_labels=svm_labels,
        action_emb_lookup=action_emb_lookup,
        asset_filter=asset_filter,
        topn=TOPN_RETRIEVAL
    )

    fused = (W_SVM * svm_proba) + (W_RF * rf_proba) + (W_KG * kg_vec)
    fused = normalize_vector(fused)

    top_idx = np.argsort(-fused)[:TOP_K]
    top_actions = []
    for idx in top_idx:
        act = str(svm_labels[idx])
        top_actions.append({
            "action": act,
            "score": round(float(fused[idx]), 4),
            "svm_score": round(float(svm_proba[idx]), 4),
            "rf_score": round(float(rf_proba[idx]), 4),
            "kg_score": round(float(kg_vec[idx]), 4),
            "probable_causes": top_action_causes.get(act, [])
        })

    return {
        "top_actions": top_actions,
        "top_causes": top_causes,
        "evidence_rows": evidence_rows,
        "subset_count": subset_count,
        "used_restrictive_subset": used_restrictive_subset
    }
st.set_page_config(page_title="KG Troubleshooting Assistant", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

lang = st.sidebar.selectbox("Language / Langue", ["English", "Français"])
T = TEXT[lang]

users_db = load_users()

if not st.session_state.authenticated:
    st.title(T["login_title"])
    col1, col2 = st.columns([1, 2])

    with col1:
        username = st.text_input(T["username"])
        password = st.text_input(T["password"], type="password")
        if st.button(T["login_btn"], use_container_width=True):
            if verify_password(username, password, users_db):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success(T["login_ok"])
                st.rerun()
            else:
                st.error(T["login_error"])

    with col2:
        st.info(
            "This application uses a knowledge-graph-enhanced retrieval pipeline with multilingual semantic matching."
            if lang == "English"
            else "Cette application utilise un pipeline de récupération enrichi par graphe de connaissances avec appariement sémantique multilingue."
        )
    st.stop()
def build_pdf_report(
    T,
    symptom_input,
    asset_filter,
    fault_filter,
    best,
    reasons,
    top_actions,
    top_causes,
    evidence_rows,
    is_eng=False
):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=8 * mm,
        leftMargin=8 * mm,
        topMargin=8 * mm,
        bottomMargin=8 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleSmall",
        parent=styles["Title"],
        fontSize=12,
        leading=14,
        spaceAfter=4
    )

    section_style = ParagraphStyle(
        "SectionSmall",
        parent=styles["Heading2"],
        fontSize=9,
        leading=10,
        spaceAfter=3,
        spaceBefore=4
    )

    body_style = ParagraphStyle(
        "BodySmall",
        parent=styles["BodyText"],
        fontSize=7,
        leading=8,
        spaceAfter=2
    )

    tiny_style = ParagraphStyle(
        "Tiny",
        parent=styles["BodyText"],
        fontSize=6,
        leading=7,
        spaceAfter=1
    )

    elements = []

    # Title
    elements.append(Paragraph(T["app_title"], title_style))
    elements.append(Spacer(1, 2))

    # Input summary
    elements.append(Paragraph("<b>Query Summary</b>" if is_eng else "<b>Résumé de la requête</b>", section_style))
    elements.append(Paragraph(f"<b>{T['symptom']}:</b> {symptom_input}", body_style))
    elements.append(Paragraph(f"<b>{T['asset']}:</b> {asset_filter if asset_filter else '-'}", body_style))
    elements.append(Paragraph(f"<b>{T['fault']}:</b> {fault_filter if fault_filter else '-'}", body_style))

    # Best action
    best_action_txt = best["action"]
    if is_eng:
        try:
            best_action_txt = translate_to_english(best_action_txt)
        except:
            pass

    elements.append(Paragraph(T["best_action"], section_style))
    elements.append(Paragraph(f"<b>{best_action_txt}</b> | {T['score']}: {best['score']}", body_style))

    # Reasons
    elements.append(Paragraph(T["best_reason"], section_style))
    for r in reasons:
        elements.append(Paragraph(f"• {r}", body_style))

    # Best probable causes
    if best.get("probable_causes"):
        cause_txt = ", ".join(
            [translate_to_english(c) if is_eng else c for c in best["probable_causes"]]
        )
        elements.append(Paragraph(f"<b>{T['probable_causes']}:</b> {cause_txt}", body_style))

    # Top actions table
    elements.append(Paragraph(T["top_actions"], section_style))
    top_actions_data = [[
        T["action"], T["score"], "SVM", "RF", "KG", T["probable_causes"]
    ]]
    for item in top_actions[:5]:
        top_actions_data.append([
            Paragraph(str(translate_to_english(item["action"]) if is_eng else item["action"]), tiny_style),
            str(item["score"]),
            str(item["svm_score"]),
            str(item["rf_score"]),
            str(item["kg_score"]),
            Paragraph("; ".join(item["probable_causes"]) if item["probable_causes"] else "", tiny_style)
        ])

    top_actions_table = Table(
        top_actions_data,
        colWidths=[42*mm, 16*mm, 14*mm, 14*mm, 14*mm, 68*mm],
        repeatRows=1
    )
    top_actions_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("LEADING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    elements.append(top_actions_table)

    # Probable causes table
    elements.append(Paragraph(T["probable_causes"], section_style))
    if top_causes:
        cause_data = [[T["cause"], "Support"]]
        for c in top_causes[:5]:
            cause_data.append([
                Paragraph(str(c["cause"]), tiny_style),
                str(c["support"])
            ])

        cause_table = Table(cause_data, colWidths=[150*mm, 25*mm], repeatRows=1)
        cause_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.black),
            ("FONTSIZE", (0, 0), (-1, -1), 6),
            ("LEADING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        elements.append(cause_table)
    else:
        elements.append(Paragraph("-", body_style))

    # Similar incidents table
    elements.append(Paragraph(T["similar_cases"], section_style))
    if evidence_rows:
        case_data = [[
            T["incident"], "Symptom", T["action"], T["cause"], T["asset_col"], T["fault_col"], T["score"]
        ]]
        for row in evidence_rows[:5]:
            case_data.append([
                Paragraph(str(row["record_id"]), tiny_style),
                Paragraph(str(row["symptom"]), tiny_style),
                Paragraph(str(row["action"]), tiny_style),
                Paragraph(str(row["cause"]), tiny_style),
                Paragraph(str(row["asset"]), tiny_style),
                Paragraph(str(row["fault_class"]), tiny_style),
                str(row["score"])
            ])

        case_table = Table(
            case_data,
            colWidths=[14*mm, 36*mm, 34*mm, 34*mm, 28*mm, 28*mm, 12*mm],
            repeatRows=1
        )
        case_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.black),
            ("FONTSIZE", (0, 0), (-1, -1), 5.5),
            ("LEADING", (0, 0), (-1, -1), 6.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
            ("TOPPADDING", (0, 0), (-1, -1), 1.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ]))
        elements.append(case_table)
    else:
        elements.append(Paragraph("-", body_style))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# main app
st.title(T["app_title"])
st.caption(T["app_subtitle"])

topbar1, topbar2 = st.columns([5, 1])
with topbar2:
    if st.button(T["logout_btn"], use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

st.info(f"{T['fusion_note']} | {T['system_note']}")

asset_options, fault_options = load_dropdown_source()

col1, col2 = st.columns([2, 1])

with col1:
    symptom_input = st.text_area(
        T["symptom"],
        height=140,
        placeholder=(
            "Example: pompe en arrêt, débit faible, problème de pistolet..."
            if lang == "Français"
            else "Example: pump stopped, low flow, nozzle problem..."
        )
    )

with col2:
    selected_asset = st.selectbox(T["asset"], [T["asset_all"]] + asset_options)
    selected_fault = st.selectbox(T["fault"], [T["fault_all"]] + fault_options)

c1, c2 = st.columns([1, 1])
analyze = c1.button(T["analyze"], use_container_width=True)
clear = c2.button(T["clear"], use_container_width=True)

if clear:
    st.rerun()

if analyze:
    if not symptom_input.strip():
        st.warning(T["empty_query"])
        st.stop()

    asset_filter = "" if selected_asset == T["asset_all"] else selected_asset
    fault_filter = "" if selected_fault == T["fault_all"] else selected_fault

    with st.spinner("Analyzing..." if lang == "English" else "Analyse en cours..."):
        result = predict_all(
            query_text=symptom_input,
            asset_filter=asset_filter,
            fault_filter=fault_filter
        )

    if result["subset_count"] < 30:
        st.warning(T["warning_filter"])

    top_actions = result["top_actions"]
    top_causes = result["top_causes"]
    evidence_rows = result["evidence_rows"]

    if not top_actions:
        st.error(T["no_results"])
        st.stop()

    best = top_actions[0]

    st.subheader(T["best_action"])
    display_action = translate_to_english(best['action']) if is_english(symptom_input) else best['action']

    st.success(f"**{display_action}**  |  {T['score']}: **{best['score']}**")

    reasons = [T["reason_1"], T["reason_2"]]
    if asset_filter or fault_filter:
        reasons.append(T["reason_3"])
    if best.get("probable_causes"):
        reasons.append(T["reason_4"])

    st.subheader(T["best_reason"])
    for r in reasons:
        st.write(f"- {r}")

    if best.get("probable_causes"):
        if is_english(symptom_input):
            cause_txt = ", ".join([translate_to_english(c) for c in best["probable_causes"]])
        else:
            cause_txt = ", ".join(best["probable_causes"])
        st.write(
            f"**{T['probable_causes']}:** {cause_txt}"
        )

    st.subheader(T["top_actions"])
    action_rows = []
    for item in top_actions:
        action_txt = translate_to_english(item["action"]) if is_english(symptom_input) else item["action"]

        action_rows.append({
            T["action"]: action_txt,
            T["score"]: item["score"],
            "SVM": item["svm_score"],
            "RF": item["rf_score"],
            "KG": item["kg_score"],
            T["probable_causes"]: "; ".join(item["probable_causes"]) if item["probable_causes"] else ""
        })
    st.dataframe(pd.DataFrame(action_rows), use_container_width=True)

    st.subheader(T["probable_causes"])
    if top_causes:
        cause_rows = [{T["cause"]: c["cause"], "Support": c["support"]} for c in top_causes]
        st.dataframe(pd.DataFrame(cause_rows), use_container_width=True)
    else:
        st.info(T["likely_causes_note"])

    st.subheader(T["similar_cases"])
    if evidence_rows:
        case_rows = []
        for row in evidence_rows:
            case_rows.append({
                T["incident"]: row["record_id"],
                "Symptom": row["symptom"],
                T["action"]: row["action"],
                T["cause"]: row["cause"],
                T["asset_col"]: row["asset"],
                T["fault_col"]: row["fault_class"],
                T["score"]: row["score"],
            })
        st.dataframe(pd.DataFrame(case_rows), use_container_width=True)
    else:
        st.info(T["no_results"])

    is_eng = is_english(symptom_input)
    pdf_bytes = build_pdf_report(
        T=T,
        symptom_input=symptom_input,
        asset_filter=asset_filter,
        fault_filter=fault_filter,
        best=best,
        reasons=reasons,
        top_actions=top_actions,
        top_causes=top_causes,
        evidence_rows=evidence_rows,
        is_eng=is_eng
    )

    st.download_button(
        label="Download PDF report" if lang == "English" else "Télécharger le rapport PDF",
        data=pdf_bytes,
        file_name="troubleshooting_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )