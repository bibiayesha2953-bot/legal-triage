import streamlit as st
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

SAFETY_KEYWORDS = ["hit", "abuse", "threat", "unsafe", "afraid", "violence", "stalk", "assault"]

@st.cache_resource
def load_model():
    return joblib.load("triage_model.joblib")

data = load_model()
category_clf = data["category_clf"]
risk_clf = data["risk_clf"]
vectorizer = data["vectorizer"]
retrieval_vectorizer = data["retrieval_vectorizer"]
retrieval_matrix = data["retrieval_matrix"]
df = data["df"]

def predict_triage(user_text, top_k=1):
    cleaned = clean_text(user_text)
    cat_vec = vectorizer.transform([cleaned])
    predicted_category = category_clf.predict(cat_vec)[0]
    predicted_risk = risk_clf.predict(cat_vec)[0]

    query_vec = retrieval_vectorizer.transform([cleaned])
    sims = cosine_similarity(query_vec, retrieval_matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    match = df.iloc[top_idx[0]]

    text_lower = user_text.lower()
    is_urgent = (
        predicted_category == "Women's Safety"
        or predicted_risk == "High"
        or any(k in text_lower for k in SAFETY_KEYWORDS)
    )

    return {
        "predicted_category": predicted_category,
        "predicted_risk_level": predicted_risk,
        "possible_pathway": match["possible_pathway"],
        "documents_or_evidence": match["documents_or_evidence"],
        "possible_authority": match["possible_authority"],
        "legal_citation": match["legal_citation"],
        "safety_note": match["safety_note"],
        "urgent_flag": is_urgent,
    }

st.set_page_config(page_title="Legal Triage & Justice Navigator", page_icon="⚖️")
st.title("⚖️ Legal Triage & Justice Navigator")
st.caption("Describe your legal problem in plain language. This is triage guidance, not legal advice.")

user_text = st.text_area("Describe your problem", height=120)

if st.button("Get guidance", type="primary") and user_text.strip():
    result = predict_triage(user_text)

    if result["urgent_flag"]:
        st.error("⚠️ This may involve immediate safety risk. If you're in danger, contact local police (100) or the Women Helpline (181) now.")

    st.subheader(f"Risk level: {result['predicted_risk_level']}")
    st.write(f"**Category:** {result['predicted_category']}")
    st.markdown("### Recommended pathway")
    st.info(result["possible_pathway"])
    st.markdown("### Documents you may need")
    st.write(result["documents_or_evidence"])
    st.markdown("### Authority to approach")
    st.write(result["possible_authority"])
    st.markdown("### Legal citation")
    st.write(result["legal_citation"])
    st.caption(result["safety_note"])