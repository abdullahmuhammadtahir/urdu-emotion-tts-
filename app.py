import streamlit as st
import pickle
import re

# -------------------------------
# LOAD MODEL
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s۔]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# INTENT DETECTION (NO WORD LIST EXPLOSION)
# -------------------------------
def intent_type(text):
    # Pure facts / routine
    if any(x in text for x in ["بارش", "دفتر", "کام", "موسم"]):
        return "factual"

    # Clear explicit emotion
    if any(x in text for x in ["دل خوش", "غصہ", "اداس", "سانس لینا مشکل"]):
        return "explicit"

    # Everything else is implicit / abstract
    return "implicit"

# -------------------------------
# EXPLICIT EMOTION
# -------------------------------
def explicit_emotion(text):
    if "سانس لینا مشکل" in text:
        return "fear"
    if "غصہ" in text:
        return "angry"
    if "اداس" in text:
        return "sad"
    if "دل خوش" in text:
        return "happy"
    return None

# -------------------------------
# SENTENCE SPLIT
# -------------------------------
def split_sentences(text):
    return [s.strip() for s in text.split("۔") if s.strip()]

# -------------------------------
# UI
# -------------------------------
st.title("💬 Urdu Emotion Detection App")
text = st.text_area("Enter Urdu text", height=160)

if st.button("Predict"):
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)

    for i, s in enumerate(sentences, 1):
        intent = intent_type(s)

        # 1️⃣ FACTUAL → NEUTRAL
        if intent == "factual":
            st.success(f"{i}. {s}")
            st.info("Emotion: neutral | Confidence: 90%")
            continue

        # 2️⃣ EXPLICIT → 100%
        if intent == "explicit":
            e = explicit_emotion(s)
            st.success(f"{i}. {s}")
            st.info(f"Emotion: {e} | Confidence: 100%")
            continue

        # 3️⃣ IMPLICIT / ABSTRACT → ML BEST GUESS
        vec = vectorizer.transform([s])
        probs = model.predict_proba(vec)[0]
        classes = model.classes_

        scores = {c: p for c, p in zip(classes, probs) if c != "neutral"}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top, p1 = ranked[0]
        second, p2 = ranked[1]

        # OPTION B: optimistic tie-break
        if abs(p1 - p2) < 0.08:
            pred = "happy"
            conf = 30.0
        else:
            pred = top
            conf = p1 * 100

        st.success(f"{i}. {s}")
        st.info(f"Emotion: {pred} | Confidence: {conf:.2f}%")
