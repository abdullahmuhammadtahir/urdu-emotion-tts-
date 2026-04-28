import streamlit as st
import pickle
import re

# ===============================
# LOAD MODEL
# ===============================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ===============================
# CLEAN TEXT
# ===============================
def clean_text(text):
    text = re.sub(r"[^\u0600-\u06FF\s۔؟]", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# SENTENCE SPLITTER
# ===============================
def split_sentences(text):
    return [s.strip() for s in re.split("[۔؟]", text) if s.strip()]

# ===============================
# QUESTION DETECTION (INTENT ONLY)
# ===============================
def is_question(text):
    return any(q in text for q in ["کیا", "کیوں", "کیسے", "کب", "کہاں", "؟"])

# ===============================
# INTENT TYPE
# ===============================
def intent_type(text):
    # factual / routine → no emotion
    if any(w in text for w in ["بارش", "دفتر", "کام", "موسم"]):
        return "factual"

    # explicit emotion → certainty
    if any(w in text for w in [
        "دل خوش ہو گیا", "بہت خوش ہوں",
        "مجھے غصہ آ گیا", "بہت غصے میں ہوں",
        "میں اداس ہوں", "بہت غمگین ہوں",
        "مجھے خوف آ رہا ہے", "میں ڈر گیا ہوں"
    ]):
        return "explicit"

    # otherwise → implicit / abstract
    return "implicit"

# ===============================
# EXPLICIT EMOTION (100%)
# ===============================
def explicit_emotion(text):
    if "غصہ" in text:
        return "angry"
    if "اداس" in text or "غمگین" in text:
        return "sad"
    if "دل خوش" in text or "خوش ہوں" in text:
        return "happy"
    if "خوف" in text or "ڈر گیا" in text:
        return "fear"
    return None

# ===============================
# ABSTRACT POLARITY (IMPLICIT LOGIC)
# ===============================
def abstract_polarity(text):
    # Fear / anxiety
    fear = [
        "خطرہ", "ڈر", "خوف", "گھبرا",
        "پریشان", "سانس لینا مشکل", "بے چینی"
    ]

    # Sadness / loss
    sad = [
        "بوجھ", "مشکل", "اداس", "غم", "مایوس",
        "اکیلا", "تنہا", "دل بھاری"
    ]

    # Anger / frustration
    angry = [
        "برداشت سے باہر", "بے انصافی",
        "غصے میں", "چڑ"
    ]

    # Positive / optimism
    happy = [
        "مسکراہٹ", "خوبصورت", "اچھا",
        "بہتر", "خوشی", "امید", "سکون"
    ]

    if any(w in text for w in fear):
        return "fear"
    if any(w in text for w in angry):
        return "angry"
    if any(w in text for w in sad):
        return "sad"
    if any(w in text for w in happy):
        return "happy"

    return None

# ===============================
# STREAMLIT UI
# ===============================
st.title("💬 Urdu Emotion Detection App")
st.write("Enter Urdu sentence or paragraph:")

text = st.text_area("Input Urdu Text", height=160)

if st.button("Predict Emotion"):
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)

    st.markdown("### 🔍 Sentence‑wise Emotion Analysis")

    for i, sent in enumerate(sentences, 1):
        intent = intent_type(sent)

        # 1️⃣ FACTUAL → NEUTRAL
        if intent == "factual":
            st.success(f"{i}. {sent}")
            st.info("Emotion: neutral | Confidence: 90%")
            continue

        # 2️⃣ EXPLICIT → 100%
        if intent == "explicit":
            emo = explicit_emotion(sent)
            st.success(f"{i}. {sent}")
            st.info(f"Emotion: {emo} | Confidence: 100%")
            continue

        # 3️⃣ IMPLICIT OR QUESTION → SAME LOGIC
        if intent == "implicit" or is_question(sent):
            pol = abstract_polarity(sent)
            if pol:
                st.success(f"{i}. {sent}")
                st.info(f"Emotion: {pol} | Confidence: 30%")
                continue

            # ML fallback (neutral excluded)
            vec = vectorizer.transform([sent])
            probs = model.predict_proba(vec)[0]
            classes = model.classes_

            scores = {c: p for c, p in zip(classes, probs) if c != "neutral"}
            pred = max(scores, key=scores.get)

            st.success(f"{i}. {sent}")
            st.info(f"Emotion: {pred} | Confidence: 25%")
            continue
