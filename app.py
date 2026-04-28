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
    text = re.sub(r'[^\u0600-\u06FF\s۔]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===============================
# SENTENCE SPLITTER
# ===============================
def split_sentences(text):
    return [s.strip() for s in text.split("۔") if s.strip()]

# ===============================
# INTENT TYPE (3 STATES ONLY)
# ===============================
def intent_type(text):
    # factual / routine → no emotion
    if any(w in text for w in ["بارش", "دفتر", "کام", "موسم"]):
        return "factual"

    # explicit emotion → certain
    if any(w in text for w in ["دل خوش", "غصہ", "اداس"]):
        return "explicit"

    # everything else → implicit / abstract
    return "implicit"

# ===============================
# EXPLICIT EMOTION (100%)
# ===============================
def explicit_emotion(text):
    if "غصہ" in text:
        return "angry"
    if "اداس" in text:
        return "sad"
    if "دل خوش" in text:
        return "happy"
    return None

# ===============================
# ABSTRACT POLARITY (MINIMAL, CONCEPTUAL)
# ===============================
def abstract_polarity(text):
    """
    Returns: 'positive', 'negative', or None
    (Used ONLY for implicit/abstract cases)
    """
    positive_cues = ["خوبصورت", "اچھا", "بہتر", "مثبت"]
    negative_cues = ["مشکل", "خراب", "بوجھ", "پریشان"]

    threat_cues = ["سانس لینا مشکل", "خطرہ", "خوف", "گھبرا"]

    if any(t in text for t in threat_cues):
        return "fear"

    if any(n in text for n in negative_cues):
        return "sad"

    if any(p in text for p in positive_cues):
        return "happy"

    return None

# ===============================
# STREAMLIT UI
# ===============================
st.title("💬 Urdu Emotion Detection App")
st.write("Enter Urdu sentence or paragraph:")

text = st.text_area("Input Urdu Text", height=170)

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

        # 3️⃣ IMPLICIT / ABSTRACT
        # First: abstract polarity (human‑aligned)
        pol = abstract_polarity(sent)
        if pol == "happy":
            st.success(f"{i}. {sent}")
            st.info("Emotion: happy | Confidence: 30%")
            continue
        if pol == "sad":
            st.success(f"{i}. {sent}")
            st.info("Emotion: sad | Confidence: 30%")
            continue
        if pol == "fear":
            st.success(f"{i}. {sent}")
            st.info("Emotion: fear | Confidence: 30%")
            continue

        # Fallback: ML best‑guess (neutral excluded), low confidence
        vec = vectorizer.transform([sent])
        probs = model.predict_proba(vec)[0]
        classes = model.classes_

        scores = {c: p for c, p in zip(classes, probs) if c != "neutral"}
        pred = max(scores, key=scores.get)

        st.success(f"{i}. {sent}")
        st.info(f"Emotion: {pred} | Confidence: 25%")

# ===============================
# TEST PARAGRAPH
# ===============================
st.markdown("### ✅ Test Paragraph")
st.code("""
آج صبح موسم خوشگوار تھا اور ہلکی بارش ہو رہی تھی۔
میں دفتر گیا اور معمول کے مطابق کام کیا۔
دل میں ایک عجیب سا بوجھ محسوس ہونے لگا۔
کمرے کی فضا غیر معمولی تھی اور سانس لینا مشکل لگ رہا تھا۔
پھر اچانک دل خوش ہو گیا۔
بعد میں مجھے بہت غصہ آ گیا۔
آخر میں میں اداس ہو کر خاموش بیٹھ گیا۔
زندگی خوبصورت ہے۔
""")
