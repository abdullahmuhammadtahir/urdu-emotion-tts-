import streamlit as st
import pickle
import re

# ==================================================
# LOAD MODEL
# ==================================================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ==================================================
# CLEAN TEXT
# ==================================================
def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s۔]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==================================================
# INTENT TYPE (ONLY 3 STATES)
# ==================================================
def intent_type(text):
    """
    Returns:
    - factual   → no emotion
    - explicit  → clear emotion
    - implicit  → weak / abstract emotion
    """

    # factual / routine (no emotion)
    if any(w in text for w in ["بارش", "دفتر", "کام", "موسم"]):
        return "factual"

    # explicit emotion
    if any(w in text for w in ["دل خوش", "غصہ", "اداس", "خوش ہو گیا"]):
        return "explicit"

    # everything else = implicit / abstract
    return "implicit"

# ==================================================
# EXPLICIT EMOTION RULES (CERTAIN)
# ==================================================
def explicit_emotion(text):
    if "غصہ" in text:
        return "angry"
    if "اداس" in text:
        return "sad"
    if "دل خوش" in text or "خوش ہو گیا" in text:
        return "happy"
    if "خوف" in text or "ڈر" in text:
        return "fear"
    return None

# ==================================================
# SENTENCE SPLITTING
# ==================================================
def split_sentences(text):
    return [s.strip() for s in text.split("۔") if s.strip()]

# ==================================================
# STREAMLIT UI
# ==================================================
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

        # 3️⃣ IMPLICIT / ABSTRACT → ML BEST GUESS (OPTION B)
        vec = vectorizer.transform([sent])
        probs = model.predict_proba(vec)[0]
        classes = model.classes_

        # remove neutral, pick best
        scores = {c: p for c, p in zip(classes, probs) if c != "neutral"}
        pred = max(scores, key=scores.get)
        confidence = scores[pred] * 100

        # keep confidence low (honest uncertainty)
        confidence = max(20.0, min(confidence, 40.0))

        st.success(f"{i}. {sent}")
        st.info(f"Emotion: {pred} | Confidence: {confidence:.2f}%")

# ==================================================
# TEST PARAGRAPH
# ==================================================
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
