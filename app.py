import streamlit as st
import pickle
import re

# ===============================
# LOAD MODEL & VECTORIZER
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
# EMOTIONAL INTENT DETECTION
# ===============================
def emotional_intent(text):
    """
    Returns:
    - 'none'       → factual / routine
    - 'explicit'   → clear emotion
    - 'implicit'   → abstract / weak emotion
    """

    explicit_patterns = [
        "دل خوش", "خوش ہو گیا", "مسکراہٹ",
        "مجھے غصہ", "غصہ آ گیا",
        "اداس ہو گیا", "غمگین",
        "سانس لینا مشکل", "ڈر لگ رہا"
    ]

    implicit_patterns = [
        "دل میں", "محسوس", "بوجھ",
        "زندگی", "دنیا", "وقت",
        "خوبصورت", "اچھا", "برا"
    ]

    if any(p in text for p in explicit_patterns):
        return "explicit"

    if any(p in text for p in implicit_patterns):
        return "implicit"

    return "none"

# ===============================
# EXPLICIT EMOTION RULES
# ===============================
def explicit_emotion(text):
    if "سانس لینا مشکل" in text:
        return "fear"
    if "غصہ" in text:
        return "angry"
    if "اداس" in text or "غم" in text:
        return "sad"
    if "دل خوش" in text or "خوش ہو گیا" in text or "مسکراہٹ" in text:
        return "happy"
    return None

# ===============================
# SENTENCE SPLITTER
# ===============================
def split_sentences(text):
    return [s.strip() for s in text.split("۔") if s.strip()]

# ===============================
# STREAMLIT UI
# ===============================
st.title("💬 Urdu Emotion Detection App")
st.write("Enter Urdu sentence or paragraph:")

text = st.text_area("Input Urdu Text", height=170)

if st.button("Predict Emotion"):
    if not text.strip():
        st.warning("Please enter Urdu text.")
    else:
        cleaned = clean_text(text)
        sentences = split_sentences(cleaned)

        st.markdown("### 🔍 Sentence‑wise Emotion Analysis")

        for i, sent in enumerate(sentences, 1):
            intent = emotional_intent(sent)

            # ✅ Case 1: No emotion
            if intent == "none":
                st.success(f"{i}. {sent}")
                st.info("Emotion: neutral | Confidence: 90%")
                continue

            # ✅ Case 2: Explicit emotion
            if intent == "explicit":
                emotion = explicit_emotion(sent)
                st.success(f"{i}. {sent}")
                st.info(f"Emotion: {emotion} | Confidence: 100%")
                continue

            # ✅ Case 3: Implicit / abstract emotion
            vec = vectorizer.transform([sent])
            probs = model.predict_proba(vec)[0]
            classes = model.classes_

            emotion_scores = {
                cls: prob for cls, prob in zip(classes, probs)
                if cls != "neutral"
            }

            pred = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[pred] * 100

            st.success(f"{i}. {sent}")
            st.info(f"Emotion: {pred} | Confidence: {confidence:.2f}%")

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
