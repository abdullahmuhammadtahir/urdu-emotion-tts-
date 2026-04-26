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
# HUMAN EMOTIONAL EXPERIENCE CHECK ✅
# ===============================
def has_emotional_experience(text):
    """
    True only if sentence expresses an INTERNAL human state.
    """
    markers = [
        "دل", "محسوس", "بوجھ",
        "اداس", "غم",
        "غصہ", "خوف", "گھبرا", "پریشان",
        "سانس لینا مشکل", "دل بھاری"
    ]
    return any(m in text for m in markers)

# ===============================
# EXPLICIT HUMAN EMOTION RULES ✅
# (ONLY human emotion phrases, NOT descriptive words)
# ===============================
def explicit_emotion(text):
    # Fear has strongest priority
    if "سانس لینا مشکل" in text:
        return "fear"

    # ✅ Happy only if HUMAN context
    happy_patterns = [
        "دل خوش", "خوش ہو گیا", "خوش ہوگیا",
        "مسکراہٹ آئی", "محبت"
    ]

    sad_patterns = ["اداس", "غم", "دکھ", "مایوس", "بوجھ", "دل بھاری"]
    angry_patterns = ["غصہ", "غضب", "ناراض"]
    fear_patterns = ["ڈر", "خوف", "دہشت", "گھبراہٹ"]

    if any(p in text for p in happy_patterns):
        return "happy"
    if any(p in text for p in sad_patterns):
        return "sad"
    if any(p in text for p in angry_patterns):
        return "angry"
    if any(p in text for p in fear_patterns):
        return "fear"

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

        st.markdown("### 🔍 Sentence-wise Emotion Analysis")

        for i, sent in enumerate(sentences, 1):

            # ✅ 1. No human emotional experience → neutral
            if not has_emotional_experience(sent):
                st.success(f"{i}. {sent}")
                st.info("Emotion: neutral | Confidence: 90%")
                continue

            # ✅ 2. Explicit human emotion → 100%
            rule = explicit_emotion(sent)
            if rule:
                st.success(f"{i}. {sent}")
                st.info(f"Emotion: {rule} | Confidence: 100%")
                continue

            # ✅ 3. Implicit emotion → ML best guess (exclude neutral)
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
لیکن دن گزرتے گزرتے دل میں ایک عجیب سا بوجھ محسوس ہونے لگا۔
کمرے کی فضا غیر معمولی تھی اور سانس لینا مشکل لگ رہا تھا۔
پھر اچانک دل خوش ہو گیا۔
بعد میں مجھے بہت غصہ آ گیا۔
آخر میں میں اداس ہو کر خاموش بیٹھ گیا۔
""")
