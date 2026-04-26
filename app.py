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
# RULE-BASED EMOTION (WITH CONTEXT GUARD ✅)
# ===============================
def rule_based_emotion(text):
    # ✅ factual / weather / routine → NOT emotion
    neutral_context = [
        "موسم", "بارش", "سردی", "گرمی", "دھوپ", "بادل",
        "آج", "کل", "صبح", "شام", "رات",
        "دفتر", "کام", "سفر"
    ]

    if any(w in text for w in neutral_context):
        return None

    # ✅ explicit emotion words
    happy_words = ["خوش", "خوشی", "مسرت", "شاد", "محبت", "مسکراہٹ", "خوشحال"]
    sad_words = ["اداس", "غم", "دکھ", "مایوس", "افسوس", "تنہا", "بوجھ", "دل بھاری"]
    angry_words = ["غصہ", "غضب", "ناراض", "نفرت", "جھگڑا"]
    fear_words = ["ڈر", "خوف", "دہشت", "گھبراہٹ", "خطرہ", "سانس لینا مشکل"]

    if any(w in text for w in happy_words):
        return "happy"
    if any(w in text for w in sad_words):
        return "sad"
    if any(w in text for w in angry_words):
        return "angry"
    if any(w in text for w in fear_words):
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

text = st.text_area("Input Urdu Text", height=160)

if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter Urdu text.")
    else:
        cleaned = clean_text(text)
        sentences = split_sentences(cleaned)

        st.markdown("### 🔍 Sentence-wise Emotion Analysis")

        for i, sent in enumerate(sentences, 1):
            # 1️⃣ Rule-based first
            rule_result = rule_based_emotion(sent)

            if rule_result:
                st.success(f"{i}. {sent}")
                st.info(f"Emotion: {rule_result} | Confidence: 100% (rule-based)")
            else:
                # 2️⃣ ML fallback with controlled neutral
                vec = vectorizer.transform([sent])
                probs = model.predict_proba(vec)[0]
                classes = model.classes_

                neutral_prob = probs[list(classes).index("neutral")]

                emotion_scores = {
                    cls: prob for cls, prob in zip(classes, probs) if cls != "neutral"
                }

                best_emotion = max(emotion_scores, key=emotion_scores.get)
                best_emotion_prob = emotion_scores[best_emotion]

                # ✅ FINAL DECISION RULE
                if best_emotion_prob > 0.20:
                    pred = best_emotion
                    confidence = best_emotion_prob * 100
                else:
                    pred = "neutral"
                    confidence = neutral_prob * 100

                st.success(f"{i}. {sent}")
                st.info(f"Emotion: {pred} | Confidence: {confidence:.2f}%")

# ===============================
# EXAMPLES
# ===============================
st.markdown("### Examples:")
st.code("""
آج بارش ہو رہی ہے۔
آج صبح موسم خوشگوار تھا اور بارش ہو رہی تھی۔
دل میں ایک عجیب سا بوجھ محسوس ہو رہا تھا۔
سانس لینا مشکل لگ رہا تھا۔
میں بہت خوش ہوں۔
مجھے غصہ آ رہا ہے۔
""")
