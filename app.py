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
# RULE-BASED EMOTION (STRONG ✅)
# ===============================
def rule_based_emotion(text):
    happy_words = ["خوش", "خوشی", "مسرت", "شاد", "محبت", "مسکراہٹ", "خوشگوار", "خوشحال"]
    sad_words = ["اداس", "غم", "دکھ", "مایوس", "افسوس", "تنہا"]
    angry_words = ["غصہ", "غضب", "ناراض", "نفرت", "جھگڑا"]
    fear_words = ["ڈر", "خوف", "دہشت", "گھبراہٹ", "خطرہ"]

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
# SENTENCE SPLITTER (URDU)
# ===============================
def split_sentences(text):
    # Split on Urdu full stop "۔"
    sentences = [s.strip() for s in text.split("۔") if s.strip()]
    return sentences

# ===============================
# STREAMLIT UI
# ===============================
st.title("💬 Urdu Emotion Detection App")
st.write("Enter Urdu text (sentence or paragraph):")

text = st.text_area("Input Urdu Text", height=150)

if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter Urdu text.")
    else:
        cleaned = clean_text(text)
        sentences = split_sentences(cleaned)

        st.markdown("### 🔍 Sentence-wise Emotion Analysis")

        for i, sent in enumerate(sentences, 1):
            # Rule-based first
            rule_result = rule_based_emotion(sent)

            if rule_result:
                st.success(f"{i}. {sent}")
                st.info(f"Emotion: {rule_result} | Confidence: 100% (rule-based)")
            else:
                # ML fallback
                vec = vectorizer.transform([sent])
                pred = model.predict(vec)[0]
                confidence = max(model.predict_proba(vec)[0]) * 100

                st.success(f"{i}. {sent}")
                st.info(f"Emotion: {pred} | Confidence: {confidence:.2f}%")

# ===============================
# EXAMPLES
# ===============================
st.markdown("### Examples:")
st.code("""
میں بہت خوش ہوں۔
تمہارے چہرے پر مسکراہٹ ہے۔
مجھے غصہ آ رہا ہے۔
مجھے خوف محسوس ہو رہا ہے۔
وہ آج بہت اداس ہے۔
""")
