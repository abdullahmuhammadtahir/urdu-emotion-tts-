import streamlit as st
import pickle
import re

# ===============================
# LOAD MODEL & VECTORIZER
# ===============================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ===============================
# CLEAN TEXT (SAME AS TRAINING)
# ===============================
def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===============================
# RULE-BASED EMOTION (IMPROVED ✅)
# ===============================
def rule_based_emotion(text):
    happy_words = [
        "خوش", "خوشی", "مسرت", "شاد", "محبت",
        "مسکراہٹ", "خوشحال", "خوشگوار"
    ]
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
# STREAMLIT UI
# ===============================
st.title("💬 Urdu Emotion Detection App")
st.write("Enter Urdu text below:")

text = st.text_area("Input Urdu Text")

if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter Urdu text.")
    else:
        cleaned_text = clean_text(text)

        # ✅ FIRST: RULE-BASED CHECK
        rule_result = rule_based_emotion(cleaned_text)

        if rule_result:
            st.success(f"Emotion: {rule_result}")
            st.info("Confidence: 100% (rule-based)")
        else:
            # ✅ FALLBACK: ML MODEL
            vec = vectorizer.transform([cleaned_text])
            pred = model.predict(vec)[0]
            confidence = max(model.predict_proba(vec)[0]) * 100

            st.success(f"Emotion: {pred}")
            st.info(f"Confidence: {confidence:.2f}%")

# ===============================
# EXAMPLES
# ===============================
st.markdown("### Examples:")
st.code("""
میں بہت خوش ہوں
تمہارے چہرے پر مسکراہٹ ہے
مجھے تم سے محبت ہے
وہ بہت اداس ہے
مجھے غصہ آ رہا ہے
مجھے خوف محسوس ہو رہا ہے
""")
