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
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===============================
# RULE-BASED EMOTION (GUARANTEED)
# ===============================
def rule_based_emotion(text):
    if "خوش" in text:
        return "happy"
    if "اداس" in text or "غم" in text:
        return "sad"
    if "غصہ" in text or "ناراض" in text:
        return "angry"
    if "ڈر" in text or "خوف" in text:
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
مجھے غصہ آ رہا ہے
وہ بہت اداس ہے
مجھے خوف محسوس ہو رہا ہے
""")
