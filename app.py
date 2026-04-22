import streamlit as st
import pickle

# load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# title
st.title("Urdu Emotion Detection App")

# input box
text = st.text_area("Enter Urdu Text")

# prediction
if st.button("Predict Emotion"):
    if text.strip() != "":
        vect_text = vectorizer.transform([text])
        prediction = model.predict(vect_text)[0]
        confidence = model.predict_proba(vect_text).max()

        st.success(f"Emotion: {prediction}")
        st.info(f"Confidence: {round(confidence*100, 2)}%")
    else:
        st.warning("Please enter text")

# example inputs
st.write("### Examples:")
st.write("میں بہت خوش ہوں")
st.write("مجھے بہت غصہ آ رہا ہے")
st.write("میں اداس ہوں")