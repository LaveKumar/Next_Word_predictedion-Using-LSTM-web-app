import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Configuration
st.set_page_config(page_title="Next Word Predictor", page_icon="🧠", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 24px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Description
st.title("🧠 Next Word Prediction with LSTM And Early Stopping")
st.markdown("""
Enter a partial sentence, and our trained LSTM model will predict the most likely **next word**.
""")

# Load the model
@st.cache_resource
def load_lstm_model():
    return load_model('next_word_lstm.h5')

# Load the tokenizer
@st.cache_data
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

model = load_lstm_model()
tokenizer = load_tokenizer()

# Prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "🤔 (No prediction found)"

# Input UI
st.subheader("📝 Input")
input_text = st.text_input("Enter a phrase", "To be or not to", max_chars=100)

if st.button("🚀 Predict Next Word"):
    if len(input_text.strip().split()) < 1:
        st.warning("Please enter at least one word to predict the next word.")
    else:
        with st.spinner('Predicting...'):
            max_sequence_len = model.input_shape[1] + 1
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.success(f"✨ Predicted Next Word: **{next_word}**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using TensorFlow and Streamlit")

