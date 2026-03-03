import streamlit as st
import joblib
import re
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Flipkart Sentiment AI",
    page_icon="🛍️",
    layout="wide"
)

# --- CUSTOM CSS FOR IMPRESSIVE UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #2874f0;
    }
    .predict-btn {
        background-color: #fb641b !important;
        color: white !important;
        font-weight: bold;
        border-radius: 5px;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .pos-card { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .neg-card { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_assets():
    model = joblib.load('models/sentiment_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    return model, tfidf

try:
    model, tfidf = load_assets()
except Exception as e:
    st.error("⚠️ Model files not found. Please run your '02_Model_Building.ipynb' first!")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img1a.flixcart.com/www/linchpin/fk-cp-zion/img/flipkart-plus_8d85f4.png", width=150)
    st.title("Project Info")
    st.info("""
    **Objective:** Classify reviews as Positive or Negative to help sellers improve.
    
    **Algorithms Compared:**
    - Logistic Regression
    - Naive Bayes
    - Random Forest
    - Decision Tree
    - KNN
    """)
    st.write("---")
    st.markdown("Developed by: **Gaurav..**")

# --- MAIN CONTENT ---
st.markdown("<h1 style='text-align: center; color: #2874f0;'>🛒 Flipkart Sentiment Analysis AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste a customer review below to detect the emotional tone instantly.</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    user_input = st.text_area("", placeholder="Type your review here (e.g., 'The product quality is amazing and delivery was fast!')", height=150)
    
    predict_button = st.button("Analyze Sentiment", use_container_width=True)

    if predict_button:
        if user_input.strip() != "":
            with st.spinner('🤖 AI is thinking...'):
                # 1. Preprocessing
                clean_input = re.sub(r'[^a-z\s]', '', user_input.lower())
                
                # 2. Prediction
                time.sleep(0.5) # Simulate processing for better UX
                vectorized = tfidf.transform([clean_input])
                prediction = model.predict(vectorized)
                probability = model.predict_proba(vectorized) if hasattr(model, 'predict_proba') else None

                # 3. Display Result
                if prediction[0] == 1:
                    st.markdown(f"""
                        <div class="result-card pos-card">
                            <h2>Positive Sentiment 😊</h2>
                            <p>The AI is confident this customer is happy!</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-card neg-card">
                            <h2>Negative Sentiment 😠</h2>
                            <p>The AI detected dissatisfaction. Check for pain points!</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Show confidence if available
                if probability is not None:
                    st.write(f"**Confidence Level:** {max(probability[0])*100:.2f}%")
        else:
            st.warning("Please enter a review to analyze.")

# --- FOOTER ---
st.markdown("<br><hr><center>Sentiment Analysis Project | End-to-End ML Pipeline</center>", unsafe_allow_html=True)