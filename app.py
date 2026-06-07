import streamlit as st
import joblib
import numpy as np
import pandas as pd

vectorizer = joblib.load('vectorizer_spam.pkl')
model = joblib.load('spam_model.pkl')
st.set_page_config(layout = 'wide')
st.sidebar.image('spam_logo.jpg')
st.sidebar.title('About Project')
st.sidebar.write('Objective of this project is to predict a message ')
st.sidebar.title('Libraries')
st.sidebar.markdown('''
- sklearn
- pandas
- numpy''')

st.sidebar.title('Cloud')
st.sidebar.markdown('Streamlit')

st.sidebar.title('Contact')
st.sidebar.markdown('7417709971')
    
st.markdown("""
<style>
.modern-header {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 0px;
    text-align: center;
    margin-bottom: 10px;
    background: linear-gradient(
        135deg,
        rgba(255, 140, 66, 0.15),
        rgba(255, 99, 132, 0.12),
        rgba(255, 206, 86, 0.10)
    );
    box-shadow: 0 10px 40px rgba(0,0,0,0.15);
}

.modern-header h1 {
    font-size: 52px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #FF8C42, #FF4D6D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}

.modern-header p {
    color: #A0A0A0;
    font-size: 18px;
    margin-top: 10px;
    font-weight: 400;
}
</style>

<div class="modern-header">
    <h1>Spam Classifier</h1>
    <p>Discover customer emotions, trends & food insights using AI</p>
</div>
""", unsafe_allow_html=True)
st.write('\n')

#col1,col2 = st.columns(2)
col1,col2 = st.columns([.5,.5])
with col1:
    st.markdown("""
    <style>
    .Single-header {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 24px;
        padding: 0px;
        text-align: center;
        margin-bottom: 10px;
        background: linear-gradient(
            135deg,
            rgba(255, 140, 66, 0.15),
            rgba(255, 99, 132, 0.12),
            rgba(255, 206, 86, 0.10)
        );
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    
    .Single-header h1 {
        font-size: 25px;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #FF8C42, #FF4D6D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    </style>
    <div class="Single-header">
        <h1>Predict Single Message</h1>
    </div>
    """, unsafe_allow_html=True)
    
    reviews = st.text_input('**Enter Message**')
    if st.button('Predict'):
        X_test = vectorizer.transform([reviews])
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)
        if pred[0]=='spam':
            st.error('**Message Type  = Spam**')
            st.warning(f'Confidence Score {prob[0][0]:.2f}')
        else:
            st.success('**Message Type = Ham**')
            st.warning(f'Confidence Score {prob[0][1]:.2f}')


with col2:
    st.markdown("""
    <style>
    .Bulk-header {
        background: rgba(150,150,150,0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(330,330,255,0.12);
        border-radius: 24px;
        padding: 0px;
        text-align: center;
        margin-bottom: 10px;
        background: linear-gradient(
            135deg,
            rgba(505, 150, 66, 0.15),
            rgba(505, 100, 132, 0.12),
            rgba(505, 200, 86, 0.10)
        );
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    
    .Bulk-header h1 {
        font-size: 25px;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #FF8C42, #FF4D6D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    </style>
    <div class="Bulk-header">
        <h1>Predict Bulk Message</h1>
    </div>
    """, unsafe_allow_html=True)
    
    file = st.file_uploader('Upload Excel/CSV File',type = ['csv','txt'])
    if file:
        df = pd.read_csv(file,header=None, names=['Msg'])
        placeholder = st.empty()
        placeholder = st.dataframe(df)
        if st.button('Bulk Predict'):
            X_test = vectorizer.transform(df.Review)
            pred = model.predict(X_test)
            prob=model.predict_proba(X_test)
            df['Message Type'] = pred
            df['Confidence'] = np.max(prob, axis = 1)
            placeholder.dataframe(df)
