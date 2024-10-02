import streamlit as st

def app():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
   
    st.subheader('About Our Website')
    st.markdown('Welcome to our platform dedicated to providing comprehensive information and effective solutions for managing leaf diseases in agricultural crops, with a focus on tomato, corn, and cotton.')
    st.subheader('Our Mission:')
    st.markdown('Our mission is to empower farmers, agricultural professionals, and enthusiasts with the knowledge and tools necessary to identify, understand, and effectively manage leaf diseases that can threaten crop health and yield.')
    st.subheader('Our Commitment')
    st.markdown('We are committed to promoting sustainable agriculture by helping farmers adopt proactive and environmentally responsible approaches to disease management. By fostering knowledge exchange and collaboration, we strive to contribute to the resilience and productivity of agricultural systems.')
    