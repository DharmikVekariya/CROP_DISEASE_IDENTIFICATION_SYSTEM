import streamlit as st
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError
from PIL import Image

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
    json_file = open('cotton.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("cotton.h5")
# print("Loaded model from disk")

    st.title('Cotton Leaf Disease Classification')
    class_name = ['bacterial_blight', 'curl_virus', 'fussarium_wilt', 'healthy']

    traslate = {"bacterial_blight": "Treatment for cotton bacterial blight involves planting disease-resistant cotton varieties, implementing strict sanitation practices to prevent the spread of the bacteria, and in severe cases, applying copper-based bactericides to manage the disease effectively..", 
    "curl_virus": "Treatment for cotton curl virus primarily focuses on controlling the whitefly vector population through insecticide application and implementing cultural practices like weed management to reduce virus reservoirs. Additionally, planting virus-resistant cotton varieties can help mitigate the impact of the disease..", 
    "fussarium_wilt": "Treatment for Fusarium wilt in cotton includes planting resistant varieties, practicing crop rotation with non-host plants, and implementing soil management techniques to reduce the pathogen's survival. Application of fungicides may also be considered in severe cases to manage the disease effectively.",
    "healthy":"There is no disease on the Cotton leaf."}

    genre = st.radio(
    "How You Want To Upload Your Image",
    ('Browse Photos', 'Camera'))

    if genre == 'Camera':
        ImagePath = st.camera_input("Take a picture")
    else:
         ImagePath = st.file_uploader("Choose a file")

# # ImagePath = st.file_uploader("Choose a file")

    if ImagePath is not None:

     try:
         image_ = Image.open(ImagePath)

         st.image(image_, width=250)

     except UnidentifiedImageError:
         st.write('Input Valid File Format !!!  [ jpeg, jpg, png only this format is supported ! ]')


     try:
        if st.button('Classify'):
             test_image = image.load_img(ImagePath, target_size=(256, 256 ))
             test_image = image.img_to_array(test_image)

             test_image = np.expand_dims(test_image, axis=0)
             result = loaded_model.predict(test_image, verbose=0)
             type_ = class_name[np.argmax(result)]
            #  tritement = traslate[class_name[np.argmax(result)]]
             st.subheader('Prediction is: ' + type_)
             Confi = str(round(np.max(result), 4) * 100)
             st.markdown('Confidence is: ' + Confi[:5] + ' %')
             st.subheader('Treatment')
             st.markdown(traslate[type_])
     except TypeError:
        st.header('Please Upload Your File !!!')

     except UnidentifiedImageError:
        st.header('Input Valid File !!!')

    