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
   
    json_file = open('corn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("corn.h5")
# print("Loaded model from disk")

    st.title('🌽Corn Leaf Disease Classification')
    class_name = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

    traslate = {"Blight": "Treatment for corn blight typically involves a combination of cultural practices like crop rotation and proper spacing, along with targeted use of fungicides when necessary. Biological controls and maintaining plant health through balanced nutrition and water management also play crucial roles in managing the disease effectively.", 
    "Common_Rust": "Treatment for common rust in corn often involves planting resistant varieties, applying fungicides when necessary, and implementing cultural practices such as crop rotation and timely removal of infected debris to minimize disease spread.", 
    "Gray_Leaf_Spot": "Treatment for gray leaf spot in corn typically includes planting resistant varieties, implementing crop rotation, and applying fungicides as needed to control the spread of the disease and minimize yield losses.",
    "Healthy":"There is no disease on the Corn leaf."}

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
             test_image = image.load_img(ImagePath, target_size=(256, 256))
             test_image = image.img_to_array(test_image)

             test_image = np.expand_dims(test_image, axis=0)
             result = loaded_model.predict(test_image, verbose=0)
             type_ = class_name[np.argmax(result)]
             st.subheader('Prediction is: ' + type_)
             Confi = str(round(np.max(result), 4) * 100)
             st.markdown('Confidence is: ' + Confi[:5] + ' %')
             st.subheader('Treatment')
             st.markdown(traslate[type_])

     except TypeError:
        st.header('Please Upload Your File !!!')

     except UnidentifiedImageError:
        st.header('Input Valid File !!!')

    