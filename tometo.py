import streamlit as st
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError, Image

def app():
    # Background Image Styling
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

    # Load the model
    try:
        with open('tometo.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("tometo.h5")
    except FileNotFoundError:
        st.error("Model files not found! Ensure 'tometo.json' and 'tometo.h5' are available.")
        return

    st.title('🍅 Tomato Leaf Disease Classification')
    class_name = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato_healthy']
    traslate = {
        "Tomato_Early_blight": "Spray with Bonide Liquid Copper Fungicide...",
        "Tomato_Late_blight": "Spray with Bonide Liquid Copper Fungicide...",
        "Tomato__Tomato_YellowLeaf__Curl_Virus": "Inspect plants for whitefly infestations...",
        "Tomato_healthy": "No disease detected."
    }

    # Image Upload Option
    genre = st.radio("How do you want to upload your image?", ('Browse Photos', 'Camera'))

    if genre == 'Camera':
        ImagePath = st.camera_input("Take a picture")
    else:
        ImagePath = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if ImagePath is not None:
        try:
            image_ = Image.open(ImagePath)
            st.image(image_, width=250)
        except UnidentifiedImageError:
            st.error('Unsupported file format! Only JPEG, JPG, and PNG are supported.')

        try:
            if st.button('Classify'):
                test_image = image.load_img(ImagePath, target_size=(256, 256))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0)

                result = loaded_model.predict(test_image, verbose=0)
                type_ = class_name[np.argmax(result)]

                st.subheader('Prediction: ' + type_)
                confidence = str(round(np.max(result), 4) * 100)
                st.markdown('Confidence: ' + confidence[:5] + ' %')
                st.subheader('Suggested Treatment')
                st.markdown(traslate[type_])

        except TypeError:
            st.error('There was an issue with the image or prediction.')
        except UnidentifiedImageError:
            st.error('Please upload a valid image!')
