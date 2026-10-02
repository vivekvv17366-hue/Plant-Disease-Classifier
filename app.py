
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd

from PIL import Image



st.set_page_config(
    page_title="PlantCare AI | Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

.stApp {
    background-color: #F6F9F6;
}

header {
    visibility: hidden;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Main heading */

.hero-title {
    font-size: 38px;
    font-weight: 750;
    color: #174D35;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 16px;
    color: #65776B;
    margin-bottom: 25px;
}

/* Section heading */

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #174D35;
    margin-top: 15px;
    margin-bottom: 12px;
}

/* Cards */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E1EAE2;
    border-radius: 14px;
    padding: 15px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #173D2D;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

.sidebar-heading {
    font-size: 25px;
    font-weight: 750;
    color: white;
}

.sidebar-description {
    color: #D0E4D5;
    font-size: 14px;
}

/* Buttons */

.stButton > button {
    background-color: #23834F;
    color: white;
    border: none;
    border-radius: 9px;
    height: 48px;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover {
    background-color: #17683D;
    color: white;
    border: none;
}

/* Metric */

div[data-testid="stMetric"] {
    background-color: #F0F7F1;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #23834F;
}

/* Footer */

.footer {
    text-align: center;
    color: #78877B;
    font-size: 13px;
    padding-top: 25px;
}

</style>
""", unsafe_allow_html=True)




@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "plant_disease_analyser.keras"
    )


@st.cache_data
def load_classes():

    with open("class_names.json", "r") as file:
        return json.load(file)


try:

    model = load_model()
    class_names = load_classes()

except Exception as e:

    st.error(
        "Unable to load the model or class names. "
        "Please check your project files."
    )

    st.code(str(e))
    st.stop()




def format_class_name(class_name):

    name = class_name.replace("Tomato_", "")
    name = name.replace("Tomato___", "")
    name = name.replace("_", " ")

    return name.title()




with st.sidebar:

    st.markdown(
        '<div class="sidebar-heading">🌿 PlantCare AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sidebar-description">'
        'Intelligent Plant Disease Classification System'
        '</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Select Page",
        ["Disease Detection", "About Project"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model Information")

    st.markdown("""
    **Architecture:** MobileNetV2

    **Framework:** TensorFlow

    **Task:** Image Classification

    **Input Size:** 224 × 224

    **Plant:** Tomato

    **Categories:** 4
    """)

    st.divider()

    st.caption("Developed with Python & Streamlit")




if page == "About Project":

    st.markdown(
        '<div class="hero-title">About PlantCare AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'An AI-powered application for identifying tomato leaf diseases.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Project Overview")

    st.write("""
    PlantCare AI is a Deep Learning-based image classification
    application designed to identify tomato leaf conditions
    from uploaded images.

    The application uses Transfer Learning with MobileNetV2
    to classify images into four categories.
    """)

    st.markdown("### Supported Categories")

    for class_name in class_names:

        st.success(format_class_name(class_name))

    st.markdown("### Technologies Used")

    st.write("""
    - Python
    - TensorFlow and Keras
    - MobileNetV2
    - NumPy
    - Pandas
    - Streamlit
    - Pillow
    """)

    st.info(
        "This application is an educational prototype. "
        "Its predictions should not replace professional "
        "agricultural diagnosis."
    )



else:

    st.markdown(
        '<div class="hero-title">'
        'Plant Disease Detection'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Upload a tomato leaf image and let AI analyze its condition.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

   

    st.markdown(
        '<div class="section-title">Image Analysis</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1], gap="large")

    with left:

        with st.container(border=True):

            st.markdown("### Upload Leaf Image")

            uploaded_file = st.file_uploader(
                "Choose an image from your device",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of a tomato leaf."
            )

            st.caption(
                "Supported formats: JPG, JPEG and PNG"
            )

            st.markdown(
                "**Tip:** Use a clear image with good lighting "
                "for better prediction results."
            )

    with right:

        with st.container(border=True):

            st.markdown("### Image Preview")

            if uploaded_file is not None:

                try:

                    image = Image.open(
                        uploaded_file
                    ).convert("RGB")

                    st.image(
                        image,
                        use_container_width=True
                    )

                except Exception:

                    st.error("Unable to read this image.")
                    image = None

            else:

                st.markdown(
                    """
                    <div style="
                        background:#F0F6F1;
                        border:2px dashed #A9C6AF;
                        border-radius:12px;
                        padding:65px 10px;
                        text-align:center;
                    ">
                        <div style="font-size:45px;">🌱</div>
                        <div style="color:#51745B;font-weight:600;">
                            Your image preview will appear here
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                image = None

    st.write("")

   

    if uploaded_file is not None and image is not None:

        predict_button = st.button(
            "🔍  Analyze Plant Image",
            use_container_width=True
        )

        if predict_button:

            with st.spinner(
                "AI model is analyzing your image..."
            ):

                try:

                    
                    processed_image = image.resize(
                        (224, 224)
                    )

                    
                    img_array = np.array(
                        processed_image
                    )

                  
                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    
                    predictions = model.predict(
                        img_array,
                        verbose=0
                    )

                  
                    probabilities = predictions[0]

                    
                    predicted_index = np.argmax(
                        probabilities
                    )

                    predicted_class = class_names[
                        predicted_index
                    ]

                    confidence = float(
                        probabilities[predicted_index]
                    ) * 100


                    st.divider()

                    st.markdown(
                        '<div class="section-title">'
                        'Analysis Results'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    result_left, result_right = st.columns(
                        [1, 1],
                        gap="large"
                    )

                    with result_left:

                        with st.container(border=True):

                            st.markdown(
                                "### Predicted Condition"
                            )

                            st.success(
                                format_class_name(
                                    predicted_class
                                )
                            )

                            st.metric(
                                "Model Confidence",
                                f"{confidence:.2f}%"
                            )

                            st.progress(
                                confidence / 100
                            )

                            st.caption(
                                "Confidence represents the model's "
                                "predicted probability, not guaranteed "
                                "diagnostic accuracy."
                            )

                    with result_right:

                        with st.container(border=True):

                            st.markdown(
                                "### Classification Summary"
                            )

                            st.markdown(
                                f"""
                                **Plant:** Tomato

                                **Predicted Category:** 
                                {format_class_name(predicted_class)}

                                **Number of Classes:** 
                                {len(class_names)}

                                **Model:** MobileNetV2
                                """
                            )

                            if "healthy" in predicted_class.lower():

                                st.success(
                                    "The model classified this image "
                                    "as a healthy tomato leaf."
                                )

                            else:

                                st.warning(
                                    "The model detected a disease "
                                    "category. Consider further "
                                    "verification."
                                )

                    

                    st.markdown(
                        '<div class="section-title">'
                        'Class Probability Analysis'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    probability_data = pd.DataFrame({
                        "Disease": [
                            format_class_name(name)
                            for name in class_names
                        ],

                        "Probability (%)": [
                            float(value) * 100
                            for value in probabilities
                        ]
                    })

                    probability_data = probability_data.sort_values(
                        by="Probability (%)",
                        ascending=False
                    )

                    with st.container(border=True):

                        st.markdown(
                            "#### Prediction Distribution"
                        )

                        st.bar_chart(
                            probability_data.set_index("Disease"),
                            horizontal=True,
                            x_label="Probability (%)"
                        )

                        for _, row in probability_data.iterrows():

                            st.write(
                                f"**{row['Disease']}** — "
                                f"{row['Probability (%)']:.2f}%"
                            )

                     
                     
                     

                    st.info(
                        "Note: This application is designed for "
                        "educational purposes. Model predictions "
                        "may be inaccurate on real-world images. "
                        "Please consult an agricultural expert "
                        "before making treatment decisions."
                    )

                except Exception as e:

                    st.error(
                        "An error occurred during prediction."
                    )

                    st.code(str(e))



st.markdown(
    """
    <div class="footer">
        <hr>
        <b>PlantCare AI</b> | AI-Powered Plant Disease Classification
        <br>
        Built with Python, TensorFlow & Streamlit
        <br><br>
        © 2026 PlantCare AI | Educational Project
    </div>
    """,
    unsafe_allow_html=True
)