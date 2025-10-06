import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide"
)

# --- Helper Function for Softmax ---
def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# --- TFLite Model Loading ---
@st.cache_resource
def load_tflite_model(model_path):
    """Loads a TFLite model and allocates tensors."""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except ValueError:
        st.error(f"Error: Unable to load the model. Make sure 'MCAR.tflite' is in the same directory as your app.py file.")
        return None

# Provide the path to your .tflite file
TFLITE_MODEL_PATH = "waste_classifier_v2.tflite"
interpreter = load_tflite_model(TFLITE_MODEL_PATH)

# Get input and output tensor details if the interpreter loaded successfully
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# --- Class Labels ---
CLASS_LABELS = {0: "Organic", 1: "Recyclable"}

# --- Did You Know Facts ---
DID_YOU_KNOW_FACTS = [
    "Recycling one aluminum can saves enough energy to run a TV for three hours.",
    "Composting organic waste can reduce greenhouse gas emissions and enrich soil.",
    "Plastic bags can take up to 1,000 years to decompose in landfills.",
    "Glass bottles can be recycled an infinite number of times without losing quality.",
    "About 80% of what Americans throw away is recyclable, but only 28% actually gets recycled.",
    "The energy saved by recycling one glass bottle could power a compact fluorescent light bulb for 20 hours.",
    "Food waste is a huge problem; roughly one-third of the food produced globally for human consumption is lost or wasted.",
    "Recycling paper saves trees, water, and energy. It takes 70% less energy to make paper from recycled materials than from raw materials.",
    "E-waste (electronic waste) is the fastest-growing waste stream in the world. Always try to repair or properly recycle old electronics."
]


# --- Sidebar ---
with st.sidebar:
    st.title("ℹ️ About This Project")
    st.info(
        """
        This application leverages a machine learning model to classify waste as either **Organic** or **Recyclable**.
        """
    )

    st.header("Why is Waste Classification Important?")
    st.write(
        """
        Proper waste segregation is a critical step towards a sustainable future. By separating organic waste from recyclables, we can:
        - **Reduce Landfill Waste**: Organic waste can be composted to create nutrient-rich soil.
        - **Conserve Resources**: Recycling materials like plastic, glass, and metal saves energy and raw materials.
        - **Protect Ecosystems**: Minimizing landfill usage prevents soil and water pollution.
        """
    )

    st.header("🚀 Tech Stack")
    st.write(
        """
        - **Model**: TensorFlow Lite (`.tflite`)
        - **Framework**: Streamlit
        - **Language**: Python
        - **Libraries**: TensorFlow, PIL, NumPy
        """
    )
    st.write("---")
    st.markdown("Developed by **Shubham Patil**")


# --- Main Application ---
st.title("♻️ Smart Waste Classifier")
st.markdown("Upload an image, and the model will predict if the waste is **Organic** or **Recyclable**.")

# --- Image Uploader ---
uploaded_file = st.file_uploader(
    "Choose an image to classify...",
    type=["jpg", "png", "jpeg"]
)

# --- Image Processing and Prediction Function ---
def process_and_predict(image_data):
    """
    Processes an image and returns the prediction, highest confidence, and all class probabilities.
    """
    input_shape = (input_details[0]['shape'][1], input_details[0]['shape'][2])
    input_dtype = input_details[0]['dtype']

    img = Image.open(image_data).convert('RGB')
    img = img.resize(input_shape)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0).astype(input_dtype)

    if input_dtype == np.float32:
        img_array = img_array / 255.0

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    probabilities = softmax(predictions)
    predicted_class_index = np.argmax(probabilities)
    predicted_label = CLASS_LABELS[predicted_class_index]
    highest_confidence = np.max(probabilities) * 100

    return predicted_label, highest_confidence, probabilities

# --- Display Results ---
if uploaded_file and interpreter:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(uploaded_file, caption="🖼️ Your Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🧠 Analyzing the image..."):
            time.sleep(1)
            predicted_label, highest_confidence, all_probabilities = process_and_predict(uploaded_file)

        st.success(f"**Prediction: This looks like {predicted_label} waste!**")
        st.write(f"**Overall Confidence:** `{highest_confidence:.2f}%`")
        
        st.markdown("---")
        st.markdown("### 🧠 Confidence Scores:")
        for i, prob in enumerate(all_probabilities):
            label = CLASS_LABELS[i]
            st.markdown(f"- **{label}:** `{prob * 100:.2f}%`")

        st.markdown("---")
        if predicted_label == "Organic":
            st.subheader("💡 Tips for Organic Waste")
            st.markdown(
                """
                - **Compost it!** Organic waste like fruit peels, vegetable scraps, and coffee grounds are great for composting.
                - **Start a compost bin** in your backyard or use a community composting service.
                - **Avoid meat and dairy** in home compost piles as they can attract pests.
                """
            )
            # --- CHANGED: Image replaced with a video about composting ---
            st.video("https://www.youtube.com/watch?v=Q5s4n9r-JGU")
        else: # Recyclable
            st.subheader("💡 Tips for Recyclable Waste")
            st.markdown(
                """
                - **Rinse before you recycle!** Clean out food containers before placing them in the recycling bin.
                - **Check local guidelines.** Different municipalities accept different types of plastics and materials.
                - **Don't bag your recyclables** unless your local service specifically asks you to.
                """
            )
            # --- CHANGED: Image replaced with a video about recycling ---
            st.video("https://www.youtube.com/watch?v=6jQ7y_qQYUA")

# --- Dynamic "Did You Know?" Section ---
st.markdown("---")
st.header("🌍 Did You Know?")
if uploaded_file:
    st.info(random.choice(DID_YOU_KNOW_FACTS))
else:
    st.info("Upload an image to learn an interesting fact about waste management!")