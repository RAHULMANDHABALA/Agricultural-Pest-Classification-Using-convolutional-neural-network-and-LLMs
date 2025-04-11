import streamlit as st
from PIL import Image
import io
import os
from ultralytics import YOLO
import requests
import json
import torch

# API key for OpenRouter
api_key = 'Your api key'

# Page config and title
st.set_page_config(page_title="Pest Classification & Information", layout="wide")
st.title("Agricultural Pest Classification and Information System")

# Label mapping for YOLO predictions
label_mapping = {
    'caterpillar': 'Caterpillar',
    'grasshopper': 'Locusts',  
    'slug': 'Slug',
    'snail': 'Gastropoda',
    'weevil': 'Curculionoidea'
}

# Function to predict insect from image
def predict_insect(image):
    """YOLO model prediction"""
    try:
        model = YOLO(r'C:\Users\prema\OneDrive\Desktop\ds\projects\Pest-Classification-Using-Convolutional-Neural-Network-main\runs-20240311T165335Z-001\runs\classify\train6\weights\best.pt')
        img = image.resize((255, 255))
        results = model(img)
        
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        predictions = [(label_mapping.get(names_dict[i], names_dict[i]), probs[i]) 
                     for i in range(len(names_dict))]
        
        return max(predictions, key=lambda x: x[1])
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

# Function to get pest information using OpenRouter API with Gemini model
def get_pest_info(pest_name):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://pest-classification-app.com",
                "X-Title": "Pest Classification App",
            },
            data=json.dumps({
                "model": "google/gemini-2.0-flash-thinking-exp:free",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Provide detailed information about {pest_name} pest. "
                                  f"Explain why and how the pest is harmful to plants. "
                                  f"Additionally, suggest effective prevention and control measures, including organic, chemical, "
                                  f"and integrated pest management strategies."
                    }
                ]
            })
        )
        
        # Parse the response
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return f"Sorry, I couldn't get information about {pest_name} at this time. Please try again later."
    
    except Exception as e:
        st.error(f"Error getting pest information: {str(e)}")
        return f"Sorry, I couldn't get information about {pest_name} due to an error."

# Main app layout
st.sidebar.header("Upload Image for Pest Identification")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Image processing section
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Display uploaded image
    image = Image.open(uploaded_file)
    col1.subheader("Uploaded Image")
    col1.image(image, caption="Uploaded Image", use_container_width =True)
    
    # Perform prediction
    with st.spinner('Analyzing image...'):
        pest_name, confidence = predict_insect(image)
    
    if pest_name and confidence:
        col1.success(f"Detected: {pest_name} (Confidence: {confidence:.2%})")
        
        # Get pest information
        with st.spinner(f"Getting information about {pest_name}..."):
            pest_info = get_pest_info(pest_name)
        
        # Display information
        st.subheader(f"Information about {pest_name}")
        st.write(pest_info)

# App footer
st.markdown("---")
