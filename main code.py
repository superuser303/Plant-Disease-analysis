import gdown
import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from pathlib import Path
import time
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
from fastapi import FastAPI, UploadFile, File
import uvicorn
import threading
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import matplotlib.pyplot as plt

# Debug statements
print(f"TensorFlow version: {tf.__version__}")
from tensorflow.keras.preprocessing import image
print("Imported tensorflow.keras.preprocessing.image successfully")

# --- Session State Initialization ---
if 'model_plant' not in st.session_state:
    st.session_state['model_plant'] = None  # TensorFlow model for plant identification
if 'model_disease' not in st.session_state:
    st.session_state['model_disease'] = None  # PyTorch model for disease detection
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False
if 'loading' not in st.session_state:
    st.session_state['loading'] = False

# --- Page Configuration ---
st.set_page_config(
    page_title="MediPlant AI | Plant Identification & Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Medicinal Plant Data ---
class_labels = ["Aloevera", "Amla", "Amruta_Balli", "Arali", "Ashoka", "Ashwagandha", "Avocado", "Bamboo", "Basale",
                "Betel", "Betel_Nut", "Brahmi", "Castor", "Curry_Leaf", "Doddapatre", "Ekka", "Ganike", "Guava",
                "Geranium", "Henna", "Hibiscus", "Honge", "Insulin", "Jasmine", "Lemon", "Lemon_grass", "Mango",
                "Mint", "Nagadali", "Neem", "Nithyapushpa", "Nooni", "Pappaya", "Pepper", "Pomegranate",
                "Raktachandini", "Rose", "Sapota", "Tulasi", "Wood_sorel"]

methods_of_preparation = {
    "Aloevera": "Slit the leaf of an aloe plant lengthwise and remove the gel from the inside, or use a commercial preparation.",
    "Amla": "Eating raw amla and candies or taking amla powder with lukewarm water",
    "Amruta_Balli": "Make a decoction or powder from the stems of Giloy. It is known for its immunomodulatory properties.",
    "Arali": "Various parts like the root bark, leaves, and fruit are used for medicinal purposes. It can be consumed in different forms, including as a decoction.",
    "Ashoka": "Different parts like the bark are used. It's often prepared as a decoction for menstrual and uterine health.",
    "Ashwagandha": "The root is commonly used, and it can be consumed as a powder, capsule, or as a decoction. It is an adaptogen known for its stress-relieving properties.",
    "Avocado": "The fruit is consumed for its nutritional benefits, including healthy fats and vitamins.",
    "Bamboo": "Bamboo shoots are consumed, and some varieties are used in traditional medicine.",
    "Basale": "The leaves are consumed as a leafy vegetable. It's rich in vitamins and minerals.",
    "Betel": "Chewing betel leaves with areca nut is a common practice in some cultures. It's believed to have digestive and stimulant properties.",
    "Betel_Nut": "The nut is often chewed with betel leaves. However, excessive consumption is associated with health risks.",
    "Brahmi": "The leaves are used for enhancing cognitive function. It can be consumed as a powder, in capsules, or as a fresh juice.",
    "Castor": "Castor oil is extracted from the seeds and used for various medicinal and cosmetic purposes.",
    "Curry_Leaf": "Curry leaves are used in cooking for flavor, and they are also consumed for their potential health benefits.",
    "Doddapatre": "The leaves are used in traditional medicine, often as a poultice for skin conditions.",
    "Ekka": "Various parts may be used in traditional medicine. It's important to note that some species of Ekka may have toxic components, and proper identification is crucial.",
    "Ganike": "The leaves are used in traditional medicine, often as a remedy for respiratory issues.",
    "Guava": "Guava fruit is consumed for its high vitamin C content and other health benefits.",
    "Geranium": "Geranium oil is extracted from the leaves and stems and is used in aromatherapy and skincare.",
    "Henna": "Henna leaves are dried and powdered to make a paste used for hair coloring and as a natural dye.",
    "Hibiscus": "Hibiscus flowers are commonly used to make teas, infusions, or extracts. They are rich in antioxidants and can be beneficial for skin and hair health.",
    "Honge": "Various parts of the tree are used traditionally, including the bark and seeds. It's often used for its anti-inflammatory properties.",
    "Insulin": "The leaves are used for their potential blood sugar-lowering properties. They can be consumed fresh or as a tea.",
    "Jasmine": "Jasmine flowers are often used to make aromatic teas or essential oils, known for their calming effects.",
    "Lemon": "Lemon juice is a common remedy for digestive issues, and the fruit is rich in vitamin C. The peel can be used for its essential oil.",
    "Lemon_grass": "Lemon grass is used to make teas and infusions, known for its soothing and digestive properties.",
    "Mango": "Mango fruit is consumed fresh and is rich in vitamins and minerals. Some parts, like the leaves, are also used in traditional medicine.",
    "Mint": "Mint leaves are commonly used to make teas, infusions, or added to dishes for flavor. It's known for its digestive properties.",
    "Nagadali": "Different parts of the plant are used traditionally. It's often prepared as a decoction.",
    "Neem": "Various parts of the neem tree are used, including leaves, bark, and oil. It's known for its antibacterial and antifungal properties.",
    "Nithyapushpa": "The flowers are used in traditional medicine, often for their calming effects.",
    "Nooni": "Different parts of the tree are used traditionally. The oil extracted from the seeds is used for various purposes.",
    "Pappaya": "Consume fruit; leaves traditionally used for certain health benefits.",
    "Pepper": "Spice for flavor; potential digestive and antimicrobial properties.",
    "Pomegranate": "Eat seeds or drink juice for antioxidant benefits.",
    "Raktachandini": "Traditional uses; some parts may be toxic, use caution.",
    "Rose": "Make tea or use petals for calming and aromatic effects.",
    "Sapota": "Consume fruit for its sweet taste and nutritional content.",
    "Tulasi": "Make tea or use leaves for immune support.",
    "Wood_sorel": "Make tea or use leaves for immune support. Use leaves in salads; some varieties contain oxalic acid."
}

use_of_medicine = {
    "Lemon_grass": [
        "Calms the nervous system and reduces anxiety.",
        "Aids digestion and relieves bloating.",
        "Has anti-inflammatory and pain-relieving properties."
    ],
    "Mango": [
        "Rich in vitamins A and C, boosts immune health.",
        "Aids in digestion and improves skin condition."
    ],
    "Mint": [
        "Soothes the digestive system and relieves nausea.",
        "Has a refreshing effect and clears nasal congestion.",
        "Acts as a natural breath freshener."
    ],
    "Nagadali": [
        "Anti-inflammatory properties, used in pain relief.",
        "Supports traditional medicinal practices."
    ],
    "Neem": [
        "Antibacterial and antifungal, supports skin health.",
        "Boosts immunity and helps purify the blood."
    ],
    "Nithyapushpa": [
        "Calming effect, used in stress and anxiety relief.",
        "Promotes mental well-being in traditional medicine."
    ],
    "Nooni": [
        "Anti-inflammatory properties, used for pain relief.",
        "Boosts immune function and overall well-being."
    ],
    "Pappaya": [
        "Aids in digestion with enzymes like papain.",
        "Rich in vitamins and supports skin health.",
        "Used traditionally for wound healing and immunity."
    ],
    "Pepper": [
        "Improves digestion and relieves bloating.",
        "Has antimicrobial properties and boosts metabolism."
    ],
    "Pomegranate": [
        "Rich in antioxidants, supports heart and skin health.",
        "Anti-inflammatory and aids in digestion."
    ],
    "Raktachandini": [
        "Anti-inflammatory and used in pain relief.",
        "Traditional use with caution due to toxicity."
    ],
    "Rose": [
        "Calming effect, used in aromatherapy.",
        "Hydrates skin and promotes relaxation."
    ],
    "Sapota": [
        "Rich in dietary fiber, aids in digestion.",
        "Provides energy and supports skin health."
    ],
    "Tulasi": [
        "Boosts immunity and supports respiratory health.",
        "Anti-inflammatory and used in treating colds."
    ],
    "Wood_sorel": [
        "Rich in vitamin C, used in treating scurvy.",
        "Anti-inflammatory and aids digestion."
    ],
    "Jasmine": [
        "Calms the mind and reduces anxiety.",
        "Used in teas for relaxation and stress relief."
    ],
    "Lemon": [
        "Rich in vitamin C, boosts immune function.",
        "Aids in digestion and supports detoxification.",
        "Promotes skin health and fights free radicals."
    ]
}

for key in use_of_medicine:
    use_of_medicine[key] = [use.strip() for use in use_of_medicine[key]]

# --- Disease Detection Data ---
disease_class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

    # Disease info
disease_info = {
        "Apple___Apple_scab": {"desc": "Fungal disease causing dark, velvety spots on leaves and fruit.", "remedy": "Apply fungicides, remove infected leaves, improve air circulation."},
        "Apple___Black_rot": {"desc": "Fungal infection leading to black, shriveled fruit and leaf spots.", "remedy": "Prune infected parts, apply fungicides, remove mummified fruit."},
        "Apple___Cedar_apple_rust": {"desc": "Fungal disease with yellow-orange spots on leaves and fruit.", "remedy": "Remove nearby cedar trees, apply fungicides, prune affected areas."},
        "Apple___healthy": {"desc": "No disease present, healthy apple plant.", "remedy": "Maintain proper care: watering, pruning, and fertilization."},
        "Blueberry___healthy": {"desc": "No disease present, healthy blueberry plant.", "remedy": "Continue regular care: watering, mulching, and monitoring."},
        "Cherry___Powdery_mildew": {"desc": "Fungal disease with white powdery coating on leaves.", "remedy": "Apply sulfur-based fungicides, improve air circulation, prune affected parts."},
        "Cherry___healthy": {"desc": "No disease present, healthy cherry plant.", "remedy": "Maintain proper irrigation and pruning practices."},
        "Corn___Cercospora_leaf_spot Gray_leaf_spot": {"desc": "Fungal disease causing grayish-white leaf spots.", "remedy": "Use resistant varieties, apply fungicides, rotate crops."},
        "Corn___Common_rust": {"desc": "Fungal infection with orange-brown pustules on leaves.", "remedy": "Plant resistant hybrids, apply fungicides, remove crop debris."},
        "Corn___Northern_Leaf_Blight": {"desc": "Fungal disease with long, grayish-white lesions on leaves.", "remedy": "Use resistant varieties, apply fungicides, practice crop rotation."},
        "Corn___healthy": {"desc": "No disease present, healthy corn plant.", "remedy": "Ensure proper nutrition and irrigation."},
        "Grape___Black_rot": {"desc": "Fungal disease causing black spots on leaves and fruit.", "remedy": "Remove infected parts, apply fungicides, improve canopy airflow."},
        "Grape___Esca_(Black_Measles)": {"desc": "Fungal disease with dark streaks in wood and leaf wilting.", "remedy": "Prune affected vines, no cure but manage with sanitation."},
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"desc": "Fungal infection with brown, necrotic leaf spots.", "remedy": "Apply fungicides, remove infected leaves, improve air circulation."},
        "Grape___healthy": {"desc": "No disease present, healthy grape vine.", "remedy": "Maintain pruning and irrigation practices."},
        "Orange___Haunglongbing_(Citrus_greening)": {"desc": "Bacterial disease causing yellowing leaves and misshapen fruit.", "remedy": "Remove infected trees, control psyllid vectors, no cure."},
        "Peach___Bacterial_spot": {"desc": "Bacterial disease with water-soaked spots on leaves and fruit.", "remedy": "Apply copper-based sprays, remove infected parts, avoid overhead watering."},
        "Peach___healthy": {"desc": "No disease present, healthy peach plant.", "remedy": "Continue proper care: pruning, watering, and pest monitoring."},
        "Pepper,_bell___Bacterial_spot": {"desc": "Bacterial infection causing dark, water-soaked spots on leaves.", "remedy": "Use copper sprays, remove infected plants, avoid wet foliage."},
        "Pepper,_bell___healthy": {"desc": "No disease present, healthy bell pepper plant.", "remedy": "Maintain consistent watering and fertilization."},
        "Potato___Early_blight": {"desc": "Fungal disease with concentric rings on leaves.", "remedy": "Apply fungicides, remove infected leaves, rotate crops."},
        "Potato___Late_blight": {"desc": "Fungal disease causing dark, wet lesions on leaves and tubers.", "remedy": "Apply fungicides, destroy infected plants, improve drainage."},
        "Potato___healthy": {"desc": "No disease present, healthy potato plant.", "remedy": "Ensure proper soil health and irrigation."},
        "Raspberry___healthy": {"desc": "No disease present, healthy raspberry plant.", "remedy": "Maintain pruning and weed control."},
        "Soybean___healthy": {"desc": "No disease present, healthy soybean plant.", "remedy": "Continue crop rotation and monitoring."},
        "Squash___Powdery_mildew": {"desc": "Fungal disease with white powdery spots on leaves.", "remedy": "Apply fungicides, improve air circulation, remove infected parts."},
        "Strawberry___Leaf_scorch": {"desc": "Fungal disease causing dark purple to brown leaf spots.", "remedy": "Remove infected leaves, apply fungicides, improve spacing."},
        "Strawberry___healthy": {"desc": "No disease present, healthy strawberry plant.", "remedy": "Maintain irrigation and mulch for soil health."},
        "Tomato___Bacterial_spot": {"desc": "Bacterial disease with small, water-soaked spots on leaves.", "remedy": "Use copper sprays, remove infected parts, avoid overhead watering."},
        "Tomato___Early_blight": {"desc": "Fungal disease with concentric rings on leaves.", "remedy": "Apply fungicides, remove lower leaves, rotate crops."},
        "Tomato___Late_blight": {"desc": "Fungal disease with large, wet lesions on leaves and fruit.", "remedy": "Apply fungicides, destroy infected plants, improve air flow."},
        "Tomato___Leaf_Mold": {"desc": "Fungal disease with yellowing leaves and mold on undersides.", "remedy": "Improve ventilation, apply fungicides, remove infected leaves."},
        "Tomato___Septoria_leaf_spot": {"desc": "Fungal disease with small, grayish spots on leaves.", "remedy": "Remove infected leaves, apply fungicides, avoid wet foliage."},
        "Tomato___Spider_mites Two-spotted_spider_mite": {"desc": "Pest damage causing stippling and webbing on leaves.", "remedy": "Use miticides, increase humidity, introduce predatory mites."},
        "Tomato___Target_Spot": {"desc": "Fungal disease with concentric spots on leaves.", "remedy": "Apply fungicides, remove infected leaves, improve spacing."},
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"desc": "Viral disease with yellowing, curling leaves.", "remedy": "Control whiteflies, remove infected plants, use resistant varieties."},
        "Tomato___Tomato_mosaic_virus": {"desc": "Viral disease causing mottled leaves and stunted growth.", "remedy": "Remove infected plants, disinfect tools, use resistant varieties."},
        "Tomato___healthy": {"desc": "No disease present, healthy tomato plant.", "remedy": "Maintain proper watering, staking, and fertilization."}
    }

# --- CSS (Unchanged) ---
def load_css():
    st.markdown("""
    <style>
    /* Enhanced Header Styles */
    .header-container {
        padding: 2rem 0;
        text-align: center;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        animation: fadeInDown 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }

    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, transparent, #059669, transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        100% { left: 100%; }
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .main-title {
        color: #064e3b;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(120deg, #064e3b, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titlePulse 2s infinite;
    }

    @keyframes titlePulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    } 
        /* Modern Team Members Section */
        .team-section {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 1rem;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(229, 231, 235, 0.5);
        }
        
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .team-member-card {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            padding: 1rem;
            border-radius: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            transition: all 0.3s ease;
            border: 1px solid rgba(5, 150, 105, 0.1);
        }
        
        .team-member-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        
        .member-avatar {
            width: 40px;
            height: 40px;
            background: #059669;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .member-info {
            flex: 1;
        }
        
        .member-name {
            color: #065f46;
            font-weight: 600;
            margin: 0;
            font-size: 1rem;
        }
        
        .member-role {
            color: #059669;
            font-size: 0.85rem;
            margin: 0;
        }
        
        /* Modern Drop Zone */
        .modern-drop-zone {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 2px dashed #059669;
            border-radius: 1rem;
            padding: 3rem 2rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .modern-drop-zone:hover {
            border-color: #065f46;
            transform: scale(1.01);
        }
        
        .modern-drop-zone.dragging {
            background: rgba(240, 253, 244, 0.9);
            border-color: #065f46;
            transform: scale(1.02);
        }
        
        .upload-icon {
            width: 64px;
            height: 64px;
            margin: 0 auto 1rem;
            color: #059669;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .drop-zone-text {
            color: #065f46;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .drop-zone-subtext {
            color: #059669;
            font-size: 0.9rem;
        }
        
        .supported-formats {
            margin-top: 1rem;
            display: flex;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .format-badge {
            background: rgba(5, 150, 105, 0.1);
            color: #059669;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .chat-container {
        background: white;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        padding: 1rem;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        cursor: pointer;
    }
    
    .chat-header:hover {
        background: linear-gradient(135deg, #065f46 0%, #05875f 100%);
    }
    
    .chat-body {
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .chat-input {
        background: #f3f4f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem;
    }
    
    /* Enhanced Results Section */
    .result-card {
        background: white;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .confidence-meter {
        background: #e5e7eb;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #059669 0%, #064e3b 100%);
        transition: width 0.5s ease;
    }
    
    .property-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .property-card {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        transition: transform 0.2s ease;
    }
    
    .property-card:hover {
        transform: translateY(-2px);
    }
    
    .expandable-section {
        margin-top: 1rem;
    }
    
    .expandable-header {
        background: #f3f4f6;
        padding: 0.75rem;
        border-radius: 0.5rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .expandable-content {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 0 0 0.5rem 0.5rem;
        margin-top: 0.25rem;
    }
    
    /* Animated Confidence Bar Styles */
    .confidence-container {
        margin: 1rem 0;
        padding: 1rem;
        background: white;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .confidence-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        color: #374151;
        font-weight: 500;
    }
    
    .confidence-bar {
        height: 1rem;
        background: #e5e7eb;
        border-radius: 1rem;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        width: 0;
        border-radius: 1rem;
        background: linear-gradient(90deg, #059669, #064e3b);
        transition: width 1s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            45deg,
            rgba(255,255,255,0.2) 25%,
            transparent 25%,
            transparent 50%,
            rgba(255,255,255,0.2) 50%,
            rgba(255,255,255,0.2) 75%,
            transparent 75%,
            transparent
        );
        background-size: 1rem 1rem;
        animation: shimmer 1s linear infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-1rem); }
        100% { transform: translateX(1rem); }
    }
    
    /* Color variations based on confidence level */
    .confidence-fill.high {
        background: linear-gradient(90deg, #059669, #064e3b);
    }
    
    .confidence-fill.medium {
        background: linear-gradient(90deg, #0ea5e9, #0369a1);
    }
    
    .confidence-fill.low {
        background: linear-gradient(90deg, #f59e0b, #d97706);
    }
    </style>

    """, unsafe_allow_html=True)

# --- Model Loading Functions ---
@st.cache_resource
def load_prediction_model():
    try:
        download_url = "https://drive.google.com/uc?id=17xebXPPkKbQYJjAE0qyxikUjoUY6BNoz"
        model_path = "Medicinal_Plant.h5"
        gdown.download(download_url, model_path, quiet=False)
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading plant model: {str(e)}")
        return None

@st.cache_resource
def load_trained_model():
    try:
        download_url = "https://drive.google.com/uc?id=1eaScRp1Oz3nzNeJeRqi78UEqWdEhbBoo"
        model_path = "plant_model.pth"
        gdown.download(download_url, model_path, quiet=False)
        model = resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 38)  # 38 classes
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(device)
    except Exception as e:
        st.error(f"Error loading disease model: {e}")
        return None

# --- Prediction Functions ---
def predict_class(img):
    try:
        st.session_state['loading'] = True
        img = Image.open(img).convert('RGB')
        img = img.resize((256, 256))
        img_array = image.img_to_array(img)  
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        if st.session_state['model_plant'] is None:
            st.session_state['model_plant'] = load_prediction_model()

        if st.session_state['model_plant'] is not None:
            predictions = st.session_state['model_plant'].predict(img_array)
            predicted_class_index = np.argmax(predictions)
            confidence = float(predictions[0][predicted_class_index]) * 100
            st.session_state['loading'] = False
            return class_labels[predicted_class_index], confidence
        return None, None
    except Exception as e:
        st.error(f"Error during plant prediction: {str(e)}")
        st.session_state['loading'] = False
        return None, None
        
def preprocess_image(img, target_size=(224, 224), brightness=1.0, contrast=1.0):
    img = img.convert("RGB")
    img_cv = np.array(img)
    img_cv = cv2.convertScaleAbs(img_cv, alpha=brightness, beta=contrast * 100 - 100)
    img = Image.fromarray(img_cv)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def predict_disease(model, img_tensor, class_names, threshold=0.9):
    with torch.no_grad():
        predictions = model(img_tensor)
        probs = torch.softmax(predictions, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
        if confidence < threshold:
            return "Unknown (Possible Anomaly)", confidence * 100
        return class_names[pred_class], confidence * 100

def estimate_severity(confidence):
    if confidence > 90: return "Mild"
    elif confidence > 70: return "Moderate"
    else: return "Severe"

def detect_species(disease_name):
    return disease_name.split("___")[0]

def generate_heatmap(model, img_tensor, pred_class):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    heatmap = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return heatmap

# --- API Setup ---
app = FastAPI()

@app.post("/predict_disease")
async def predict_api(file: UploadFile = File(...)):
    img = Image.open(file.file)
    img_tensor = preprocess_image(img).to(device)
    disease, confidence = predict_disease(st.session_state['model_disease'], img_tensor, disease_class_names)
    return {"disease": disease, "confidence": confidence}

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- Main App ---
def main():
    load_css()
    
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 class="main-title">🌿 MediPlant AI</h1>
            <p class="subtitle" style="color: #065f46; font-size: 1.2rem; margin-bottom: 1rem;">
                Advanced medicinal plant identification and disease detection powered by AI.
            </p>
            <p class="slogan" style="color: #059669; font-style: italic; font-size: 1.1rem; margin-bottom: 1rem;">
                "Unlocking Nature's Medicine Cabinet with AI"
            </p>
            <div class="guide-info" style="background: rgba(5, 150, 105, 0.1); padding: 0.5rem; border-radius: 0.5rem; display: inline-block;">
                <p style="color: #065f46; margin: 0;">
                    <strong>Project Guide:</strong> Mrs. A Anitharani
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Team Section
    st.markdown("""
    <div class="team-section">
        <h4 style="color: #065f46; margin: 0 0 0.5rem 0;">Project Team</h4>
        <div class="team-grid">
            <div class="team-member-card">
                <div class="member-avatar">S</div>
                <div class="member-info">
                    <p class="member-name">Santhoshkumar J</p>
                    <p class="member-role">Team Member</p>
                </div>
            </div>
            <div class="team-member-card">
                <div class="member-avatar">R</div>
                <div class="member-info">
                    <p class="member-name">Raghul M</p>
                    <p class="member-role">Team Member</p>
                </div>
            </div>
            <div class="team-member-card">
                <div class="member-avatar">S</div>
                <div class="member-info">
                    <p class="member-name">Shivam Sinha</p>
                    <p class="member-role">Team Member</p>
                </div>
            </div>
            <div class="team-member-card">
                <div class="member-avatar">K</div>
                <div class="member-info">
                    <p class="member-name">S Keerthika</p>
                    <p class="member-role">Team Member</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load both models
    if st.session_state['model_plant'] is None:
        st.session_state['model_plant'] = load_prediction_model()
    if st.session_state['model_disease'] is None:
        st.session_state['model_disease'] = load_trained_model()

    # Single Upload Interface
    st.subheader("Upload Plant Image")
    uploaded_file = st.file_uploader(
        "Upload an image for plant identification and disease detection",
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG",
        key="combined_uploader"
    )

    # Tabs for Results
    tab1, tab2 = st.tabs(["Plant Identification", "Disease Detection"])

    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            # Make sure the file position is reset before displaying
            uploaded_file.seek(0)
            st.image(img, caption="Uploaded Image", width=None)
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
            st.warning("Please try uploading a different image file.")

        with tab1:
            st.subheader("Plant Identification Results")
            if st.session_state['loading']:
                st.markdown("""
                    <div class="loading-container">
                        <div class="loading-spinner"></div>
                        <div class="loading-text">Analyzing plant image...</div>
                    </div>
                """, unsafe_allow_html=True)
                
            predicted_class, confidence = predict_class(uploaded_file)
                
            if predicted_class and not st.session_state['loading']:
                st.markdown(f"""
                    <div class="prediction-container">
                        <h2 style="color: #064e3b; margin-bottom: 0.5rem;">
                            {predicted_class}
                        </h2>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence}%"></div>
                        </div>
                        <p style="color: #374151;">Confidence: {confidence:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                    <div class="info-section">
                        <h3 class="info-title">📝 Method of Preparation</h3>
                """, unsafe_allow_html=True)
                st.write(methods_of_preparation.get(predicted_class, "No information available"))
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("""
                    <div class="info-section">
                        <h3 class="info-title">💊 Medicinal Uses</h3>
                """, unsafe_allow_html=True)
                uses = use_of_medicine.get(predicted_class, "No information available")
                if isinstance(uses, list):
                    for use in uses:
                        st.markdown(f"• {use}")
                else:
                    st.markdown(f"• {uses}")
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
                st.subheader("Disease Detection Results")            
            
                # Initialize LLM with error handling
                if "llm_pipeline" not in st.session_state:
                    try:
                        model_name = "distilgpt2"
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModelForCausalLM.from_pretrained(model_name)
                        st.session_state.llm_pipeline = pipeline(
                            "text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            max_new_tokens=100,
                            return_full_text=False
                        )
                        # Clear memory
                        import gc
                        gc.collect()
                    except Exception as e:
                        st.error(f"Failed to load LLM: {str(e)}")
                        st.session_state.llm_pipeline = None
            
                        with st.sidebar:
                                    st.header("🌿 PhytoSense Chat Assistant")
                                    st.markdown("Ask about plant diseases or medicinal uses!")
                            
                                    # Chat history
                                    if "chat_history" not in st.session_state:
                                        st.session_state.chat_history = [
                                            {"role": "assistant", "content": "Hi! Ask about tomato blight or neem uses."}
                                        ]
                            
                                    # Display chat messages
                                    chat_container = st.container()
                                    with chat_container:
                                        for message in st.session_state.chat_history:
                                            with st.chat_message(message["role"], avatar="🌱" if message["role"] == "assistant" else None):
                                                st.markdown(message["content"])
                            
                                    # User input
                                    prompt = st.chat_input("Ask a question (e.g., 'What causes tomato blight?')")
                                    if prompt:
                                        st.session_state.chat_history.append({"role": "user", "content": prompt})
                                        with chat_container:
                                            with st.chat_message("user"):
                                                st.markdown(prompt)
                            
                                        with st.spinner("Generating response..."):
                                            if st.session_state.llm_pipeline:
                                                llm_prompt = f"Act as a plant care expert. Answer concisely: {prompt}"
                                                try:
                                                    outputs = st.session_state.llm_pipeline(llm_prompt, max_new_tokens=100)
                                                    response = outputs[0]["generated_text"].strip() if outputs else "Sorry, no response generated."
                                                    if not response:
                                                        response = "Sorry, I couldn't generate a response. Try rephrasing."
                                                except Exception as e:
                                                    response = f"LLM error: {str(e)}. Try a simpler question."
                                                # Clear memory
                                                import gc
                                                gc.collect()
                                            else:
                                                response = "LLM unavailable. Try asking about tomato blight or neem."
                            
                                            st.session_state.chat_history.append({
                                                "role": "assistant",
                                                "content": response
                                            })
                                            with chat_container:
                                                with st.chat_message("assistant", avatar="🌱"):
                                                    st.markdown(response)
                    
                with st.sidebar.expander("How to Use"):
                    st.write("Adjust settings for disease detection.")

                confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 90)
                brightness = st.slider("Brightness", 0.5, 1.5, 1.0)
                contrast = st.slider("Contrast", 0.5, 1.5, 1.0)
                    
                global device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                img_tensor = preprocess_image(img, brightness=brightness, contrast=contrast).to(device)
                            
                with st.spinner("Analyzing..."):
                    disease, disease_confidence = predict_disease(st.session_state['model_disease'], img_tensor, disease_class_names, confidence_threshold / 100)
                    species = detect_species(disease)
                    severity = estimate_severity(disease_confidence)
                            
                    st.write(f"**Species:** {species}")
                    st.write(f"**Disease:** {disease.replace('___', ' - ')}")
                    st.write(f"**Confidence:** {disease_confidence:.2f}%")
                    st.write(f"**Severity:** {severity}")
                                            
                if disease in disease_info:
                    st.write(f"**Description:** {disease_info[disease]['desc']}")
                    st.write(f"**Remedy:** {disease_info[disease]['remedy']}")
                            
                    heatmap = generate_heatmap(st.session_state['model_disease'], img_tensor, disease_class_names.index(disease) if "Unknown" not in disease else 0)
                    st.image(heatmap, caption="Heatmap", width=200)
                                            
                    fig, ax = plt.subplots()
                    ax.bar(disease_class_names, st.session_state['model_disease'](img_tensor)[0].cpu().softmax(dim=0).detach().numpy())
                    ax.tick_params(axis='x', rotation=90)
                    st.pyplot(fig)
                    plt.close(fig)  # Free memory
                            
                    feedback = st.radio("Prediction correct?", ("Yes", "No"), key="fb_disease")
                    if feedback == "No":
                        with open("feedback.txt", "a") as f:
                            f.write(f"{disease},{disease_confidence}\n")
                        
                                
                
                
    # About Section
    st.markdown("""
        <div class="glass-card">
            <h3 style="color: #064e3b; margin-bottom: 1rem;">About MediPlant AI</h3>
            <p style="color: #374151; line-height: 1.6;">
                MediPlant AI uses advanced machine learning to identify medicinal plants and detect plant diseases. 
                Our system can identify 40 medicinal plant species and 38 disease conditions with high accuracy, 
                providing detailed information on traditional uses, preparation methods, and remedies.
            </p>
            <br>
            <h4 style="color: #064e3b; margin-bottom: 0.5rem;">Features:</h4>
            <ul style="color: #374151; line-height: 1.6;">
                <li>Real-time plant identification and disease detection</li>
                <li>Detailed preparation methods and medicinal uses</li>
                <li>Disease severity assessment and remedies</li>
                <li>Confidence scoring and heatmap visualization</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Start API in background
threading.Thread(target=run_api, daemon=True).start()

if __name__ == "__main__":
    main()
