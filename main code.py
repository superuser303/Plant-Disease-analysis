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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.image import show_cam_on_image
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import io
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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
