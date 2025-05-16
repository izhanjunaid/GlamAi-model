import streamlit as st
from PIL import Image
from training.inference import Inference
from training.config import get_config
import argparse
import torch
import os

print("Checkpoint Exists:", os.path.exists("ckpts/sow_pyramid_a5_e3d2_remapped.pth"))

# Configuration and argument setup
config = get_config()
args = argparse.Namespace(
    load_path="ckpts/G.pth",  # Path to your model checkpoint
    device="cuda" if torch.cuda.is_available() else "cpu"
)
inference = Inference(config, args, model_path=args.load_path)

# Streamlit interface
st.title("Interactive Makeup Transfer")
st.write("Upload a source (non-makeup) image and a reference (makeup) image to transfer makeup.")

# Upload source and reference images
source_file = st.file_uploader("Upload Source Image", type=["jpg", "png"])
reference_file = st.file_uploader("Upload Reference Image", type=["jpg", "png"])

# Options to control which part of the face to apply makeup to
st.write("Choose which parts of makeup to apply:")
apply_lip = st.checkbox("Apply Lip Makeup", value=True)
apply_skin = st.checkbox("Apply Skin Makeup", value=True)
apply_eye = st.checkbox("Apply Eye Makeup", value=True)

if source_file and reference_file:
    source_img = Image.open(source_file).convert('RGB')
    reference_img = Image.open(reference_file).convert('RGB')

    st.image(source_img, caption="Source Image", use_column_width=True)
    st.image(reference_img, caption="Reference Image", use_column_width=True)

    # Perform makeup transfer
    with st.spinner("Applying makeup transfer..."):
        if apply_lip and apply_skin and apply_eye:
            result = inference.joint_transfer(source_img, reference_img, reference_img, reference_img)
        elif apply_lip and apply_skin:
            result = inference.joint_transfer(source_img, reference_img, reference_img, reference_img)
        elif apply_lip and apply_eye:
            result = inference.joint_transfer(source_img, reference_img, reference_img, reference_img)
        elif apply_skin and apply_eye:
            result = inference.joint_transfer(source_img, reference_img, reference_img, reference_img)
        elif apply_lip:
            result = inference.joint_transfer(source_img, reference_img, reference_img, reference_img)
        elif apply_skin:
            result = inference.joint_transfer(source_img, reference_img, reference_img, reference_img)
        elif apply_eye:
            result = inference.joint_transfer(source_img, reference_img, reference_img, reference_img)
        else:
            st.error("Please select at least one part to apply makeup.")
            result = None
    
    if result:
        st.image(result, caption="Result Image", use_column_width=True)
        st.success("Makeup transfer successful!")
    else:
        st.error("Failed to apply makeup transfer.")
