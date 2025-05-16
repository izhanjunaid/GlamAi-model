import streamlit as st
from PIL import Image
import torch

# Import your project modules
from training.config import get_config
from training.inference import Inference

# ------------------------------------------------------------------------------
# Helper: Load Inference Model (using caching so it isn’t reloaded on every run)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_inference(checkpoint_path: str):
    config = get_config()
    # Create a dummy args object. You can add more attributes as needed.
    class DummyArgs:
        pass
    args = DummyArgs()
    # Use GPU if available; otherwise, fallback to CPU.
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Dummy values for required attributes
    args.save_folder = "dummy_save_folder"
    args.name = "dummy_name"
    # Instantiate the inference wrapper with the trained checkpoint
    inference_instance = Inference(config, args, model_path=checkpoint_path)
    return inference_instance

# ------------------------------------------------------------------------------
# Streamlit Sidebar: Settings and Mode Selection
# ------------------------------------------------------------------------------
st.sidebar.title("Makeup Transfer Settings")
mode = st.sidebar.radio("Transfer Mode", ("Global", "Region-Specific"))
# Path to your trained generator checkpoint (update as needed)
checkpoint_path = st.sidebar.text_input("Checkpoint Path", "ckpts/G.pth")

# For region-specific mode, allow control of saturation for each region.
if mode == "Region-Specific":
    lip_sat = st.sidebar.slider("Lip Saturation", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    skin_sat = st.sidebar.slider("Skin Saturation", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    eye_sat = st.sidebar.slider("Eye Saturation", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

# Load the inference model (cached)
inference = load_inference(checkpoint_path)

# ------------------------------------------------------------------------------
# Streamlit Main Area: Title and File Uploaders
# ------------------------------------------------------------------------------
st.title("Makeup Transfer App")
st.write("Upload a source image (without makeup) and reference image(s) to perform makeup transfer.")

if mode == "Global":
    source_file = st.file_uploader("Upload Source Image (No Makeup)", type=["jpg", "jpeg", "png"])
    reference_file = st.file_uploader("Upload Reference Image (Makeup)", type=["jpg", "jpeg", "png"])
else:
    source_file = st.file_uploader("Upload Source Image (No Makeup)", type=["jpg", "jpeg", "png"])
    ref_lip_file = st.file_uploader("Upload Lip Reference Image", type=["jpg", "jpeg", "png"])
    ref_skin_file = st.file_uploader("Upload Skin Reference Image", type=["jpg", "jpeg", "png"])
    ref_eye_file = st.file_uploader("Upload Eye Reference Image", type=["jpg", "jpeg", "png"])

# ------------------------------------------------------------------------------
# Transfer Makeup Button and Processing
# ------------------------------------------------------------------------------
if st.button("Transfer Makeup"):
    if source_file is None:
        st.error("Please upload a source image!")
    else:
        source_img = Image.open(source_file).convert("RGB")
        if mode == "Global":
            if reference_file is None:
                st.error("Please upload a reference image for global transfer!")
            else:
                reference_img = Image.open(reference_file).convert("RGB")
                with st.spinner("Transferring makeup (global mode)..."):
                    # Call the inference wrapper’s global transfer method.
                    # The transfer method takes the source and reference images
                    result_img = inference.transfer(source_img, reference_img, postprocess=True)
                st.image(result_img, caption="Result Image", use_column_width=True)
        else:
            # Region-specific mode
            if ref_lip_file is None or ref_skin_file is None or ref_eye_file is None:
                st.error("Please upload all three region-specific reference images!")
            else:
                ref_lip = Image.open(ref_lip_file).convert("RGB")
                ref_skin = Image.open(ref_skin_file).convert("RGB")
                ref_eye = Image.open(ref_eye_file).convert("RGB")
                with st.spinner("Transferring makeup (region-specific mode)..."):
                    # Preprocess the images using the same pipeline as during training.
                    source_input, face, crop_face = inference.preprocess(source_img)
                    lip_input, _, _ = inference.preprocess(ref_lip)
                    skin_input, _, _ = inference.preprocess(ref_skin)
                    eye_input, _, _ = inference.preprocess(ref_eye)
                    
                    if not (source_input and lip_input and skin_input and eye_input):
                        st.error("Failed to process one or more images. Please try different images.")
                    else:
                        source_mask = source_input[1]
                        source_sample = inference.generate_source_sample(source_input)
                        # Generate reference samples with the user–controlled saturation values.
                        reference_samples = [
                            inference.generate_reference_sample(lip_input, source_mask=source_mask, mask_area='lip', saturation=lip_sat),
                            inference.generate_reference_sample(skin_input, source_mask=source_mask, mask_area='skin', saturation=skin_sat),
                            inference.generate_reference_sample(eye_input, source_mask=source_mask, mask_area='eye', saturation=eye_sat)
                        ]
                        # Fuse the features from the source and region-specific references.
                        result_img = inference.interface_transfer(source_sample, reference_samples)
                        # Postprocess the result (resize, blend with Laplacian differences, denoise, etc.)
                        result_img = inference.postprocess(source_img, crop_face, result_img)
                st.image(result_img, caption="Result Image", use_column_width=True)
