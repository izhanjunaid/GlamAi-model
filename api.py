from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
from training.config import get_config
from training.inference import Inference
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model (using the same setup as in app.py)
@torch.no_grad()
def load_inference(checkpoint_path: str):
    config = get_config()
    class DummyArgs:
        pass
    args = DummyArgs()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.save_folder = "dummy_save_folder"
    args.name = "dummy_name"
    inference_instance = Inference(config, args, model_path=checkpoint_path)
    return inference_instance

# Load the model once at startup
inference = load_inference("ckpts/G.pth")

def validate_saturation(value: float, name: str) -> float:
    """Validate and log saturation values."""
    if not 0 <= value <= 2:
        logger.warning(f"Invalid {name} saturation value: {value}. Clamping to range [0, 2]")
        return max(0, min(2, value))
    logger.info(f"{name} saturation value: {value}")
    return value

@app.post("/transfer/region-specific")
async def transfer_makeup(
    source: UploadFile = File(...),
    ref_lip: UploadFile = File(...),
    ref_skin: UploadFile = File(...),
    ref_eye: UploadFile = File(...),
    lip_sat: float = Form(...),
    skin_sat: float = Form(...),
    eye_sat: float = Form(...)
):
    try:
        # Log and validate saturation values
        lip_sat = validate_saturation(lip_sat, "Lip")
        skin_sat = validate_saturation(skin_sat, "Skin")
        eye_sat = validate_saturation(eye_sat, "Eye")

        # Log file information
        logger.info(f"Processing files:")
        logger.info(f"Source: {source.filename}")
        logger.info(f"Lip reference: {ref_lip.filename}")
        logger.info(f"Skin reference: {ref_skin.filename}")
        logger.info(f"Eye reference: {ref_eye.filename}")

        # Read and convert uploaded files to PIL Images
        source_img = Image.open(io.BytesIO(await source.read())).convert("RGB")
        ref_lip_img = Image.open(io.BytesIO(await ref_lip.read())).convert("RGB")
        ref_skin_img = Image.open(io.BytesIO(await ref_skin.read())).convert("RGB")
        ref_eye_img = Image.open(io.BytesIO(await ref_eye.read())).convert("RGB")

        # Log image sizes
        logger.info(f"Image sizes:")
        logger.info(f"Source: {source_img.size}")
        logger.info(f"Lip reference: {ref_lip_img.size}")
        logger.info(f"Skin reference: {ref_skin_img.size}")
        logger.info(f"Eye reference: {ref_eye_img.size}")

        # Preprocess the images
        logger.info("Starting image preprocessing...")
        source_input, face, crop_face = inference.preprocess(source_img)
        lip_input, _, _ = inference.preprocess(ref_lip_img)
        skin_input, _, _ = inference.preprocess(ref_skin_img)
        eye_input, _, _ = inference.preprocess(ref_eye_img)

        if not (source_input and lip_input and skin_input and eye_input):
            logger.error("Failed to process one or more images")
            raise HTTPException(status_code=400, detail="Failed to process one or more images")

        # Generate source sample
        logger.info("Generating source sample...")
        source_mask = source_input[1]
        source_sample = inference.generate_source_sample(source_input)

        # Generate reference samples with the provided saturation values
        logger.info("Generating reference samples...")
        reference_samples = [
            inference.generate_reference_sample(lip_input, source_mask=source_mask, mask_area='lip', saturation=lip_sat),
            inference.generate_reference_sample(skin_input, source_mask=source_mask, mask_area='skin', saturation=skin_sat),
            inference.generate_reference_sample(eye_input, source_mask=source_mask, mask_area='eye', saturation=eye_sat)
        ]

        # Perform the transfer
        logger.info("Performing makeup transfer...")
        result_img = inference.interface_transfer(source_sample, reference_samples)
        
        # Postprocess the result
        logger.info("Postprocessing result...")
        result_img = inference.postprocess(source_img, crop_face, result_img)

        # Convert the result to bytes
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        logger.info("Transfer completed successfully!")
        # Return the image as a streaming response
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        logger.error(f"Error during makeup transfer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 