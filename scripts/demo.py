import os
import sys
import argparse
import numpy as np
from PIL import Image
import glob
import torch

sys.path.append('.')

from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args


def main(config, args):
    # Create logger
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    # Initialize inference
    inference = Inference(config, args, args.load_path)

    # Collect images
    source_files = glob.glob(args.source_dir + '/*.png')
    reference_files = glob.glob(args.reference_dir + '/*.png')

    if not source_files:
        logger.warning(f"No source images found in {args.source_dir}")
        return
    if not reference_files:
        logger.warning(f"No reference images found in {args.reference_dir}")
        return

    logger.info(f"Found {len(source_files)} source images and {len(reference_files)} reference images")

    # Process each source-reference pair
    for imga_name in source_files:
        if not imga_name.lower().endswith(('.jpg', '.png')):
            logger.warning(f"Skipping unsupported file type: {imga_name}")
            continue

        try:
            imgA = Image.open(imga_name).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading source image {imga_name}: {e}")
            continue

        for imgb_name in reference_files:
            try:
                imgB = Image.open(imgb_name).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading reference image {imgb_name}: {e}")
                continue

            logger.info(f"Processing: Source={imga_name}, Reference={imgb_name}")
            
            try:
                result = inference.transfer(imgA, imgB, postprocess=True)
                if result is None:
                    logger.warning(f"Transfer failed for: Source={imga_name}, Reference={imgb_name}")
                    continue

                # Save the result
                save_path = os.path.join(
                    args.save_folder,
                    f"{os.path.basename(imga_name).split('.')[0]}-{os.path.basename(imgb_name).split('.')[0]}.png"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                result.save(save_path)
                logger.info(f"Saved result to {save_path}")

            except Exception as e:
                logger.error(f"Error during transfer for Source={imga_name}, Reference={imgb_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument for makeup transfer demo")
    parser.add_argument("--name", type=str, default='cacd')
    parser.add_argument("--save_path", type=str, default='/content/output', help="Path to save output images")
    parser.add_argument("--load_path", type=str, default='ckpts/sow_pyramid_a5_e3d2_remapped.pth', help="Model checkpoint path")
    parser.add_argument("--source_dir", type=str, default="assets/images/non-makeup")
    parser.add_argument("--reference_dir", type=str, default="assets/images/makeup")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use")

    args = parser.parse_args()
    if torch.cuda.is_available():
        args.gpu = 'cuda:' + args.gpu
    else:
        args.gpu = 'cpu'
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path + '-' + args.name)
    os.makedirs(args.save_folder, exist_ok=True)

    config = get_config()
    main(config, args)
