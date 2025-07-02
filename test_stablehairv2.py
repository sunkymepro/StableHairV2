#!/usr/bin/env python3
import argparse
import logging
import os
import random
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, UniPCMultistepScheduler,UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from ref_encoder.reference_unet import CCProjection
from ref_encoder.latent_controlnet import ControlNetModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline as Hair3dPipeline
from src.utils.util import save_videos_grid
from omegaconf import OmegaConf

def log_validation(
    vae, tokenizer, image_encoder, denoising_unet,
    args, device, logger, cc_projection,
    controlnet, hair_encoder, feature_extractor=None
):
    """
    Run inference on validation pairs and save generated videos.
    """
    logger.info("Starting validation inference...")

    # Initialize inference pipeline
    pipeline = Hair3dPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        controlnet=controlnet,
        vae=vae,
        tokenizer=tokenizer,
        denoising_unet=denoising_unet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
    ).to(device)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    # Create output directory
    output_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(output_dir, exist_ok=True)

    print(output_dir)

    # Generate camera trajectory
    x_coords = [0.4 * np.sin(2 * np.pi * i / 120) for i in range(60)]
    y_coords = [-0.05 + 0.3 * np.cos(2 * np.pi * i / 120) for i in range(60)]
    X = [x_coords[0]]
    Y = [y_coords[0]]
    for i in range(20):
        X.append(x_coords[i * 3 + 2])
        Y.append(y_coords[i * 3 + 2])
    x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
    y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1).to(device)

    # Load reference images
    id_image = cv2.cvtColor(cv2.imread(args.validation_ids[0]), cv2.COLOR_BGR2RGB)
    id_image = cv2.resize(id_image, (512, 512))
    id_list = [id_image for _ in range(12)]
    hair_image = cv2.cvtColor(cv2.imread(args.validation_hairs[0]), cv2.COLOR_BGR2RGB)
    hair_image = cv2.resize(hair_image, (512, 512))
    prompt_img = cv2.cvtColor(cv2.imread(args.validation_ids[0]), cv2.COLOR_BGR2RGB)
    prompt_img = cv2.resize(prompt_img, (512, 512))
    prompt_img = [prompt_img]

    # Perform inference and save videos
    for idx in range(args.num_validation_images):
        result = pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=512,
            height=512,
            controlnet_condition=id_list,
            controlnet_conditioning_scale=1.0,
            generator=torch.Generator(device).manual_seed(args.seed),
            ref_image=hair_image,
            prompt_img=prompt_img,
            reference_encoder=hair_encoder,
            poses=None,
            x=x_tensor,
            y=y_tensor,
            video_length=21,
            context_frames=12,
        )
        video = torch.cat([result.videos, result.videos], dim=0)
        video_path = os.path.join(output_dir, f"generated_video_{idx}.mp4")
        save_videos_grid(video, video_path, n_rows=5, fps=24)
        logger.info(f"Saved generated video: {video_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for 3D hairstyle generation"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, required=True,
        help="Path or ID of the pretrained pipeline"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path or ID of the pretrained pipeline"
    )
    parser.add_argument(
        "--image_encoder", type=str, required=True,
        help="Path or ID of the CLIP vision encoder"
    )
    parser.add_argument(
        "--controlnet_model_name_or_path", type=str, default=None,
        help="Path or ID of the ControlNet model"
    )
    parser.add_argument(
        "--revision", type=str, default=None,
        help="Model revision or Git reference"
    )
    parser.add_argument(
        "--output_dir", type=str, default="inference_output",
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_validation_images", type=int, default=3,
        help="Number of videos to generate per input pair"
    )
    parser.add_argument(
        "--validation_ids", type=str, nargs='+', required=True,
        help="Path(s) to identity conditioning images"
    )
    parser.add_argument(
        "--validation_hairs", type=str, nargs='+', required=True,
        help="Path(s) to hairstyle reference images"
    )
    parser.add_argument(
        "--use_fp16", action="store_true",
        help="Enable fp16 inference"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Setup device and logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Set random seed
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Load models
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder,
        revision=args.revision
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision
    ).to(device)

    infer_config = OmegaConf.load('./configs/inference/inference_v2.yaml')

    unet2 = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, torch_dtype=torch.float16
    ).to(device)
    conv_in_8 = torch.nn.Conv2d(8, unet2.conv_in.out_channels, kernel_size=unet2.conv_in.kernel_size, padding=unet2.conv_in.padding)
    conv_in_8.requires_grad_(False)
    unet2.conv_in.requires_grad_(False)
    torch.nn.init.zeros_(conv_in_8.weight)
    conv_in_8.weight[:,:4,:,:].copy_(unet2.conv_in.weight)
    conv_in_8.bias.copy_(unet2.conv_in.bias)
    unet2.conv_in = conv_in_8

    # Load or initialize ControlNet
    controlnet = ControlNetModel.from_unet(unet2).to(device)
    state_dict2 = torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location=torch.device('cpu'))
    controlnet.load_state_dict(state_dict2, strict=False)

    # Load 3D UNet motion module
    prefix = "motion_module"
    ckpt_num = "4140000"
    save_path = os.path.join(args.model_path, f"{prefix}-{ckpt_num}.pth")

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        args.pretrained_model_name_or_path,
        save_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(device)

    # Load projection and hair encoder
    cc_projection = CCProjection().to(device)
    state_dict3 = torch.load(os.path.join(args.model_path, "pytorch_model_1.bin"), map_location=torch.device('cpu'))
    cc_projection.load_state_dict(state_dict3, strict=False)

    from ref_encoder.reference_unet import ref_unet
    Hair_Encoder = ref_unet.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, low_cpu_mem_usage=False, device_map=None, ignore_mismatched_sizes=True
    ).to(device)

    state_dict2 = torch.load(os.path.join(args.model_path, "pytorch_model_2.bin"), map_location=torch.device('cpu'))
    #state_dict2 = torch.load(os.path.join('/home/jichao.zhang/code/3dhair/train_sv3d/checkpoint-30000/', "pytorch_model.bin"))
    Hair_Encoder.load_state_dict(state_dict2, strict=False)

    # Run validation inference
    log_validation(
        vae, tokenizer, image_encoder, denoising_unet,
        args, device, logger,
        cc_projection, controlnet, Hair_Encoder
    )

if __name__ == "__main__":
    main()
