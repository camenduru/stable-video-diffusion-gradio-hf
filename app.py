import math
import os
from glob import glob
from pathlib import Path
from typing import Optional


import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from huggingface_hub import hf_hub_download

from simple_video_sample import sample

num_frames = 25
num_steps = 30
model_config = "scripts/sampling/configs/svd_xt.yaml"
device = "cuda"

hf_hub_download(repo_id="stabilityai/stable-video-diffusion-img2vid-xt", filename="svd_xt.safetensors", local_dir="checkpoints", token=os.getenv("HF_TOKEN"))

css = '''
.gradio-container{max-width:850px !important}
'''

def sample(
    input_path: str,
    num_frames: Optional[int] = 25,
    num_steps: Optional[int] = 30,
    version: str = "svd_xt",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 7,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
):
    output_folder = str(uuid.uuid4())
    sample(input_path, version, output_folder, decoding_t)
    return f"{output_folder}/000000.mp4"

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


import gradio as gr
import uuid
def resize_image(image_path, output_size=(1024, 576)):
    with Image.open(image_path) as image:
        # Calculate aspect ratios
        target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
        image_aspect = image.width / image.height  # Aspect ratio of the original image
    
        # Resize then crop if the original image is larger
        if image_aspect > target_aspect:
            # Resize the image to match the target height, maintaining aspect ratio
            new_height = output_size[1]
            new_width = int(new_height * image_aspect)
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Calculate coordinates for cropping
            left = (new_width - output_size[0]) / 2
            top = 0
            right = (new_width + output_size[0]) / 2
            bottom = output_size[1]
        else:
            # Resize the image to match the target width, maintaining aspect ratio
            new_width = output_size[0]
            new_height = int(new_width / image_aspect)
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Calculate coordinates for cropping
            left = 0
            top = (new_height - output_size[1]) / 2
            right = output_size[0]
            bottom = (new_height + output_size[1]) / 2
    
        # Crop the image
        cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image

with gr.Blocks(css=css) as demo:
  gr.Markdown('''# Stable Video Diffusion - Image2Video - XT
Generate 25 frames of video from a single image with SDV-XT. [Join the waitlist](https://stability.ai/contact) for the text-to-video web experience
  ''')
  with gr.Column():
    image = gr.Image(label="Upload your image (it will be center cropped to 1024x576)", type="filepath")
    generate_btn = gr.Button("Generate")
    #with gr.Accordion("Advanced options", open=False):
    #  cond_aug = gr.Slider(label="Conditioning augmentation", value=0.02, minimum=0.0)
    #  seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=int(1e9), step=1)
      #decoding_t = gr.Slider(label="Decode frames at a time", value=6, minimum=1, maximum=14, interactive=False)
    #  saving_fps = gr.Slider(label="Saving FPS", value=6, minimum=6, maximum=48, step=6)
  with gr.Column():
    video = gr.Video()
  image.upload(fn=resize_image, inputs=image, outputs=image)
  generate_btn.click(fn=sample, inputs=[image], outputs=video, api_name="video")

demo.launch()
