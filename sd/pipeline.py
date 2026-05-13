import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH= 512 // 8
LATENTS_HEIGHT= 512 // 8

def generate(prompt: str,
             uncond_prompt: str,
             input_image=None,
             strength=0.8,
             do_cfg=True,
             cfg_scale=7.5,
             sampler_name="ddpm",
             n_inference_steps=50,
             models={},
             seed=None,
             device=None,
             idle_device=None,
             tokenizer=None):
    with torch.inference_mode():

        if not (0 < strength <= 1):
            raise ValueError("Strength should be in (0, 1]")
        
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator()