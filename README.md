# Stable Diffusion from Scratch — PyTorch

A from-scratch PyTorch implementation of **Stable Diffusion v1.5** for text-to-image generation, built as a learning project following Umar Jamil's [*Coding Stable Diffusion from scratch in PyTorch*](https://www.youtube.com/watch?v=ZBKpAp_6TGI) tutorial.

The goal is pedagogical: reproduce every component of the Stable Diffusion architecture — VAE, CLIP text encoder, U-Net with cross-attention, DDPM sampler — in pure PyTorch with `nn.Module`, no `diffusers` library, no shortcuts. Pretrained weights from a standard SD v1.5 checkpoint are remapped into this implementation's parameter names via `model_converter.py`.

---

## What's in this implementation

Stable Diffusion is a **latent diffusion model**. Instead of denoising in pixel space (expensive), it denoises in the compressed latent space of a pretrained VAE. The conditioning signal is a text embedding from a CLIP text encoder, injected into the U-Net via cross-attention.

The full inference path is:

```
text prompt
   │
   ▼
[Tokenizer]  ──►  [CLIP text encoder]  ──►  context (77, 768)
                                                   │
                                                   ▼
random noise (4, 64, 64)  ──►  [U-Net] × N steps  ──►  denoised latent (4, 64, 64)
                                    ▲                            │
                                    │                            ▼
                            time embedding                  [VAE Decoder]
                                                                 │
                                                                 ▼
                                                        image (3, 512, 512)
```

Each box is a separate module in this repo:

| File | What it contains |
|---|---|
| `sd/attention.py` | `SelfAttention` and `CrossAttention` — the two attention flavors used throughout |
| `sd/clip.py` | `CLIP` text encoder — 12-layer transformer, vocab 49408, 768-dim, 77 tokens |
| `sd/encoder.py` | `VAE_Encoder` — RGB image → 4-channel latent at 1/8 spatial resolution |
| `sd/decoder.py` | `VAE_Decoder` + shared `VAE_ResidualBlock` and `VAE_AttentionBlock` |
| `sd/diffusion.py` | `Diffusion` wrapper, `UNET`, `TimeEmbedding`, U-Net residual/attention/upsample blocks |
| `sd/ddpm.py` | `DDPMSampler` — forward noising schedule and reverse `step()` for sampling |
| `sd/pipeline.py` | `generate()` — full text-to-image inference loop with classifier-free guidance |
| `sd/model_loader.py` | Instantiate all four modules and load remapped weights |
| `sd/model_converter.py` | Remap standard SD v1.5 checkpoint keys → this repo's parameter names |
| `sd/demo.ipynb` | Inference demo notebook |

---

## Architecture details

### VAE — image ↔ latent

The VAE compresses 512 × 512 RGB images into 64 × 64 × 4 latents (an 8× spatial compression).

**Encoder** (`encoder.py`): three downsampling stages of residual + group-norm + SiLU blocks, with one self-attention block at the bottleneck (32 × 32 × 512). Final 1×1 conv outputs 8 channels, split into `mean` and `log_var` — the latent is sampled `z = mean + std * noise` and scaled by the SD constant `0.18215`.

**Decoder** (`decoder.py`): mirror of the encoder — divides out the `0.18215` scale, then three upsampling stages of residual blocks back to 512 × 512 × 3.

### CLIP text encoder

`clip.py` reimplements the CLIP-L/14 text encoder used by SD v1.5:

- Vocab size: **49,408**
- Sequence length: **77** (fixed, padded)
- Embedding dim: **768**
- Layers: **12**, each with 12-head self-attention (causal mask) + QuickGELU FFN
- Learned position embeddings (not sinusoidal)

Output: a `(batch, 77, 768)` tensor — this is the `context` consumed by the U-Net's cross-attention layers.

### U-Net with cross-attention

`diffusion.py` implements the conditional noise predictor. Structure:

- **`TimeEmbedding`**: sinusoidal timestep → 320-dim → 1280-dim (two linear layers with SiLU)
- **Encoder path** (downsampling): four stages from `(4, 64, 64)` → `(1280, 8, 8)`. Each stage stacks `UNET_ResidualBlock` (gets time embedding) and `UNET_AttentionBlock` (gets text context via cross-attention).
- **Bottleneck**: residual → attention → residual at the deepest level.
- **Decoder path** (upsampling): mirror of encoder with skip connections, ending at `(320, 64, 64)`.
- **`UNET_OutputLayer`**: group-norm + SiLU + 3×3 conv → `(4, 64, 64)` predicted noise.

The `SwitchSequential` helper dispatches forward arguments by layer type so the same container can route `time`, `context`, or neither to the right submodule.

**`UNET_AttentionBlock`** is the transformer block stitched into the U-Net:

1. Self-attention over spatial tokens (no causal mask)
2. **Cross-attention** with the CLIP `context` as keys/values — this is where text conditioning enters
3. GEGLU feed-forward
4. All three with pre-LayerNorm and residual connections

### DDPM sampler

`ddpm.py` implements the [Ho et al., 2020](https://arxiv.org/abs/2006.11239) sampler:

- Linear-in-`sqrt(beta)` schedule from `β_start=0.00085` to `β_end=0.0120`
- `add_noise()` — forward process for training
- `step()` — reverse process for inference: given the predicted noise, returns `x_{t-1}`

### Classifier-Free Guidance (CFG)

`pipeline.py` runs CFG by default at `cfg_scale=7.5`. Each step encodes both the prompt and the empty unconditional prompt, runs both through the U-Net in a single batched forward, and combines:

```
ε_guided = ε_uncond + s · (ε_cond − ε_uncond)
```

---

## Inference pipeline

`generate()` in `pipeline.py` is the entry point. The flow:

1. Tokenize the prompt (and unconditional prompt) → CLIP → `context (2, 77, 768)`
2. Initialize latents as `N(0, I)` at shape `(1, 4, 64, 64)`
3. For each of the 50 inference timesteps:
   - Compute sinusoidal time embedding
   - Duplicate latents for CFG
   - U-Net predicts noise
   - Split into conditional / unconditional, apply CFG
   - Sampler `step()` denoises by one step
4. Run final latents through the VAE decoder → 512 × 512 image
5. Rescale `[-1, 1] → [0, 255]` uint8

Memory management: each model is moved to `device` only while in use, then to `idle_device` (typically CPU) — important for running SD on a single 8 GB GPU.

---

## Setup

```bash
pip install torch numpy pillow tqdm transformers
```

This repo doesn't include the model weights. Download a Stable Diffusion v1.5 checkpoint (e.g., `v1-5-pruned-emaonly.ckpt`) and point `model_loader.preload_models_from_standard_weights(ckpt_path, device)` at it. `model_converter.py` remaps the checkpoint's parameter names to this implementation's naming convention.

For the tokenizer, use HuggingFace's CLIP tokenizer:

```python
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
```

---

## Usage

```python
from PIL import Image
from transformers import CLIPTokenizer
from sd import model_loader, pipeline

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
models = model_loader.preload_models_from_standard_weights(
    "data/v1-5-pruned-emaonly.ckpt", device="cuda"
)

image = pipeline.generate(
    prompt="a photograph of an astronaut riding a horse",
    uncond_prompt="",
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    seed=42,
    models=models,
    device="cuda",
    idle_device="cpu",
    tokenizer=tokenizer,
)

Image.fromarray(image[0]).save("out.png")
```

`sd/demo.ipynb` shows the same flow as a runnable notebook.

---

## Roadmap

- [ ] Add DDIM sampler for faster inference
- [ ] Add image-to-image and inpainting pipelines
- [ ] Benchmark against `diffusers` reference outputs at matched seeds
- [ ] Add training loop on a small custom dataset

---

## Credits

- Architecture and code structure follow Umar Jamil's tutorial: [Coding Stable Diffusion from scratch in PyTorch](https://www.youtube.com/watch?v=ZBKpAp_6TGI) and the companion repo [hkproj/pytorch-stable-diffusion](https://github.com/hkproj/pytorch-stable-diffusion).
- Weight conversion logic adapted from [kjsman/stable-diffusion-pytorch](https://github.com/kjsman/stable-diffusion-pytorch).
- Stable Diffusion v1.5 model weights: CompVis / Stability AI / Runway.

## References

1. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR.
2. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
3. Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance.* arXiv:2207.12598.
4. Radford, A. et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision.* ICML (CLIP).
