# src/sampler.py
from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

class Sampler:
    def __init__(self, model_name="stabilityai/sd-turbo", half=True):
        dtype = torch.float16 if half else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=dtype
        ).to("mps")
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def generate(self, prompt: str, seed: int, steps: int = 15, out_dir="output"):
        g = torch.Generator(device="mps").manual_seed(seed)
        img = self.pipe(prompt,
                        num_inference_steps=steps,
                        generator=g).images[0]
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        fp = Path(out_dir) / f"{prompt.replace(' ','_')}_{seed}.png"
        img.save(fp)
        return fp