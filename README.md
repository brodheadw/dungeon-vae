# dungeon-vae
**CLIP‑guided tile generation on top of Stable Diffusion**

Generate cohesive, tile‑able terrain (or city) assets from **text prompts**. The pipeline

1. samples 40 candidates with a lightweight Stable‑Diffusion backbone,
2. embeds each image + the prompt with **CLIP ViT‑L/14**,
3. ranks by cosine similarity, and
4. stitches the **top‑k** into a 3 × 3 preview grid.

The repo also contains a scraper + LoRA fine‑tuning script so you can bias
SD toward a style such as *Watabou / Perilous Shores*.

![sample grid](docs/sample_grid.png)

---

## Features

| Area             | Details |
|------------------|---------|
| **Text‑to‑tiles**| 40 diffusion samples → CLIP re‑rank → top‑9 grid |
| **Dual workflows**| • *Sampler* – fresh diffusion scenes<br> • *Dataset* – assemble from scraped tiles |
| **LoRA ready**   | Fine‑tune on as few as 200 images (DreamBooth/LoRA) |
| **Runs on Metal**| Works out of the box on Apple Silicon (MPS backend) |

---

## Installation

### 1 · One‑click Colab

> <https://colab.research.google.com/github/brodheadw/clip-tile-diffusion/blob/main/notebooks/ClipTileGenerator.ipynb>

### 2 · Local setup (conda)

```bash
# clone & enter
git clone https://github.com/brodheadw/clip-tile-diffusion.git
cd clip-tile-diffusion

# create env
conda env create -f environment.yml
conda activate tilediff
```

### 3 · (Optionally) install bleeding‑edge diffusers
```bash
pip install --upgrade "git+https://github.com/huggingface/diffusers.git"
```

---

## Quick start

### A · Generate 9‑tile grid from scratch
```bash
python demo.py --mode sampler --prompt "lush canyon ruins"
# → output/grid.png
```

### B · Assemble from your scraped tiles
```bash
python demo.py --mode dataset \
               --prompt "ancient crystal desert oasis" \
               --data_dir data/tiles \
               --out output/oasis_grid.png
```

---

## 🔬 Optional – Fine‑tune a LoRA on Watabou maps

```bash
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --instance_data_dir data/perilous \
  --instance_prompt "a fantasy hex‑map in the style of Perilous Shores" \
  --resolution 512 \
  --train_batch_size 1 \
  --num_train_epochs 30 \
  --learning_rate 1e-4 \
  --mixed_precision "no" \
  --output_dir lora-perilous-shores
```
> **Apple Silicon notes**
> * Reduce `--train_batch_size` to 1.
> * You can raise the MPS memory cap with `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` if you still OOM.

---

## Dataset scraping

| Generator                | License                | Export |
|--------------------------|------------------------|---------|
| **Perilous Shores**      | CC BY 3.0 by Watabou   | PNG/SVG |
| **Medieval Fantasy City**| GPL‑3 / CC‑BY‑SA 4.0   | SVG→PNG |

Run the scraper:
```bash
python scrape_perilous.py --maps 100
# collects → data/perilous/perilous_*seed*.png
```

---

## Repo layout
```
clip-tile-diffusion/
├─ demo.py               # CLI entrypoint
├─ scrape_perilous.py    # Watabou map scraper (Selenium/Puppeteer)
├─ train_dreambooth_lora.py
├─ src/
│   ├─ sampler.py        # SD‑Turbo wrapper
│   ├─ ranker.py         # CLIP re‑ranking
│   └─ stitch.py         # grid composer
├─ environment.yml
└─ docs/
    └─ sample_grid.png
```

---

## Roadmap
- Poisson blending for seamless borders
- Automatic elevation + biome masks → Houdini asset
- Unity/Unreal importer
- Gradio web GUI with prompt history & seed browser

---

## License
- **Code** – MIT
- **Generated assets** – follow Watabou’s CC BY 3.0 (credit *@watabou*)

---

## Acknowledgements
- **Watabou** – Procgen Arcana generators
- **Stability AI** – `sd‑turbo`
- **OpenAI** – CLIP
- **HuggingFace** – diffusers & Accelerate