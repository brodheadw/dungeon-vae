# clip-tile-diffusion  
**CLIP-Guided Tile Diffusion**  
Generate cohesive, tile-able terrain (or city) assets from **text prompts** using a Stable-Diffusion backbone, CLIP guidance, and a tiny dataset scraped from **Watabou’s Procgen Arcana** generators.

![sample grid](docs/sample_grid.png)

---

## 🚀 Features  
- **Text-to-tiles**: 40 diffusion candidates → rank by CLIP → top-_k_ grid  
- **Dual workflows**:  
  - **Sampler mode**: generate fresh diffusion scenes  
  - **Dataset mode**: assemble from your own scraped tile library  
- **LoRA-ready**: fine-tune on 200+ Watabou images  
- **Portable**: runs locally or in Colab  

---

## 🔧 Installation

### 1. One-click Colab  
[Open in Colab](https://colab.research.google.com/github/your-repo/clip-tile-diffusion/blob/main/notebooks/ClipTileGenerator.ipynb)

### 2. Local setup
```bash
git clone https://github.com/your-repo/clip-tile-diffusion.git
cd clip-tile-diffusion
conda env create -f environment.yml
conda activate tilediff
```

---

## 🎯 Usage

### a) Diffusion + CLIP (default)
```bash
python demo.py --mode sampler --prompt "lush canyon ruins"
```
Generates 40 SD-turbo scenes, ranks them with CLIP ViT-L/14, and stitches the best 9 into `output/grid.png`.

### b) Assemble from scraped tiles
```bash
python demo.py --mode dataset \
               --prompt "ancient crystal desert oasis" \
               --data_dir data/tiles \
               --out output/oasis_grid.png
open output/oasis_grid.png
```
Loads `data/tiles/*.png`, ranks by semantic similarity, and stitches the top 9.

---

## 📦 Dataset

| Generator                              | License               | Specs        |
|----------------------------------------|-----------------------|--------------|
| Watabou – Medieval Fantasy City        | GPL-3 / CC-BY-SA 4.0   | SVG → PNG 512² |
| Watabou – Dungeon Generator            | GPL-3 / CC-BY-SA 4.0   | Darker palette |

Run the scraper to harvest ~500 tiles in ~10 min:
```bash
python scraper/scrape_watabou.py --maps 100
```
Uses **pyppeteer** to load the canvas, capture, crop into 512×512 PNGs, and dump into `data/tiles/`.

---

## 🛠️ Model Pipeline

1. **Stable Diffusion** (`stabilityai/sd-turbo`)  
2. **CLIP** (ViT-L/14) for prompt/image encoding  
3. **Re-rank** 40 candidates → keep top _k_ for cohesion  
4. **Grid stitcher** builds a seamless 3 × 3 preview  

> *Optional:* fine-tune a **LoRA** on scraped tiles:  
> ```bash
> accelerate launch train_lora.py \
>   --pretrained_model sd-turbo \
>   --dataset_dir data/tiles \
>   --resolution 512 \
>   --lr 1e-4
> ```

---

## 🗂️ Repo Structure

```
clip-tile-diffusion/
├─ notebooks/                   ← Colab demo
│   └─ ClipTileGenerator.ipynb
├─ scraper/                     ← Watabou scraper
│   └─ scrape_watabou.py
├─ demo.py                      ← CLI entrypoint
├─ train_lora.py                ← optional fine-tuning
├─ environment.yml
└─ docs/
   └─ sample_grid.png
```

---

## 🛣️ Roadmap  

- Edge-aware Poisson blending for seamless tiling  
- Automated heightmap & biome extraction → Houdini asset  
- Unity/Unreal importer + planet-scale UV paging  
- Web GUI (Gradio) with prompt history & seed browser  

---

## 📄 License

**Code**: MIT  
**Generated assets** (from Watabou): CC BY 3.0 (credit **@watabou**)

---

## 🙏 Acknowledgements

- **Watabou** – Procgen Arcana generators  
- **Stability AI** – `sd-turbo`  
- **OpenAI** – CLIP  
- **Hugging Face diffusers** community