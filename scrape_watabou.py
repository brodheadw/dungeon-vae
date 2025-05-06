import os
import random
from pathlib import Path
from io import BytesIO
from PIL import Image
# if you go the requests route (only works if the generator supports PNG export via URL)
import requests
# or: from selenium import webdriver

DATA_DIR = Path("data/tiles")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_map_image(seed: int, size: int = 512) -> Image.Image:
    # Example for a hypothetical URL‐export endpoint:
    url = f"https://watabou.github.io/your‐generator‐path/?export=png&seed={seed}&size={size}"
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))

def crop_and_save(img: Image.Image, seed: int, tile_w: int = 128, tile_h: int = 128):
    w, h = img.size
    cols = w // tile_w
    rows = h // tile_h
    for i in range(cols):
        for j in range(rows):
            box = (i*tile_w, j*tile_h, (i+1)*tile_w, (j+1)*tile_h)
            tile = img.crop(box)
            tile_fp = DATA_DIR / f"tile_{seed}_{i}_{j}.png"
            tile.save(tile_fp)

def main(n_maps: int = 50):
    for _ in range(n_maps):
        seed = random.randint(0, 999999)
        full_map = fetch_map_image(seed)
        crop_and_save(full_map, seed)

if __name__ == "__main__":
    main(100)  # e.g. grab and crop 100 random maps → ~100*(w/tile_w)*(h/tile_h) tiles