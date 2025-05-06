# src/stitch.py
from PIL import Image
from pathlib import Path

def grid(images, rows=3, cols=3, save_to="output/grid.png"):
    w, h = images[0].size
    canvas = Image.new("RGB", (cols*w, rows*h))
    for idx, im in enumerate(images):
        canvas.paste(im, ((idx % cols)*w, (idx // cols)*h))
    Path(save_to).parent.mkdir(exist_ok=True, parents=True)
    canvas.save(save_to)
    return save_to