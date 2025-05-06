# demo.py
import argparse, random, sys, os
from pathlib import Path

# Let Python find the local 'src' package
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.sampler import Sampler
from src.ranker  import Ranker
from src.stitch  import grid

def dataset_grid(prompt: str, data_dir="data/tiles", k=9, out="output/watabou_grid.png"):
    # 1. collect all tiles
    paths = list(Path(data_dir).glob("*.png"))
    # 2. rank them by CLIP similarity
    ranker = Ranker()
    top_imgs = ranker.top_k(prompt, paths, k=k)
    # 3. stitch into a grid
    fp = grid(top_imgs, rows=3, cols=3, save_to=out)
    print("✅  Watabou grid saved:", fp)

def main(prompt):
    sampler = Sampler()
    ranker  = Ranker()
    seeds   = [random.randint(0, 99999) for _ in range(40)]

    paths = [sampler.generate(prompt, s, out_dir=f"output/{prompt}") for s in seeds]
    top9  = ranker.top_k(prompt, paths, k=9)
    grid_fp = grid(top9)
    print("✅  Grid saved:", grid_fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="lush canyon ruins")
    args = parser.parse_args()
    main(args.prompt)
