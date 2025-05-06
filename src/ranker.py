# src/ranker.py
import torch, clip
from PIL import Image
from pathlib import Path

class Ranker:
    def __init__(self, clip_variant="ViT-L/14"):
        self.model, self.preprocess = clip.load(clip_variant, device="mps")

    @torch.inference_mode()
    def top_k(self, prompt: str, paths, k=9):
        text = clip.tokenize([prompt]).to("mps")
        tfeat = self.model.encode_text(text).float()

        imgs, ifeats = [], []
        for p in paths:
            im = Image.open(p).convert("RGB")
            imgs.append(im)
            ifeats.append(self.model.encode_image(
                self.preprocess(im).unsqueeze(0).to("mps"))[0])
        ifeats = torch.stack(ifeats).float()
        sims = (ifeats @ tfeat.T).squeeze()
        best_idx = sims.topk(k).indices.tolist()
        return [imgs[i] for i in best_idx]