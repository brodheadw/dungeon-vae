# train_dataset.py
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class PerilousDataset(Dataset):
    def __init__(self, root, img_size=256):
        self.paths = list(Path(root).glob("*.png"))
        self.tf = T.Compose([
            T.CenterCrop(min(Image.open(p).size)),       # make square
            T.Resize(img_size),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),                                # [0–1]
            T.Normalize([0.5]*3, [0.5]*3)                # [–1,1]
        ])

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img)
