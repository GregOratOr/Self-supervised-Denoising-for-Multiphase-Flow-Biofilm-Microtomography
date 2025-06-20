import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from utils import compute_spectral_image, get_unique_path
from torch.utils.data import Dataset
import argparse
import warnings

class CTScans(Dataset):
    def __init__(self, images, spectrals):    
        self.images = images.astype(np.float32)
        self.spectrals = spectrals.astype(np.complex64)
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.spectrals[idx]
    
    @classmethod
    def create(cls, images_path=Path(""), save_path=Path(""), name="new_dataset")-> None:
        file_paths = sorted([f for f in images_path.iterdir() if f.suffix.lower() == ".png"])
        if not file_paths:
            raise ValueError(f"No PNG files found at - {str(images_path)}")

        print(f"Source Images Path - {images_path}")
        temp_img = np.array(Image.open(file_paths[0]).convert("L"))
        H, W = temp_img.shape
        L = len(file_paths)
        
        imgs = np.zeros((L, H, W), dtype=np.float32)
        specs = np.zeros((L, H, W), dtype=np.complex64)
        
        for idx, file_path in tqdm(enumerate(file_paths), total=len(file_paths), desc="Images"):
            imgs[idx] = (np.array(Image.open(file_path).convert("L"), dtype=np.float32) / 255.0) - 0.5
            specs[idx] = compute_spectral_image(imgs[idx])

        dataset = cls(imgs, specs)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(dataset, get_unique_path(base_path=(save_path / f"{name}.pth")) , pickle_protocol=4)
        print(f"Dataset [{name}.pth] saved at: {save_path}")

if __name__=="__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Create CTScans Dataset.")
    parser.add_argument("-src","--src_dir", type=str, required=True,
                        help="Path to the folder with .png files")
    
    parser.add_argument("-dst", "--dst_dir", type=str, default=".",
                        help="Path to save the dataset file.")
    
    parser.add_argument("-n", "--file_name", type=str, default="new_dataset",
                        help="Name of the dataset file.")

    args = parser.parse_args()

    CTScans.create(images_path=Path(args.src_dir), save_path=Path(args.dst_dir), name=args.file_name)