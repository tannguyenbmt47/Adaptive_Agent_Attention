from torchvision import datasets
import os
from torchvision import transforms
# dataset.py for Adaptive_Agent_Attention
# Minimal loader for Oxford-IIIT Pet, batch small for low-memory GPU

def build_train_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def build_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
def build_imagenet_dataset(data_path, split, transform):
    # For Oxford-IIIT Pet, use ImageFolder structure
    split_path = os.path.join(data_path, split)
    dataset = datasets.ImageFolder(split_path, transform=transform)
    return dataset
def build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

# ========== Parquet Dataset Support ==========
import torch
import glob
import io
try:
    import pyarrow.parquet as pq
    from PIL import Image
except ImportError:
    pq = None
    Image = None

class ParquetDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_path, transform=None):
        self.transform = transform
        table = pq.read_table(parquet_path)
        self.labels = table.column("label").to_pylist()
        img_col = table.column("image")
        sample = img_col[0].as_py()
        if isinstance(sample, dict):
            self.image_bytes = [row.as_py()["bytes"] for row in img_col]
        else:
            self.image_bytes = [row.as_py() for row in img_col]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.image_bytes[idx])).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def build_imagenet_dataset(data_path, split, transform):
    """
    Tự động nhận diện kiểu dữ liệu: ImageFolder hoặc Parquet
    Hỗ trợ cả layout:
      1. data_path/train-*.parquet, data_path/val-*.parquet (flat)
      2. data_path/train/*.parquet, data_path/val/*.parquet (subfolder)
    """
    data_path = data_path.rstrip('/')
    
    # Cách 1: Tìm file flat với pattern split-*.parquet trong data_path
    parquet_files = glob.glob(os.path.join(data_path, f"{split}-*.parquet"))
    
    # Cách 2: Nếu vẫn không tìm thấy, thử tìm trong subfolder data_path/split/
    if not parquet_files:
        split_path = os.path.join(data_path, split)
        parquet_files = glob.glob(os.path.join(split_path, "*.parquet"))
    else:
        split_path = None
    
    # Cách 3: Nếu split là val, thử tìm test-*.parquet
    if not parquet_files and split == "val":
        parquet_files = glob.glob(os.path.join(data_path, "test-*.parquet"))
    
    if pq is not None and parquet_files:
        # Parquet mode
        parquet_datasets = [ParquetDataset(p, transform=transform) for p in sorted(parquet_files)]
        if len(parquet_datasets) == 1:
            return parquet_datasets[0]
        return torch.utils.data.ConcatDataset(parquet_datasets)
    else:
        # ImageFolder mode
        if split_path is None:
            split_path = os.path.join(data_path, split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"Không tìm thấy dữ liệu ở {data_path}/{split}-*.parquet hoặc {split_path}"
            )
        return datasets.ImageFolder(split_path, transform=transform)
