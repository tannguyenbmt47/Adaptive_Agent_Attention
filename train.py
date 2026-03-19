"""
Training script for DeiT-Tiny with Token Agent Attention on Oxford-IIIT Pet
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.deit_token_agent import DeiTTinyTokenAgent
from utils.dataset import build_imagenet_dataset, build_train_transform, build_val_transform


# ==== Load config ====
def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ==== Build model ====
def build_model(cfg):
    mcfg = cfg["model"]
    model = DeiTTinyTokenAgent(
        img_size=mcfg.get("img_size", 224),
        patch_size=mcfg.get("patch_size", 16),
        in_chans=3,
        num_classes=mcfg.get("num_classes", 1000),
        embed_dim=mcfg.get("embed_dim", 192),
        depth=mcfg.get("depth", 12),
        num_heads=mcfg.get("num_heads", 3),
        mlp_ratio=mcfg.get("mlp_ratio", 4.0),
        qkv_bias=mcfg.get("qkv_bias", True),
        drop_rate=mcfg.get("drop_rate", 0.0),
        attn_drop_rate=mcfg.get("attn_drop_rate", 0.0),
        drop_path_rate=mcfg.get("drop_path_rate", 0.1),
        agent_num=mcfg.get("agent_num", 4),
        sparse_ratio=mcfg.get("sparse_ratio", 0.5),
        aggr_ratio=mcfg.get("aggr_ratio", 0.4),
        dim_ratio=mcfg.get("dim_ratio", 0.2)
    )
    return model


# ==== Training one epoch ====
def train_one_epoch(model, loader, optimizer, device, cfg, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} (train)", leave=True)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total_samples += images.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total_samples:.1f}%'
        })
    
    return total_loss / total_samples, 100.0 * correct / total_samples


# ==== Validation ====
@torch.no_grad()
def validate(model, loader, device, cfg, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} (val)", leave=True)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        
        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total_samples += images.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total_samples:.1f}%'
        })
    
    return total_loss / total_samples, 100.0 * correct / total_samples


# ==== Main training script ====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build dataset
    train_set = build_imagenet_dataset(
        cfg["train"]["data_path"], "train", build_train_transform(cfg["train"]["resize"])
    )
    val_set = build_imagenet_dataset(
        cfg["train"]["data_path"], "val", build_val_transform(cfg["val"]["resize"])
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg["train"]["batch_size"], 
        shuffle=True, 
        num_workers=cfg["train"].get("workers", 0), 
        pin_memory=cfg["train"].get("pin_memory", True)
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg["val"]["batch_size"], 
        shuffle=False, 
        num_workers=cfg["train"].get("workers", 0), 
        pin_memory=cfg["train"].get("pin_memory", True)
    )
    
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")
    
    # Build model
    model = build_model(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg["train"]["lr"], 
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"].get("betas", [0.9, 0.999]))
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg["train"]["epochs"], 
        eta_min=cfg["train"]["min_lr"]
    )
    
    # Training loop
    best_acc = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device, cfg, epoch)
        
        # LR schedule
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{cfg['train']['epochs']} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.2e}\n")
        
        # Keep track of best val acc
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"✓ Best val acc: {best_acc:.2f}%\n")
    
    print(f"Training complete! Best val acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
