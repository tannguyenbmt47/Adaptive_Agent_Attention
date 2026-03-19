#!/usr/bin/env python3
"""
Download ImageNet-1K dataset from HuggingFace datasets library.

Requirements:
    pip install datasets huggingface-hub pillow

Setup:
    1. Request access to ILSVRC/imagenet-1k on HuggingFace at:
       https://huggingface.co/datasets/ILSVRC/imagenet-1k
    2. Get your token from: https://huggingface.co/settings/tokens
    3. Update HF_TOKEN below or pass --token argument
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login

# ⬇️ PUT YOUR HUGGINGFACE TOKEN HERE ⬇️
HF_TOKEN = "hf_YOUR_TOKEN_HERE"

def download_imagenet1k(cache_dir=None, split="train", token=None):
    """
    Download ImageNet-1K dataset from HuggingFace.
    
    Args:
        cache_dir: Directory to cache dataset. If None, uses ~/.cache/huggingface/datasets
        split: 'train' or 'validation'
        token: HuggingFace token (optional, uses HF_TOKEN if not provided)
    """
    
    print(f"🔄 Loading ImageNet-1K [{split} split]...")
    print(f"📦 Cache directory: {cache_dir or '~/.cache/huggingface/datasets'}")
    print()
    
    # Load dataset - automatically downloads and caches
    dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        cache_dir=cache_dir,
        split=split,
        token=token,
        trust_remote_code=True,
        download_mode="force_redownload",  # Set to "reuse_cache_if_exists" to reuse
    )
    
    print(f"✅ Loaded {len(dataset)} images")
    print(f"📊 Dataset info:")
    print(f"   - Features: {dataset.features.keys()}")
    print(f"   - First sample: {dataset[0].keys()}")
    
    return dataset

def download_both_splits(cache_dir=None, token=None):
    """Download both train and validation splits."""
    
    print("=" * 60)
    print("📥 ImageNet-1K Dataset Download")
    print("=" * 60)
    print()
    
    # Use provided token or global HF_TOKEN
    if not token:
        token = HF_TOKEN
    
    # Check token
    if token == "hf_YOUR_TOKEN_HERE" or not token:
        print("❌ No HuggingFace token found!")
        print()
        print("📍 Get your token:")
        print("   1. Visit: https://huggingface.co/settings/tokens")
        print("   2. Create a new token (read access is enough)")
        print("   3. Update HF_TOKEN in this file OR pass --token argument")
        print()
        print("📍 Request ImageNet-1K access:")
        print("   1. Visit: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("   2. Click 'Agree and access repository'")
        sys.exit(1)
    
    # Login with token
    try:
        login(token=token, add_to_git_credential=False)
        print(f"✓ HuggingFace authenticated")
        print()
    except Exception as e:
        print(f"❌ Auth error: {e}")
        sys.exit(1)
    
    # Download train split
    print("1️⃣  Downloading TRAIN split...")
    print("-" * 60)
    train_dataset = download_imagenet1k(cache_dir=cache_dir, split="train", token=token)
    print()
    
    # Download validation split
    print("2️⃣  Downloading VALIDATION split...")
    print("-" * 60)
    val_dataset = download_imagenet1k(cache_dir=cache_dir, split="validation", token=token)
    print()
    
    print("=" * 60)
    print("✅ Download Complete!")
    print("=" * 60)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    print("📍 To use in training, update config:")
    print("   data_format: huggingface")
    print("   dataset_name: ILSVRC/imagenet-1k")
    print(f"   cache_dir: {cache_dir or 'null'}")
    print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ImageNet-1K dataset")
    parser.add_argument("--cache-dir", default=None, 
                        help="Cache directory (default: ~/.cache/huggingface/datasets)")
    parser.add_argument("--split", choices=["train", "validation", "both"], default="both",
                        help="Which split to download")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (if not provided, uses HF_TOKEN from script)")
    
    args = parser.parse_args()
    
    # Use provided token or global HF_TOKEN
    token = args.token or HF_TOKEN
    
    if args.split == "both":
        download_both_splits(cache_dir=args.cache_dir, token=token)
    else:
        print(f"Downloading {args.split} split...")
        download_imagenet1k(cache_dir=args.cache_dir, split=args.split, token=token)
        print("✅ Done!")
