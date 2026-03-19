import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from models.token_agent_attention import TokenAgentAttention
from utils.dataset import build_imagenet_dataset, build_train_transform, build_val_transform

# ==== Load config ====
def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# ==== Build model ====
def build_model(cfg):
    mcfg = cfg["model"]
    tacfg = cfg.get("token_agent", {})
    model = TokenAgentAttention(
        dim=mcfg["embed_dim"],
        num_heads=mcfg["num_heads"],
        qkv_bias=True,
        attn_drop=mcfg.get("attn_dropout", 0.0),
        proj_drop=mcfg.get("dropout", 0.0),
        agent_num=mcfg["agent_num"],
        window=int(mcfg["img_size"] // mcfg["patch_size"]),
        sparse_ratio=tacfg.get("sparse_ratio", 0.5),
        aggr_ratio=tacfg.get("aggr_ratio", 0.4),
        dim_ratio=tacfg.get("dim_ratio", 0.2)
    )
    return model

# ==== Main training script ====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Build dataset
    train_set, val_set = build_imagenet_dataset(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["system"]["num_workers"])
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["system"]["num_workers"])

    # Build model
    model = build_model(cfg).cuda()
    model.train()

    # Optimizer, loss, scheduler setup (giống Linear_Integral)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"], betas=tuple(cfg["train"].get("betas", [0.9, 0.999])))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"], eta_min=cfg["train"]["min_lr"])
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop (rút gọn, bạn có thể mở rộng thêm)
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} - Loss: {loss.item():.4f}")
        # TODO: Thêm validate, save checkpoint, augmentation, mixup, cutmix, ...

if __name__ == "__main__":
    main()
