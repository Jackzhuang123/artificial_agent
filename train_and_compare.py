import os, yaml, time, json, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from model import ModernTransformer
from utils import Lion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# -------------------- 环境与随机种子 --------------------
IS_KAGGLE = os.path.exists("/kaggle/input")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------- Dataset --------------------
class TinyStoriesDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        return input_ids, input_ids.clone()

def collate_fn(batch, pad_token_id=0):
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return input_ids, labels

# -------------------- 训练一个配置 --------------------
def train_one_config(config, tokenizer, device, train_texts, val_texts, epochs, steps_per_epoch):
    train_dataset = TinyStoriesDataset(train_texts, tokenizer, max_len=config["max_seq_len"])
    val_dataset   = TinyStoriesDataset(val_texts, tokenizer, max_len=config["max_seq_len"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    val_loader   = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                              collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    model = ModernTransformer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if config.get("optimizer", "adamw") == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    else:
        optimizer = Lion(model.parameters(), lr=config["lr"], weight_decay=0.01)

    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=steps_per_epoch,
                    desc=f"{config.get('name','')} E {epoch+1}/{epochs}")
        for step, (input_ids, labels) in pbar:
            if step >= steps_per_epoch:
                break
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits, aux_loss = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss + aux_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = train_loss / steps_per_epoch
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                logits, _ = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                val_loss += nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        print(f"{config.get('name','')} - Epoch {epoch+1} Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    del model
    torch.cuda.empty_cache()
    return train_losses, val_losses

# -------------------- 主程序 --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- Tokenizer 加载（修复 HuggingFace 格式不兼容问题） ----------
    tokenizer_path = "tinystories_tokenizer"

    if not os.path.exists(tokenizer_path):
        from utils import train_tokenizer
        print("Training BPE tokenizer (will take a few minutes)...")
        train_tokenizer(tokenizer_path, vocab_size=8192)

    # 安全加载：先尝试 AutoTokenizer，失败则用 PreTrainedTokenizerFast
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        from transformers import PreTrainedTokenizerFast
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_file}")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        tokenizer.pad_token = "<|pad|>"
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.unk_token = None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size}")

    # ---------- 数据加载 ----------
    # ---------- 数据加载（使用 Kaggle 挂载的 CSV）----------
    if IS_KAGGLE:
        # 注意：这里是你实际的挂载路径
        base = "/kaggle/input/datasets/thedevastator/tinystories-narrative-classification"
        train_csv = os.path.join(base, "train.csv")
        val_csv = os.path.join(base, "validation.csv")
        if os.path.exists(train_csv) and os.path.exists(val_csv):
            print("Loading TinyStories from Kaggle CSVs...")
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            train_texts = train_df.iloc[:, 0].tolist()
            val_texts = val_df.iloc[:, 0].tolist()
        else:
            raise FileNotFoundError(f"CSVs not found in {base}")
    else:
        from datasets import load_dataset
        print("Loading TinyStories from HuggingFace (local)...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        texts = dataset.select(range(8000))["text"]
        split = int(0.8 * len(texts))
        train_texts = texts[:split]
        val_texts = texts[split:]

    print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")

    # ---------- 基础配置 ----------
    base_config = {
        "vocab_size": 8192,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "n_kv_heads": 2,
        "multiple_of": 256,
        "ffn_dim_multiplier": 1.3,
        "norm_eps": 1e-6,
        "max_seq_len": 256,
        "dropout": 0.1,
        "n_experts": 2,
        "batch_size": 8,
        "lr": 3e-4,
        "steps": 500,
        "optimizer": "adamw",
        "use_rope": True,
        "use_swiglu": True
    }

    # ---------- 对比实验组 ----------
    experiments = {
        "Baseline (MoE+GQA+SwiGLU+RoPE)": base_config,
        "No MoE": {**base_config, "n_experts": 0},
        "MHA (n_kv_heads=8)": {**base_config, "n_kv_heads": 8},
        "Lion optimizer": {**base_config, "optimizer": "lion"},
        "ReLU FFN (no SwiGLU)": {**base_config, "use_swiglu": False},
        "Learned PE (no RoPE)": {**base_config, "use_rope": False},
    }

    EPOCHS = 2
    STEPS_PER_EPOCH = 200  # 可根据时间调整

    results = {}
    start_time = time.time()
    for name, cfg in experiments.items():
        print(f"\n{'='*50}\nRunning: {name}\n{'='*50}")
        cfg["name"] = name
        train_losses, val_losses = train_one_config(
            cfg, tokenizer, device, train_texts, val_texts,
            epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH
        )
        results[name] = {"train": train_losses, "val": val_losses}
        # 中间备份
        with open("results_backup.json", "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nAll experiments completed in {total_time/60:.1f} minutes.")

    # ---------- 保存最终结果并绘图 ----------
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    plt.figure(figsize=(14, 6))
    for name, losses in results.items():
        epochs_range = range(1, len(losses["train"])+1)
        plt.plot(epochs_range, losses["train"], 'o-', label=f'{name} - Train')
        plt.plot(epochs_range, losses["val"], 's--', label=f'{name} - Val')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training & Validation Loss Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=150)
    print("\n📊 Loss comparison saved to loss_comparison.png")

    # ---------- 输出结论 ----------
    print("\n======== Training Conclusions ========")
    print(f"{'Configuration':<30} {'Final Val Loss':>15}")
    print("-"*50)
    for name, losses in results.items():
        print(f"{name:<30} {losses['val'][-1]:>15.4f}")
    print("=======================================")

if __name__ == "__main__":
    main()