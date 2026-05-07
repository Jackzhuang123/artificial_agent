import os, yaml, time, json, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import PreTrainedTokenizerFast
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

    model = ModernTransformer(config)
    # 多 GPU 并行
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器与调度器
    if config.get("optimizer", "adamw") == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.01))
    else:
        # Lion 需要低学习率，配置中已单独设置
        optimizer = Lion(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.01))

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
            loss = loss + aux_loss.mean()
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
                logits, aux_loss = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                val_loss += (loss + aux_loss.mean()).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        print(f"{config.get('name','')} - Epoch {epoch+1} Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # 卸下 DataParallel 包装
    if isinstance(model, nn.DataParallel):
        model = model.module
    return train_losses, val_losses, model

# -------------------- 主程序 --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | GPUs: {torch.cuda.device_count()}")

    # ---------- Tokenizer ----------
    tokenizer_path = "tinystories_tokenizer"
    if not os.path.exists(tokenizer_path):
        from utils import train_tokenizer
        print("Tokenizer not found, training a new one...")
        train_tokenizer(tokenizer_path, vocab_size=8192)

    tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token=None,
    )
    tokenizer.save_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size}")

    # ---------- 数据加载与清洗 ----------
    if IS_KAGGLE:
        base = "/kaggle/input/datasets/thedevastator/tinystories-narrative-classification"
        train_csv = os.path.join(base, "train.csv")
        val_csv = os.path.join(base, "validation.csv")
        if os.path.exists(train_csv) and os.path.exists(val_csv):
            print("Loading TinyStories from Kaggle CSVs...")
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            # 提取文本列并清洗
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

    # 清洗：转为字符串，去掉空字符串和仅含空白字符的样本
    train_texts = [str(t) for t in train_texts if pd.notna(t)]
    train_texts = [t for t in train_texts if t.strip()]
    val_texts = [str(t) for t in val_texts if pd.notna(t)]
    val_texts = [t for t in val_texts if t.strip()]

    print(f"Train samples (after cleaning): {len(train_texts)}, Val samples: {len(val_texts)}")

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
        "batch_size": 16,
        "lr": 6e-4,
        "weight_decay": 0.01,
        "optimizer": "adamw",
        "use_rope": True,
        "use_swiglu": True
    }

    # ---------- 对比实验组 ----------
    experiments = {
        "Baseline (MoE+GQA+SwiGLU+RoPE)": base_config,
        "No MoE": {**base_config, "n_experts": 0},
        "MHA (n_kv_heads=8)": {**base_config, "n_kv_heads": 8},
        "Lion optimizer": {**base_config, "optimizer": "lion", "lr": 1e-4, "weight_decay": 0.001},
        "ReLU FFN (no SwiGLU)": {**base_config, "use_swiglu": False},
        "Learned PE (no RoPE)": {**base_config, "use_rope": False},
    }

    EPOCHS = 8
    STEPS_PER_EPOCH = 1200

    results = {}
    best_model_state = None
    best_val = float('inf')
    best_config = None

    start_time = time.time()
    for name, cfg in experiments.items():
        print(f"\n{'='*50}\nRunning: {name}\n{'='*50}")
        cfg["name"] = name
        if "weight_decay" not in cfg:
            cfg["weight_decay"] = 0.01
        train_losses, val_losses, model = train_one_config(
            cfg, tokenizer, device, train_texts, val_texts,
            epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH
        )
        results[name] = {"train": train_losses, "val": val_losses}

        final_val = val_losses[-1]
        if final_val < best_val:
            best_val = final_val
            best_model_state = model.state_dict()
            best_config = cfg

        with open("results_backup.json", "w") as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"\nAll experiments completed in {total_time/60:.1f} minutes.")

    # ---------- 保存结果并绘图 ----------
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

    print("\n======== Training Conclusions ========")
    print(f"{'Configuration':<30} {'Final Val Loss':>15}")
    print("-"*50)
    for name, losses in results.items():
        print(f"{name:<30} {losses['val'][-1]:>15.4f}")
    print("=======================================")

    # ---------- 保存最佳模型并演示对话 ----------
    if best_model_state is not None:
        torch.save(best_model_state, "best_model.pt")
        print(f"\n🏆 Best model saved with Val Loss: {best_val:.4f} (Config: {best_config.get('name')})")

        print("\n--- Chat Demo with Best Model ---")
        model = ModernTransformer(best_config).to(device)
        model.load_state_dict(best_model_state)
        model.eval()

        prompts = [
            "Once upon a time",
            "The little girl",
            "He said",
        ]
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_k=50,
                    eos_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Prompt: {prompt}")
            print(f"Model: {generated}\n")
    else:
        print("No model saved, training may have failed.")

if __name__ == "__main__":
    main()