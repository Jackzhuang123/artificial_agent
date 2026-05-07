import os, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import PreTrainedTokenizerFast
from model import ModernTransformer
import numpy as np
from tqdm import tqdm

# -------------------- 环境配置 --------------------
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
            text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        return enc["input_ids"].squeeze(0), enc["input_ids"].squeeze(0).clone()

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return input_ids, labels

# -------------------- 训练函数 --------------------
def train_model(config, tokenizer, device, train_texts, val_texts,
                epochs=12, steps_per_epoch=800, resume_from=None):
    """训练模型，支持从 checkpoint 继续"""
    train_dataset = TinyStoriesDataset(train_texts, tokenizer, max_len=config["max_seq_len"])
    val_dataset   = TinyStoriesDataset(val_texts, tokenizer, max_len=config["max_seq_len"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    model = ModernTransformer(config)
    # 加载已有权重（微调模式）
    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        state_dict = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        init_lr = 2e-4          # 微调学习率
        print(f"Using fine-tuning learning rate: {init_lr}")
    else:
        init_lr = config["lr"]
        print(f"Training from scratch, lr: {init_lr}")

    # 双卡并行
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr,
                                  weight_decay=config.get("weight_decay", 0.01))
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps,
                                                           eta_min=1e-6)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=steps_per_epoch,
                    desc=f"Epoch {epoch+1}/{epochs}")
        for step, (input_ids, labels) in pbar:
            if step >= steps_per_epoch:
                break
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits, aux_loss = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                         shift_labels.view(-1))
            loss = loss + aux_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = train_loss / steps_per_epoch
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                logits, aux_loss = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                             shift_labels.view(-1))
                val_loss += (loss + aux_loss.mean()).item()
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # 保存最佳模型
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(save_model.state_dict(), "final_model.pt")
            print(f"  -> New best model saved with val loss {best_val_loss:.4f}")

    return best_val_loss

# -------------------- 主程序 --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | GPUs: {torch.cuda.device_count()}")

    # ---------- Tokenizer ----------
    tokenizer_path = "tinystories_tokenizer"
    if not os.path.exists(tokenizer_path):
        from utils import train_tokenizer
        print("Training tokenizer...")
        train_tokenizer(tokenizer_path, vocab_size=8192)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tokenizer_path, "tokenizer.json"),
        bos_token="<|endoftext|>", eos_token="<|endoftext|>",
        pad_token="<|pad|>", unk_token=None,
    )
    tokenizer.save_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size}")

    # ---------- 数据 ----------
    if IS_KAGGLE:
        base = "/kaggle/input/datasets/thedevastator/tinystories-narrative-classification"
        train_df = pd.read_csv(os.path.join(base, "train.csv"))
        val_df   = pd.read_csv(os.path.join(base, "validation.csv"))
        train_texts = [str(t) for t in train_df.iloc[:, 0].tolist() if pd.notna(t)]
        val_texts   = [str(t) for t in val_df.iloc[:, 0].tolist() if pd.notna(t)]
        train_texts = [t for t in train_texts if t.strip()]
        val_texts   = [t for t in val_texts if t.strip()]
    else:
        from datasets import load_dataset
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        texts = dataset.select(range(8000))["text"]
        split = int(0.8 * len(texts))
        train_texts, val_texts = texts[:split], texts[split:]
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # ---------- 配置 ----------
    # 选择 1：保持原 d_model=256（微调已有 best_model.pt）
    config_small = {
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
        "n_experts": 0,          # No MoE
        "batch_size": 16,
        "lr": 6e-4,
        "weight_decay": 0.01,
        "use_rope": True,
        "use_swiglu": True,
    }

    # 选择 2：更大的 d_model=512（从头训练，提升容量）
    config_large = {
        "vocab_size": 8192,
        "d_model": 512,
        "n_layers": 4,
        "n_heads": 8,
        "n_kv_heads": 2,
        "multiple_of": 256,
        "ffn_dim_multiplier": 1.3,
        "norm_eps": 1e-6,
        "max_seq_len": 256,
        "dropout": 0.1,
        "n_experts": 0,
        "batch_size": 16,
        "lr": 6e-4,
        "weight_decay": 0.01,
        "use_rope": True,
        "use_swiglu": True,
    }

    # 你可以在这里切换：使用哪个配置，以及是否从已有模型继续
    # -------------------------------------------------
    # 设为 True 则使用 d_model=512 从头训练；False 则基于 best_model.pt 微调
    USE_LARGE_MODEL = False
    # -------------------------------------------------

    if USE_LARGE_MODEL:
        config = config_large
        resume_from = None      # 从头训练
    else:
        config = config_small
        resume_from = "best_model.pt" if os.path.exists("best_model.pt") else None

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"Resume from: {resume_from}")

    # 训练参数
    EPOCHS = 12                 # 足够充分的原/微调
    STEPS_PER_EPOCH = 1200      # 可根据时间调整 (800~1200)

    best_val = train_model(config, tokenizer, device, train_texts, val_texts,
                           epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                           resume_from=resume_from)

    print(f"\nFinal best val loss: {best_val:.4f}")

    # ---------- 加载最终模型并对话 ----------
    final_model = ModernTransformer(config).to(device)
    final_model.load_state_dict(torch.load("final_model.pt", map_location=device))
    final_model.eval()

    print("\n--- Final model dialogue samples ---")
    prompts = [
        "Once upon a time",
        "The brave knight",
        "She opened the door and",
    ]
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            output_ids = final_model.generate(
                input_ids, max_new_tokens=50, temperature=0.7, top_k=50,
                eos_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace('Ġ', ' ')
        print(f"Prompt: {prompt}")
        print(f"Model : {text}\n")

    print("Done! Final model saved as 'final_model.pt'.")

if __name__ == "__main__":
    main()