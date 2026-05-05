import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from model import ModernTransformer
from utils import train_tokenizer, Lion
import wandb  # optional, remove if not installed
from model import ModernTransformer

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def collate_fn(batch, tokenizer, max_len=512):
    texts = [item["text"] for item in batch]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    input_ids = encodings["input_ids"]
    # For causal LM, labels are input_ids shifted right (teacher forcing)
    labels = input_ids.clone()
    return input_ids, labels

def train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    for step, (input_ids, labels) in enumerate(dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits, aux_loss = model(input_ids)
        # Shift logits/labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss + aux_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        if step % 50 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    for input_ids, labels in dataloader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits, aux_loss = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- Tokenizer ----------
    tokenizer_path = "tinystories_tokenizer"
    if not os.path.exists(tokenizer_path):
        print("Training BPE tokenizer on TinyStories...")
        train_tokenizer(tokenizer_path, vocab_size=config["vocab_size"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    # ---------- Dataset ----------
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.select(range(20000))  # use 20k stories for speed
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                            collate_fn=lambda b: collate_fn(b, tokenizer, config["max_seq_len"]))

    # ---------- Model ----------
    model = ModernTransformer(config).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # ---------- Optimizer ----------
    if config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    elif config["optimizer"] == "lion":
        optimizer = Lion(model.parameters(), lr=config["lr"], weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["steps"])

    # ---------- Training ----------
    print("Starting training...")
    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0)
        print(f"Epoch {epoch}: train loss = {train_loss:.4f}")

        # Generate a sample
        prompt = "Once upon a time"
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.7, top_k=50)
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Sample: {generated}\n")

    # Save final model
    torch.save(model.state_dict(), "final_model.pt")
    print("Training complete.")

if __name__ == "__main__":
    main()