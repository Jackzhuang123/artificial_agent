import os
import torch
from torch.optim import Optimizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

def train_tokenizer(save_path: str, vocab_size: int = 8192):
    """Train a byte‑level BPE tokenizer on TinyStories and save it in HuggingFace‑compatible format."""
    from datasets import load_dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i+batch_size]["text"]

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Train the raw tokenizer with the tokenizers library
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<|pad|>"])
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Save the tokenizer.json inside the directory
    tokenizer_file = os.path.join(save_path, "tokenizer.json")
    tokenizer.save(tokenizer_file)

    # Wrap it in a HuggingFace PreTrainedTokenizerFast and save config files
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token=None,
    )
    hf_tokenizer.save_pretrained(save_path)

# Lion optimizer (from https://arxiv.org/abs/2302.06675)
class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p.data)
                exp_avg = state["exp_avg"]
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.data.add_(torch.sign(update), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss