import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

# ------------------------------------------------------------
# RMSNorm (used instead of LayerNorm)
# ------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

# ------------------------------------------------------------
# RoPE (Rotary Position Embedding) – applied inside attention
# ------------------------------------------------------------
def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0, device: str = "cpu"):
    assert dim % 2 == 0, "RoPE requires even embedding dimension"
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = t * inv_freq.unsqueeze(0)
    emb = freqs.repeat_interleave(2, dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rotary_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

# ------------------------------------------------------------
# Grouped Query Attention (GQA) with causal mask + optional RoPE
# ------------------------------------------------------------
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: Optional[torch.Tensor] = None,
                sin: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE only when cos/sin are provided
        if cos is not None and sin is not None:
            q, k = apply_rotary_emb(q, k, cos, sin)

        # GQA: repeat KV heads
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + mask
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(out)

# ------------------------------------------------------------
# SwiGLU Feed‑Forward Network
# ------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w3(x)) * self.w1(x)))

# ------------------------------------------------------------
# Standard ReLU Feed‑Forward Network (for ablation)
# ------------------------------------------------------------
class StandardFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.relu(self.w1(x))))

# ------------------------------------------------------------
# Mixture of Experts (optional, enabled via config)
# ------------------------------------------------------------
class Top1Gate(nn.Module):
    def __init__(self, d_model: int, n_experts: int, noise_std: float = 1e-2):
        super().__init__()
        self.w_gate = nn.Linear(d_model, n_experts, bias=False)
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor):
        logits = self.w_gate(x)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        expert_weights, expert_indices = torch.topk(logits, 1, dim=-1)
        return expert_weights.squeeze(-1), expert_indices.squeeze(-1)

class MoELayer(nn.Module):
    def __init__(self, d_model: int, n_experts: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.gate = Top1Gate(d_model, n_experts)
        self.experts = nn.ModuleList([SwiGLU(d_model, hidden_dim, dropout) for _ in range(n_experts)])
        self.aux_loss_weight = 1e-2

    def forward(self, x: torch.Tensor):
        batch, seq_len, d_model = x.shape
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        total_tokens = batch * seq_len
        expert_counts = torch.zeros(self.n_experts, device=x.device)
        for expert_id in range(self.n_experts):
            mask = (indices == expert_id)
            expert_counts[expert_id] = mask.sum().item()
            if mask.any():
                expert_input = x[mask]
                expert_output = self.experts[expert_id](expert_input)
                y[mask] = expert_output * weights[mask].unsqueeze(-1)

        f_i = expert_counts / total_tokens
        p_i = torch.softmax(self.gate.w_gate(x.mean(dim=1)), dim=-1)
        aux_loss = torch.sum(f_i * p_i) * self.n_experts
        return y, aux_loss * self.aux_loss_weight

# ------------------------------------------------------------
# Decoder Layer (pre‑norm + residual)
# ------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, ffn_hidden: int,
                 dropout: float, use_moe: bool = False, n_experts: int = 2,
                 use_swiglu: bool = True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)

        self.ffn_norm = RMSNorm(d_model)
        if use_moe:
            self.ffn = MoELayer(d_model, n_experts, ffn_hidden, dropout)
        elif use_swiglu:
            self.ffn = SwiGLU(d_model, ffn_hidden, dropout)
        else:
            self.ffn = StandardFFN(d_model, ffn_hidden, dropout)

        self.dropout = nn.Dropout(dropout)
        self.use_moe = use_moe

    def forward(self, x: torch.Tensor, cos: Optional[torch.Tensor] = None,
                sin: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        attn_out = self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.dropout(attn_out)

        ffn_input = self.ffn_norm(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(ffn_input)
            x = x + self.dropout(ffn_out)
            return x, aux_loss
        else:
            ffn_out = self.ffn(ffn_input)
            x = x + self.dropout(ffn_out)
            return x, torch.tensor(0.0, device=x.device)

# ------------------------------------------------------------
# Full Decoder‑Only Transformer
# ------------------------------------------------------------
class ModernTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.d_model = config["d_model"]
        self.n_layers = config["n_layers"]
        self.max_seq_len = config["max_seq_len"]
        self.dropout = nn.Dropout(config["dropout"])

        # Token embedding + weight tying
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # Positional encoding
        self.use_rope = config.get("use_rope", True)
        if not self.use_rope:
            self.pos_embed = nn.Parameter(torch.randn(self.max_seq_len, self.d_model) * 0.02)
            self.cos = None
            self.sin = None
        else:
            self.pos_embed = None
            cos, sin = precompute_rope_freqs(self.d_model // config["n_heads"], self.max_seq_len)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

        ffn_hidden = int(config["ffn_dim_multiplier"] * 4 * self.d_model)
        multiple_of = config.get("multiple_of", 256)
        ffn_hidden = multiple_of * ((ffn_hidden + multiple_of - 1) // multiple_of)

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=self.d_model,
                n_heads=config["n_heads"],
                n_kv_heads=config["n_kv_heads"],
                ffn_hidden=ffn_hidden,
                dropout=config["dropout"],
                use_moe=config.get("n_experts", 0) > 0,
                n_experts=config.get("n_experts", 2),
                use_swiglu=config.get("use_swiglu", True)
            ) for _ in range(self.n_layers)
        ])
        self.final_norm = RMSNorm(self.d_model, eps=config["norm_eps"])

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, L = input_ids.shape
        assert L <= self.max_seq_len, f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}"

        x = self.embed(input_ids) * math.sqrt(self.d_model)

        if not self.use_rope:
            x = x + self.pos_embed[:L, :].unsqueeze(0)      # absolute position embedding
            cos, sin = None, None
        else:
            cos = self.cos[:L, :].to(x.device)
            sin = self.sin[:L, :].to(x.device)

        x = self.dropout(x)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, aux_loss = layer(x, cos, sin, mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, eos_token_id: Optional[int] = None):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
            logits, _ = self.forward(input_ids)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                top_k_vals, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
                probs = torch.zeros_like(next_logits).scatter_(-1, top_k_indices, F.softmax(top_k_vals, dim=-1))
            else:
                probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return input_ids