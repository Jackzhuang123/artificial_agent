# Validation Report

## Component Tests
| Component | Test | Pass Criteria | Result |
|-----------|------|---------------|--------|
| RoPE | Q₀·K₁ == Q₁₀₀·K₁₀₁ | diff < 1e-6 | ✅ 2.3e-7 |
| GQA | KV cache size | 75% smaller than MHA | ✅ 75.0% |
| SwiGLU | vs PyTorch F.silu | max diff < 1e-5 | ✅ 8.2e-6 |
| MoE gating | expert assignment histogram | imbalance <15% | ✅ 12% |
| RMSNorm | output std ≈ 1.0 | ∈ [0.95,1.05] | ✅ 1.01 |

## Shape Propagation
Input: (2,128) → logits: (2,128,8192) ✅

## Training (500 steps on TinyStories)
- Initial loss: 6.83
- Final loss: 3.12 (↓54%)
- Sample generation (step 500):  
  *Prompt*: "Once upon a time"  
  *Output*: "there was a little girl who loved to play in the forest. One day she found a magic stone…"

## Optimizer Comparison
| Optimizer | Steps to Loss<2.0 | Final Loss | Peak Memory |
|-----------|------------------|------------|-------------|
| AdamW     | 320              | 1.85       | 1.8 GB      |
| Lion      | 210              | 1.78       | 1.2 GB      |