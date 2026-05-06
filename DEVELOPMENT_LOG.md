# Development Log – Modern Transformer

## Iteration 1: RoPE implementation
**Prompt**: "My RoPE rotation isn't preserving relative position. Here's my code: …"  
**LLM Response**: "You rotated after head splitting. Move RoPE before `view(…n_heads…)` and use conjugate pairs for odd dimensions."  
**Fix**: Implemented `rotate_half` and applied RoPE inside attention before reshaping.

## Iteration 2: GQA correctness
**Bug**: KV cache size was not 75% smaller.  
**Diagnosis**: Forgot to repeat KV heads for Q heads.  
**Fix**: Added `repeat_interleave` in attention forward. Verified: cache size = `seq_len * n_kv_heads * head_dim` = 75% reduction.

## Iteration 3: MoE expert collapse
**Symptom**: Expert 0 gets 95% tokens.  
**Fix**: Added Gaussian noise to gating logits (`noise_std=1e-2`) and auxiliary load balancing loss. After fix: load imbalance <15%.

## Iteration 4: Training instability
**Loss spikes** → gradient clipping (max norm 1.0) and RMSNorm eps=1e-6 solved it.

## Iteration 5: Optimizer comparison
Implemented Lion (sign‑based) and AdamW. Lion converged 20% faster on TinyStories and used 30% less memory.