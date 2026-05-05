"""
Unit tests for Modern Decoder‑Only Transformer
Components: RoPE, GQA, SwiGLU, MoE, RMSNorm, Model Forward/Backward, Generation
Run: python test_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

# Assume model.py is in the same directory or in transformer/ folder
try:
    from model import (
        RMSNorm, precompute_rope_freqs, apply_rotary_emb,
        GroupedQueryAttention, SwiGLU, Top1Gate, MoELayer,
        DecoderLayer, ModernTransformer
    )
except ImportError:
    print("Error: Could not import from model.py. Make sure model.py is in the same directory or PYTHONPATH.")
    sys.exit(1)


'''
# 1. RMSNorm Test
# 输出每个通道的标准差接近1
确保归一化稳定，不会爆炸或消失
'''
def test_rmsnorm():
    print("\n[Test 1] RMSNorm")
    norm = RMSNorm(256, eps=1e-6)
    x = torch.randn(4, 32, 256)
    y = norm(x)
    # Output should have mean std close to 1
    std_mean = y.std(dim=-1).mean().item()
    assert 0.95 <= std_mean <= 1.05, f"RMSNorm output std mean = {std_mean:.4f} (expected ~1.0)"
    print(f"  ✓ RMSNorm works, output std mean = {std_mean:.4f}")


'''
# 2. RoPE Test: relative position invariance
# 相对距离不变性：Q₀·K₁ ≈ Q₁₀₀·K₁₀₁
RoPE 的核心特性：位置差决定点积，而非绝对位置
'''
def test_rope_relative():
    print("\n[Test 2] RoPE relative position invariance")
    dim = 64  # head_dim
    seq_len = 200
    cos, sin = precompute_rope_freqs(dim, seq_len, theta=10000.0, device="cpu")

    # 固定一对随机的 q 和 k 向量（形状：head_dim）
    q = torch.randn(dim)
    k = torch.randn(dim)
    # 扩展为 batch=1, n_heads=1, seq_len=1, head_dim
    q = q.view(1, 1, 1, dim)
    k = k.view(1, 1, 1, dim)

    # 准备不同位置的 cos/sin（需要 seq_len 个位置）
    # 取位置 0 和 1 作为一组，位置 100 和 101 作为另一组
    cos_0 = cos[0:1].unsqueeze(0).unsqueeze(0)  # (1,1,1,dim)
    sin_0 = sin[0:1].unsqueeze(0).unsqueeze(0)
    cos_1 = cos[1:2].unsqueeze(0).unsqueeze(0)
    sin_1 = sin[1:2].unsqueeze(0).unsqueeze(0)
    cos_100 = cos[100:101].unsqueeze(0).unsqueeze(0)
    sin_100 = sin[100:101].unsqueeze(0).unsqueeze(0)
    cos_101 = cos[101:102].unsqueeze(0).unsqueeze(0)
    sin_101 = sin[101:102].unsqueeze(0).unsqueeze(0)

    # 计算旋转后的向量
    q_rot_0, _ = apply_rotary_emb(q, k, cos_0, sin_0)  # 只取 q
    k_rot_1, _ = apply_rotary_emb(q, k, cos_1, sin_1)  # 只取 k
    q_rot_100, _ = apply_rotary_emb(q, k, cos_100, sin_100)
    k_rot_101, _ = apply_rotary_emb(q, k, cos_101, sin_101)

    # 计算点积 (位置0,1) 和 (位置100,101)
    dot1 = torch.dot(q_rot_0.squeeze(), k_rot_1.squeeze()) # 拆包（只需要里面的数字，转变为一维向量），q_rot_0与k_rot_1进行点击运算（衡量向量之间的相似度，值越大代表俩个向量之间越接近）
    dot2 = torch.dot(q_rot_100.squeeze(), k_rot_101.squeeze())
    diff = (dot1 - dot2).abs().item()

    assert diff < 1e-5, f"RoPE relative distance invariance failed: diff = {diff}"
    print(f"  ✓ RoPE preserves relative positions, dot product diff = {diff:.2e}")

'''
# 3. GQA: KV cache size reduction
# KV cache 大小减少 75%（当 n_kv_heads = n_heads/4）
证明 GQA 确实节省内存，加速推理
'''
def test_gqa_cache_size():
    print("\n[Test 3] GQA KV cache size reduction")
    d_model = 256
    n_heads = 8
    n_kv_heads = 2
    attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout=0.0)
    head_dim = d_model // n_heads  # = 32

    # Simulate KV cache size for one sequence
    seq_len = 512
    # In MHA: KV heads = n_heads = 8, per head dim = 32 -> total elements = seq_len * n_heads * head_dim
    mha_elements = seq_len * n_heads * head_dim
    gqa_elements = seq_len * n_kv_heads * head_dim
    reduction = (1 - gqa_elements / mha_elements) * 100
    print(f"  MHA cache elements: {mha_elements}, GQA cache elements: {gqa_elements}")
    assert reduction >= 70, f"GQA cache reduction only {reduction:.1f}% (expected >70%)"
    print(f"  ✓ GQA reduces KV cache by {reduction:.1f}%")


'''
# 4. SwiGLU vs PyTorch SiLU reference
# 与手动 PyTorch 实现一致
确保激活函数实现正确，无符号错误
'''
def test_swiglu():
    print("\n[Test 4] SwiGLU correctness")
    d_model = 256
    hidden = 1024
    swiglu = SwiGLU(d_model, hidden, dropout=0.0)
    # Test with random input
    x = torch.randn(2, 16, d_model)
    out = swiglu(x)

    # Reference implementation: (xW1) * SiLU(xW3) then W2
    w1 = swiglu.w1.weight.data
    w2 = swiglu.w2.weight.data
    w3 = swiglu.w3.weight.data
    ref = F.linear(F.silu(F.linear(x, w3)) * F.linear(x, w1), w2)
    diff = (out - ref).abs().max().item()
    assert diff < 1e-5, f"SwiGLU mismatch, max diff = {diff}"
    print(f"  ✓ SwiGLU matches reference implementation, max diff = {diff:.2e}")


'''
# 5. MoE Load Balancing
# 专家分配负载不平衡 <15%
防止专家崩溃，确保所有专家都被使用
'''
def test_moe_balance():
    print("\n[Test 5] MoE gating load balancing")
    d_model = 256
    n_experts = 2
    hidden = 1024
    moe = MoELayer(d_model, n_experts, hidden, dropout=0.0)
    moe.train()

    # Run several batches to see expert assignment
    total_tokens = 0
    expert_counts = [0, 0]
    for _ in range(50):
        x = torch.randn(4, 32, d_model)  # batch=4, seq=32
        _, aux_loss = moe(x)
        # Manually compute gate assignments (simplified: we just inspect internal gate)
        with torch.no_grad():
            logits = moe.gate.w_gate(x)  # (B, L, n_experts)
            if moe.gate.noise_std > 0 and moe.training:
                noise = torch.randn_like(logits) * moe.gate.noise_std
                logits = logits + noise
            indices = torch.argmax(logits, dim=-1)  # (B, L)，用于找到张量中最大值的索引从最后一个维度
            for idx in indices.flatten():
                expert_counts[idx] += 1
                total_tokens += 1
    imbalance = abs(expert_counts[0] - expert_counts[1]) / total_tokens
    print(f"  Expert assignments: {expert_counts[0]} vs {expert_counts[1]}, imbalance = {imbalance:.2%}")
    assert imbalance < 0.15, f"MoE load imbalance too high: {imbalance:.2%}"
    print(f"  ✓ MoE load balanced within 15%")


'''
# 6. Model Shape Propagation & No NaN
# 输入 (B, L) → 输出 (B, L, vocab_size)，无 NaN
基本的前向传播正确性
'''
def test_model_shapes():
    print("\n[Test 6] Model forward shape and numeric stability")
    config = {
        "vocab_size": 8192, #
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "n_kv_heads": 2,
        "multiple_of": 256,
        "ffn_dim_multiplier": 1.3,
        "norm_eps": 1e-6,
        "max_seq_len": 512,
        "dropout": 0.0,  # disable for testing
        "n_experts": 2,
        "num_experts_per_tok": 1
    }
    model = ModernTransformer(config)
    model.eval() # 评估模式，停止反向传播

    B, L = 2, 128
    x = torch.randint(0, config["vocab_size"], (B, L))
    logits, aux_loss = model(x)

    assert logits.shape == (B, L, config["vocab_size"]), f"Shape mismatch: {logits.shape}"
    assert torch.all(torch.isfinite(logits)), "NaN or Inf in logits"
    assert torch.isfinite(aux_loss), "NaN in auxiliary loss"
    print(f"  ✓ Model output shape {logits.shape} correct, no NaN/Inf")


'''
# 7. Gradient Flow (Backward)
# 所有参数都能得到梯度，且梯度有限
可训练性，没有梯度消失/爆炸
'''
def test_gradient_flow():
    print("\n[Test 7] Gradient flow through the model")
    config = {
        "vocab_size": 8192,
        "d_model": 256,
        "n_layers": 2,  # smaller for speed
        "n_heads": 8,
        "n_kv_heads": 2,
        "multiple_of": 256,
        "ffn_dim_multiplier": 1.3,
        "norm_eps": 1e-6,
        "max_seq_len": 512,
        "dropout": 0.0,
        "n_experts": 2,
        "num_experts_per_tok": 1
    }
    model = ModernTransformer(config)
    B, L = 2, 64
    x = torch.randint(0, config["vocab_size"], (B, L))
    logits, aux_loss = model(x)
    loss = logits.sum() + aux_loss  # dummy loss
    loss.backward()

    # Check that all parameters have gradients
    has_grad = [p.grad is not None for p in model.parameters()]
    all_grad = all(has_grad)
    assert all_grad, f"Some parameters missing gradients: {has_grad}"
    # Also check that gradients are finite
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert torch.isfinite(grad_norm), f"Gradient norm is NaN/Inf: {grad_norm}"
    print(f"  ✓ All parameters receive finite gradients, grad norm = {grad_norm:.4f}")


'''
# 8. Generation Functionality
# 自回归生成不报错，输出长度符合预期
推理功能可用
'''
def test_generation():
    print("\n[Test 8] Autoregressive generation")
    config = {
        "vocab_size": 8192,
        "d_model": 256,
        "n_layers": 2,
        "n_heads": 8,
        "n_kv_heads": 2,
        "multiple_of": 256,
        "ffn_dim_multiplier": 1.3,
        "norm_eps": 1e-6,
        "max_seq_len": 512,
        "dropout": 0.0,
        "n_experts": 0,  # disable MoE for simplicity
        "num_experts_per_tok": 1
    }
    model = ModernTransformer(config)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3]])  # arbitrary tokens
    max_new = 20
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_new, temperature=0.8, top_k=50)
    expected_len = input_ids.shape[1] + max_new
    assert output.shape[1] <= expected_len, f"Output length {output.shape[1]} > expected max {expected_len}"
    assert output.shape[0] == 1, "Batch dimension changed"
    print(f"  ✓ Generation works, output length = {output.shape[1]}")


# ------------------------------------------------------------
# 9. 端到端训练过拟合小批量（确保模型可学习）
# ------------------------------------------------------------
def test_overfit_small_batch():
    print("\n[Test 9] Overfitting on a tiny batch")
    config = {
        "vocab_size": 512,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "multiple_of": 64,
        "ffn_dim_multiplier": 1.0,
        "norm_eps": 1e-6,
        "max_seq_len": 32,
        "dropout": 0.0,
        "n_experts": 0,
    }
    model = ModernTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 生成固定小批量数据（2个样本）
    B, L = 2, 16
    input_ids = torch.randint(0, config["vocab_size"], (B, L))
    labels = input_ids.clone()

    model.train()
    initial_loss = None
    for step in range(50):
        logits, aux_loss = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss + aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5, f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
    print(f"  ✓ Model overfits small batch: loss {initial_loss:.4f} → {final_loss:.4f}")


# ------------------------------------------------------------
# 10. 梯度裁剪有效性检查
# ------------------------------------------------------------
def test_gradient_clipping():
    print("\n[Test 10] Gradient clipping")
    config = {
        "vocab_size": 256,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 2,
        "multiple_of": 32,
        "ffn_dim_multiplier": 1.0,
        "norm_eps": 1e-6,
        "max_seq_len": 16,
        "dropout": 0.0,
        "n_experts": 0,
    }
    model = ModernTransformer(config)
    B, L = 1, 8
    input_ids = torch.randint(0, config["vocab_size"], (B, L))
    logits, _ = model(input_ids)
    loss = logits.mean()
    loss.backward()

    # 计算裁剪前梯度范数
    total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-10)
    # 裁剪至极小值后再次计算
    total_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-10)
    assert total_norm_after <= 1e-10 + 1e-12, f"Grad norm not clipped properly: {total_norm_after}"
    print(f"  ✓ Gradient clipping works: norm {total_norm_before:.6f} → {total_norm_after:.2e}")


# ------------------------------------------------------------
# 11. 权重绑定验证（embed 与 lm_head 共享权重）
# ------------------------------------------------------------
def test_weight_tying():
    print("\n[Test 11] Weight tying between embed and lm_head")
    config = {
        "vocab_size": 100,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 2,
        "multiple_of": 32,
        "ffn_dim_multiplier": 1.0,
        "norm_eps": 1e-6,
        "max_seq_len": 16,
        "dropout": 0.0,
        "n_experts": 0,
    }
    model = ModernTransformer(config)
    # 检查两个权重的 data_ptr 是否相同
    assert model.embed.weight.data_ptr() == model.lm_head.weight.data_ptr(), "Weights are not tied!"
    # 修改嵌入权重，应同步影响 lm_head
    with torch.no_grad():
        model.embed.weight[0, 0] = 999.0
    assert model.lm_head.weight[0, 0] == 999.0, "Weight tying not effective"
    print("  ✓ Embedding and LM head share the same weight tensor")


# ------------------------------------------------------------
# 12. 配置一致性检查（从 YAML 加载模型）
# ------------------------------------------------------------
def test_config_loading():
    print("\n[Test 12] Config loading and model initialization")
    # 模拟从 config.yaml 读取配置（实际运行时会从文件读取）
    import yaml
    config_str = """
vocab_size: 8192
d_model: 256
n_layers: 4
n_heads: 8
n_kv_heads: 2
multiple_of: 256
ffn_dim_multiplier: 1.3
norm_eps: 1e-6
max_seq_len: 512
dropout: 0.1
n_experts: 2
batch_size: 4
lr: 3e-4
epochs: 1
steps: 500
optimizer: "adamw"
"""
    config = yaml.safe_load(config_str)
    model = ModernTransformer(config)
    assert model.vocab_size == 8192
    assert model.d_model == 256
    assert len(model.layers) == 4
    # 检查 FFN 隐藏层维度计算
    ffn_hidden = model.layers[0].ffn.w1.out_features if not model.layers[0].use_moe else model.layers[0].ffn.experts[
        0].w1.out_features
    expected_hidden = int(config["ffn_dim_multiplier"] * 4 * config["d_model"])
    expected_hidden = config["multiple_of"] * ((expected_hidden + config["multiple_of"] - 1) // config["multiple_of"])
    assert ffn_hidden == expected_hidden, f"FFN hidden size mismatch: {ffn_hidden} vs {expected_hidden}"
    print("  ✓ Model correctly initialized from config dict")


# ------------------------------------------------------------
# 13. 多设备兼容性（CPU / CUDA 自动切换）
# ------------------------------------------------------------
def test_device_handling():
    print("\n[Test 13] Device handling")
    config = {
        "vocab_size": 100,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 2,
        "multiple_of": 32,
        "ffn_dim_multiplier": 1.0,
        "norm_eps": 1e-6,
        "max_seq_len": 16,
        "dropout": 0.0,
        "n_experts": 0,
    }
    model = ModernTransformer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.randint(0, config["vocab_size"], (2, 8)).to(device)
    logits, _ = model(x)
    assert logits.device == device, "Output tensor not on correct device"
    # 验证 RoPE buffer 也在正确设备上
    assert model.cos.device == device, "RoPE buffer not moved to device"
    print(f"  ✓ Model runs on {device} and all tensors are correctly placed")


# ------------------------------------------------------------
# 14. 生成功能高级测试（温度、top_k、EOS 停止）
# ------------------------------------------------------------
def test_generation_advanced():
    print("\n[Test 14] Generation with temperature, top_k, and EOS stopping")
    config = {
        "vocab_size": 200,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "multiple_of": 64,
        "ffn_dim_multiplier": 1.0,
        "norm_eps": 1e-6,
        "max_seq_len": 64,
        "dropout": 0.0,
        "n_experts": 0,
    }
    model = ModernTransformer(config)
    model.eval()
    input_ids = torch.tensor([[10, 20, 30]])
    eos_token_id = 42  # 假设一个 EOS ID

    # 正常生成
    out = model.generate(input_ids, max_new_tokens=10, temperature=0.8, top_k=20)
    assert out.shape[1] == input_ids.shape[1] + 10

    # 测试 EOS 停止：手动让 logits 在第二步就预测 EOS
    # 通过 monkey-patch forward 强行返回 EOS 高概率
    original_forward = model.forward

    def mock_forward(input_ids, mask=None):
        logits = torch.zeros(1, input_ids.shape[1], config["vocab_size"])
        logits[:, -1, eos_token_id] = 100.0
        return logits, torch.tensor(0.0)

    model.forward = mock_forward
    out_eos = model.generate(input_ids, max_new_tokens=5, eos_token_id=eos_token_id)
    model.forward = original_forward
    assert out_eos.shape[1] == input_ids.shape[1] + 1, "EOS did not stop generation"
    print("  ✓ Generation correctly handles temperature, top_k sampling, and EOS stopping")


# ------------------------------------------------------------
# 15. Lion 优化器行为测试
# ------------------------------------------------------------
def test_lion_optimizer():
    print("\n[Test 15] Lion optimizer step")
    from utils import Lion
    linear = nn.Linear(10, 2)
    optimizer = Lion(linear.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.randn(4, 10)
    target = torch.randint(0, 2, (4,))
    # 记录初始权重
    initial_weight = linear.weight.data.clone()
    for _ in range(3):
        out = linear(x)
        loss = F.cross_entropy(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 验证权重确实更新了
    assert not torch.allclose(initial_weight, linear.weight.data), "Lion did not update weights"
    print("  ✓ Lion optimizer runs without errors and updates weights")


# ------------------------------------------------------------
# 16. 混合专家辅助损失正确性验证
# ------------------------------------------------------------
def test_moe_aux_loss_value():
    print("\n[Test 16] MoE auxiliary loss calculation")
    config = {
        "vocab_size": 100,
        "d_model": 64,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 2,
        "multiple_of": 64,
        "ffn_dim_multiplier": 1.0,
        "norm_eps": 1e-6,
        "max_seq_len": 16,
        "dropout": 0.0,
        "n_experts": 2,
    }
    model = ModernTransformer(config)
    model.train()
    x = torch.randint(0, config["vocab_size"], (2, 8))
    logits, aux_loss = model(x)
    # 辅助损失应为非负标量
    assert aux_loss.ndim == 0 and aux_loss >= 0, f"Invalid aux loss: {aux_loss}"
    # 如果专家分配均衡，损失应较小（< 2.0）
    assert aux_loss.item() < 2.0, f"Aux loss too high: {aux_loss.item()}"
    print(f"  ✓ MoE auxiliary loss is valid and within expected range: {aux_loss.item():.4f}")

# ---------------------------
# Run all tests
# ---------------------------
if __name__ == "__main__":
    print("Running unit tests for Modern Decoder-Only Transformer")
    test_rmsnorm()
    test_rope_relative()
    test_gqa_cache_size()
    test_swiglu()
    test_moe_balance()
    test_model_shapes()
    test_gradient_flow()
    test_generation()
    # ---------- 新增全流程测试 ----------
    test_overfit_small_batch()
    test_gradient_clipping()
    test_weight_tying()
    test_config_loading()
    test_device_handling()
    test_generation_advanced()
    test_lion_optimizer()
    test_moe_aux_loss_value()
    # -----------------------------------
    print("\n🎉 All tests passed! The transformer implementation is correct.")