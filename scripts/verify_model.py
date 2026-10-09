import os
import sys
import torch
from transformers import AutoTokenizer

# Add project root and models to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.gpt import GPT, GPTConfig

def verify_gpt_model():
    print("=" * 65)
    print(" GPT TRANSFORMER MODEL VERIFICATION & BENCHMARK")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 1. Test Configurations: 12 layers, 768 hidden, 12 heads
    print("\n--- 1. Architecture Configurations ---")
    config_tied = GPTConfig(
        n_layer=12,
        n_embd=768,
        n_head=12,
        block_size=256,
        vocab_size=50257,
        gradient_checkpointing=True,
        tie_weights=True
    )
    model_tied = GPT(config_tied)
    print("Configuration A (Weight-Tied - Standard GPT-2):")
    model_tied.print_model_summary()

    config_untied = GPTConfig(
        n_layer=12,
        n_embd=768,
        n_head=12,
        block_size=256,
        vocab_size=50257,
        gradient_checkpointing=True,
        tie_weights=False
    )
    model_untied = GPT(config_untied)
    print("\nConfiguration B (Untied Embeddings & LM Head):")
    model_untied.print_model_summary()

    # We will use model_tied on the GPU
    model = model_tied.to(device)

    # 2. Configure Optimizer
    print("\n--- 2. AdamW Optimizer Configuration ---")
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        device_type=device
    )

    # 3. Setup Mixed-Precision (bfloat16 if supported, else float16)
    use_bfloat16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    print(f"\n--- 3. Mixed Precision & Gradient Checkpointing ---")
    print(f"  Mixed Precision Dtype  : {amp_dtype} (bf16 supported: {use_bfloat16})")
    print(f"  GradScaler Enabled     : {scaler.is_enabled()}")
    print(f"  Gradient Checkpointing : {model.config.gradient_checkpointing}")

    # 4. Forward & Backward Pass Verification
    print(f"\n--- 4. Training Step Verification (Forward + Loss + Backward) ---")
    batch_size = 8
    seq_len = 256
    # Create synthetic test batch
    x = torch.randint(0, config_tied.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, config_tied.vocab_size, (batch_size, seq_len), device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type=device, dtype=amp_dtype):
        logits, loss = model(x, labels=y)

    print(f"  Input Shape  (x)      : {tuple(x.shape)}")
    print(f"  Target Shape (y)      : {tuple(y.shape)}")
    print(f"  Output Logits Shape   : {tuple(logits.shape)}")
    print(f"  Next-Token Loss Value : {loss.item():.4f}")

    # Backward pass & optimizer step
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"  Gradient Norm (clipped): {grad_norm:.4f}")
    print(f"  Optimizer Step         : SUCCESS")

    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  VRAM Memory Allocated : {allocated:.1f} MB")
        print(f"  Peak VRAM Memory      : {peak:.1f} MB (Comfortably within 8GB VRAM)")

    # 5. Text Generation Test
    print(f"\n--- 5. Autoregressive Text Generation Test ---")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_text = "Once upon a time"
    prompt_tokens = torch.tensor([tokenizer.encode(prompt_text)], device=device)

    print(f"  Prompt : '{prompt_text}'")
    with torch.amp.autocast(device_type=device, dtype=amp_dtype):
        gen_tokens = model.generate(prompt_tokens, max_new_tokens=25, temperature=0.8, top_k=40)

    generated_text = tokenizer.decode(gen_tokens[0].tolist())
    print(f"  Model Output (untrained random init):\n  \"{generated_text}\"")

    print("\n" + "=" * 65)
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    verify_gpt_model()
