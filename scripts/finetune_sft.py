import os
import sys
import math
import time
import json
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.gpt import GPT, GPTConfig

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def evaluate_sft(model, val_loader, device, amp_dtype, max_eval_batches=20):
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_eval_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device, dtype=amp_dtype):
            _, loss = model(x, labels=y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(1, count)

def finetune_instructions(
    base_checkpoint: str = "models/checkpoints/best_model.pt",
    train_data_path: str = "data/processed/sft_train.pt",
    val_data_path: str = "data/processed/sft_val.pt",
    output_dir: str = "models/checkpoints",
    epochs: int = 3,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    max_lr: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 15
):
    print("=" * 65)
    print(" INSTRUCTION FINE-TUNING (SFT) FOR 124M GPT MODEL")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Data
    train_dict = torch.load(train_data_path, weights_only=True)
    val_dict = torch.load(val_data_path, weights_only=True)

    train_ds = TensorDataset(train_dict["input_ids"], train_dict["labels"])
    val_ds = TensorDataset(val_dict["input_ids"], val_dict["labels"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=(device=="cuda"))

    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = steps_per_epoch * epochs

    print(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")
    print(f"Epochs: {epochs} | Steps per epoch: {steps_per_epoch} | Total optimization steps: {total_steps}")

    # 2. Load Model with Pre-trained Foundation Weights
    print("Initializing model with pre-trained 124M weights for fluent language comprehension...")
    model = GPT.from_pretrained("gpt2", override_args=dict(block_size=256, gradient_checkpointing=True))
    config = model.config

    model.to(device)
    model.train()

    # 3. Setup Optimizer & Precision
    optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=max_lr, device_type=device)
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    best_val = evaluate_sft(model, val_loader, device, amp_dtype)
    print(f"\nInitial Instruction Val Loss: {best_val:.4f}")

    global_step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_iter = iter(train_loader)
        running_loss = 0.0

        pbar = tqdm(total=steps_per_epoch, desc=f"SFT Epoch {epoch}/{epochs}", unit="step", dynamic_ncols=True)

        for _ in range(steps_per_epoch):
            global_step += 1
            lr = get_lr(global_step, warmup_steps, total_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(grad_accum_steps):
                x, y = next(train_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                    _, loss = model(x, labels=y)
                    loss = loss / grad_accum_steps

                accum_loss += loss.item()
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running_loss = 0.9 * running_loss + 0.1 * accum_loss if running_loss > 0 else accum_loss
            pbar.set_postfix({"loss": f"{accum_loss:.3f}", "ema": f"{running_loss:.3f}", "lr": f"{lr:.1e}"})
            pbar.update(1)

        pbar.close()

        val_loss = evaluate_sft(model, val_loader, device, amp_dtype)
        print(f"End of Epoch {epoch}: Train Loss = {running_loss:.4f} | Val Loss = {val_loss:.4f}")

        # Save Best SFT Checkpoint
        sft_best_path = os.path.join(output_dir, "sft_best_model.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "step": global_step,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_loss": val_loss,
                "type": "instruction_tuned"
            }, sft_best_path)
            print(f" >>> [Saved New Best SFT Model to {sft_best_path}]")

        # Test Text Generation on Key Prompts
        print("\n  [Testing Instruction Answering]")
        test_prompts = [
            "Write an email to my class teacher asking for leave.",
            "What does a GPU do?"
        ]
        for tp in test_prompts:
            prompt_formatted = f"### Instruction:\n{tp}\n\n### Response:\n"
            prompt_tensor = torch.tensor([tokenizer.encode(prompt_formatted)], device=device)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                gen_ids = model.generate(prompt_tensor, max_new_tokens=60, temperature=0.7, top_k=40)
            answer_full = tokenizer.decode(gen_ids[0].tolist(), errors="replace")
            # Extract only response part
            response_only = answer_full.split("### Response:\n")[-1].split("<|endoftext|>")[0].strip()
            safe_resp = response_only.encode("ascii", errors="replace").decode("ascii")
            print(f"  Q: \"{tp}\"")
            print(f"  A: {safe_resp}\n")

    print("=" * 65)
    print(f"SFT Training Complete in {time.time() - start_time:.1f}s!")
    print(f"Instruction Model Saved: {os.path.join(output_dir, 'sft_best_model.pt')}")
    print("=" * 65)

if __name__ == "__main__":
    finetune_instructions(epochs=3)
