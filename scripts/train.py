import os
import sys
import math
import time
import json
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for plotting
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

# Ensure imports work from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.gpt import GPT, GPTConfig
from scripts.dataset import create_dataloaders


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate_loss(model: GPT, val_loader, device: str, amp_dtype, max_eval_batches: int = 30) -> float:
    """Calculates cross-entropy loss over validation batches."""
    model.eval()
    total_loss = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if i >= max_eval_batches:
            break
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device, dtype=amp_dtype):
            _, loss = model(x, labels=y)

        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(1, count)


def plot_loss_curves(history: dict, output_path: str = "models/loss_curve.png"):
    """Visualizes training and validation loss curves and learning rate schedule."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150)

    # Subplot 1: Train vs Validation Loss
    steps = history["steps"]
    train_losses = history["train_losses"]
    val_steps = history["val_steps"]
    val_losses = history["val_losses"]

    ax1.plot(steps, train_losses, label="Train Loss", color="#1f77b4", alpha=0.7, linewidth=1.5)
    
    # Calculate exponential moving average for smoother training curve
    if len(train_losses) > 5:
        smoothed = []
        ema = train_losses[0]
        alpha = 0.1
        for val in train_losses:
            ema = alpha * val + (1 - alpha) * ema
            smoothed.append(ema)
        ax1.plot(steps, smoothed, label="Train Loss (EMA Smoothed)", color="#003366", linewidth=2.0)

    if val_losses:
        ax1.plot(val_steps, val_losses, label="Validation Loss", color="#e74c3c", marker="o", markersize=5, linewidth=2.0)

    ax1.set_title("Language Model Training & Validation Loss", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel("Optimization Step", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10, loc="upper right")

    # Subplot 2: Learning Rate Schedule
    lrs = history["learning_rates"]
    ax2.plot(steps, lrs, label="Learning Rate (Cosine Schedule)", color="#2ca02c", linewidth=2.0)
    ax2.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Optimization Step", fontsize=11)
    ax2.set_ylabel("Learning Rate", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(0,0))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"\n[Chart Saved] Training loss curve saved to: {output_path}")


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    step: int,
    epoch: int,
    loss: float,
    val_loss: float,
    path: str
):
    """Saves complete model checkpoint with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "train_loss": loss,
        "val_loss": val_loss,
    }, path)


def train_language_model(
    epochs: int = 3,
    batch_size: int = 8,
    block_size: int = 256,
    grad_accum_steps: int = 2,
    max_lr: float = 6e-4,
    min_lr: float = 6e-5,
    warmup_steps: int = 20,
    eval_interval: int = 40,
    save_interval: int = 100,
    tokens_path: str = "data/processed/tokens.pt",
    checkpoint_dir: str = "models/checkpoints",
    plot_path: str = "models/loss_curve.png",
    metrics_path: str = "models/training_metrics.json"
):
    print("=" * 70)
    print(" GPT LANGUAGE MODEL TRAINING LOOP")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device : {device.upper()}")
    if device == "cuda":
        print(f"GPU Model      : {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 1. Load Preprocessed Corpus
    if not os.path.exists(tokens_path):
        print(f"Tokens cache '{tokens_path}' not found. Generating...")
        from scripts.preprocess_data import run_preprocessing_pipeline
        run_preprocessing_pipeline()

    tokens = torch.load(tokens_path, weights_only=True)
    train_loader, val_loader, stats = create_dataloaders(
        tokens,
        train_ratio=0.8,
        block_size=block_size,
        batch_size=batch_size,
        pin_memory=(device == "cuda")
    )

    batches_per_epoch = len(train_loader)
    steps_per_epoch = batches_per_epoch // grad_accum_steps
    total_steps = steps_per_epoch * epochs

    print(f"Dataset Statistics:")
    print(f"  Total Tokens       : {stats['total_tokens']:,}")
    print(f"  Train Sequences    : {stats['train_samples']:,} (Block Size: {block_size})")
    print(f"  Val Sequences      : {stats['val_samples']:,}")
    print(f"  Batches Per Epoch  : {batches_per_epoch} (Batch Size: {batch_size})")
    print(f"  Total Epochs       : {epochs}")
    print(f"  Gradient Accum.    : {grad_accum_steps} (Effective Batch: {batch_size * grad_accum_steps})")
    print(f"  Total Opt Steps    : {total_steps}")

    # 2. Initialize Model Architecture (12 layers, 768 hidden, 12 heads)
    config = GPTConfig(
        n_layer=12,
        n_embd=768,
        n_head=12,
        block_size=block_size,
        vocab_size=50257,
        dropout=0.1,
        gradient_checkpointing=True,
        tie_weights=True
    )
    model = GPT(config).to(device)
    model.print_model_summary()

    # 3. Setup Optimizer & Mixed Precision
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_lr,
        betas=(0.9, 0.95),
        device_type=device
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    print(f"Mixed Precision : {amp_dtype} (bf16: {use_bf16})")

    # 4. Tokenizer for generation samples
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Metrics history
    history = {
        "steps": [],
        "train_losses": [],
        "val_steps": [],
        "val_losses": [],
        "learning_rates": []
    }

    # Initial Validation Baseline
    print("\n[Baseline] Calculating initial validation loss...")
    initial_val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
    print(f"  Initial Val Loss: {initial_val_loss:.4f} (Perplexity: {math.exp(min(initial_val_loss, 20)):.2f})")
    history["val_steps"].append(0)
    history["val_losses"].append(initial_val_loss)

    best_val_loss = initial_val_loss
    global_step = 0
    start_time = time.time()
    total_processed_tokens = 0

    print("\n" + "=" * 70)
    print(f" TRAINING PROGRESS ({epochs} Epochs | {total_steps} Steps)")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_tokens = 0
        running_loss = 0.0

        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}/{epochs}", unit="step", dynamic_ncols=True)

        train_iter = iter(train_loader)
        for _ in range(steps_per_epoch):
            global_step += 1

            # Update Learning Rate
            lr = get_lr(global_step, warmup_steps, total_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            # Gradient accumulation micro-steps
            for micro in range(grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                x = batch["input_ids"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device, dtype=amp_dtype):
                    _, loss = model(x, labels=y)
                    loss = loss / grad_accum_steps

                accum_loss += loss.item()

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_tokens += x.numel()
                total_processed_tokens += x.numel()

            # Optimizer Step & Gradient Clipping
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss = 0.9 * running_loss + 0.1 * accum_loss if running_loss > 0 else accum_loss

            # Record metrics
            history["steps"].append(global_step)
            history["train_losses"].append(accum_loss)
            history["learning_rates"].append(lr)

            # Measure Speed & VRAM
            elapsed = time.time() - start_time
            tokens_per_sec = total_processed_tokens / max(1.0, elapsed)
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0

            pbar.set_postfix({
                "loss": f"{accum_loss:.3f}",
                "ema": f"{running_loss:.3f}",
                "lr": f"{lr:.1e}",
                "tok/s": f"{tokens_per_sec:.0f}",
                "VRAM": f"{vram_mb:.0f}MB"
            })
            pbar.update(1)

            # Periodic Validation Check
            if global_step % eval_interval == 0:
                val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
                history["val_steps"].append(global_step)
                history["val_losses"].append(val_loss)

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = os.path.join(checkpoint_dir, "best_model.pt")
                    save_checkpoint(model, optimizer, config, global_step, epoch, accum_loss, val_loss, best_ckpt)
                    tqdm.write(f" >>> [Step {global_step}] *New Best Val Loss*: {val_loss:.4f} (Saved to {best_ckpt})")
                else:
                    tqdm.write(f" >>> [Step {global_step}] Val Loss: {val_loss:.4f}")

            # Periodic Checkpoint
            if global_step % save_interval == 0:
                ckpt_file = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
                val_loss_curr = history["val_losses"][-1] if history["val_losses"] else accum_loss
                save_checkpoint(model, optimizer, config, global_step, epoch, accum_loss, val_loss_curr, ckpt_file)
                tqdm.write(f" >>> [Step {global_step}] Periodic Checkpoint Saved: {ckpt_file}")

        pbar.close()

        # End of Epoch Validation
        epoch_val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
        if not history["val_steps"] or history["val_steps"][-1] != global_step:
            history["val_steps"].append(global_step)
            history["val_losses"].append(epoch_val_loss)

        epoch_time = time.time() - start_time
        print(f"\n--- End of Epoch {epoch}/{epochs} ---")
        print(f"  Train Loss (EMA) : {running_loss:.4f}")
        print(f"  Val Loss         : {epoch_val_loss:.4f} (Perplexity: {math.exp(min(epoch_val_loss, 20)):.2f})")

        # Save Epoch Checkpoint
        epoch_ckpt = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        save_checkpoint(model, optimizer, config, global_step, epoch, running_loss, epoch_val_loss, epoch_ckpt)
        print(f"  Saved Epoch Checkpoint: {epoch_ckpt}")

        # Qualitative Text Generation Sample
        print(f"\n  [Sample Generation @ Epoch {epoch}]")
        sample_prompt = "Artificial intelligence and machine learning"
        prompt_ids = torch.tensor([tokenizer.encode(sample_prompt)], device=device)
        with torch.amp.autocast(device_type=device, dtype=amp_dtype):
            gen_ids = model.generate(prompt_ids, max_new_tokens=35, temperature=0.75, top_k=40)
        generated_str = tokenizer.decode(gen_ids[0].tolist(), errors="replace")
        safe_str = generated_str.encode("ascii", errors="replace").decode("ascii")
        print(f"  Prompt : \"{sample_prompt}\"")
        print(f"  Output : \"{safe_str}\"")
        print("-" * 70)

    # 5. Finalize: Plot & Save Metrics
    total_time = time.time() - start_time
    print(f"\n[Training Complete in {total_time:.1f}s | {total_processed_tokens / total_time:.0f} avg tok/s]")

    # Plot Loss Curves
    plot_loss_curves(history, output_path=plot_path)

    # Save metrics JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Metrics Saved] Training metrics saved to: {metrics_path}")

    # Final summary
    print("\n" + "=" * 70)
    print(" TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Initial Val Loss : {initial_val_loss:.4f}")
    print(f"  Final Val Loss   : {history['val_losses'][-1]:.4f}")
    print(f"  Best Val Loss    : {best_val_loss:.4f}")
    print(f"  Saved Checkpoints in: {checkpoint_dir}")
    print(f"  Loss Curve Chart    : {plot_path}")
    print("=" * 70)

    return model, history


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    train_language_model(epochs=4)
