import os
import sys
import argparse
from typing import Generator
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.gpt import GPT, GPTConfig

def load_model(
    checkpoint_path: str = "models/checkpoints/best_model.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple[GPT, AutoTokenizer, dict]:
    """
    Loads a trained GPT model and tokenizer from a checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config")
    if config is None:
        # Default fallback config if not saved inside checkpoint
        config = GPTConfig(
            n_layer=12,
            n_embd=768,
            n_head=12,
            block_size=256,
            vocab_size=50257,
            gradient_checkpointing=False,
            tie_weights=True
        )

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    metadata = {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "train_loss": checkpoint.get("train_loss", None),
        "val_loss": checkpoint.get("val_loss", None),
        "device": device
    }

    return model, tokenizer, metadata


def sample_next_token(
    logits: torch.Tensor,
    context_tokens: list[int],
    temperature: float = 0.2,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    prompt_len: int = 0
) -> int:
    """
    Applies repetition penalty, temperature, top-k, and top-p (nucleus) filtering.
    Defaults to deterministic greedy decoding when temperature <= 0.05 or top_k == 1.
    """
    logits = logits.clone()

    # 1. Repetition penalty (applied strictly to generated tokens, NEVER prompt words)
    if repetition_penalty != 1.0 and len(context_tokens) > prompt_len:
        gen_tokens = context_tokens[prompt_len:]
        recent_window = gen_tokens[-64:] if len(gen_tokens) > 64 else gen_tokens
        for token_id in set(recent_window):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

    # 2. Pure deterministic greedy decoding (zero-hallucination argmax)
    if temperature <= 0.05 or top_k == 1:
        return torch.argmax(logits, dim=-1).item()

    # 3. Temperature scaling
    temperature = max(1e-4, temperature)
    logits = logits / temperature

    # 4. Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        val, _ = torch.topk(logits, top_k)
        logits[logits < val[:, [-1]]] = -float("Inf")

    # 5. Top-P (Nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep the first token exceeding threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("Inf")

    # 6. Softmax & Categorical Sampling
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


@torch.no_grad()
def generate_stream(
    model: GPT,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.2,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Generator[str, None, None]:
    """
    Generator function that yields individual text tokens in real time.
    """
    model.eval()
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    if not encoded:
        encoded = [tokenizer.eos_token_id]

    tokens = list(encoded)
    prompt_len = len(tokens)
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

    eos_token_id = tokenizer.eos_token_id
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for _ in range(max_new_tokens):
        # Crop context window if sequence exceeds model block_size
        if input_tensor.size(1) > model.config.block_size:
            idx_cond = input_tensor[:, -model.config.block_size:]
        else:
            idx_cond = input_tensor

        with torch.amp.autocast(device_type=device, dtype=amp_dtype):
            logits, _ = model(idx_cond)

        # Get logits for the last token position: (1, vocab_size)
        next_logits = logits[:, -1, :]

        next_token_id = sample_next_token(
            next_logits,
            context_tokens=tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            prompt_len=prompt_len
        )

        # Stop if end-of-text token is produced
        if next_token_id == eos_token_id:
            break

        tokens.append(next_token_id)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

        # Decode newly generated token
        token_str = tokenizer.decode([next_token_id], errors="replace")
        yield token_str


def generate_text(
    model: GPT,
    tokenizer: AutoTokenizer,
    prompt: str,
    **kwargs
) -> str:
    """Convenience function returning full generated text as a string."""
    tokens = list(generate_stream(model, tokenizer, prompt, **kwargs))
    return prompt + "".join(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with custom trained GPT model")
    parser.add_argument("--prompt", type=str, default="Write an email to my class teacher asking for leave.", help="Initial prompt")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/sft_best_model.pt", help="Checkpoint path")
    parser.add_argument("--instruct", action="store_true", default=True, help="Format prompt as instruction")
    parser.add_argument("--no-instruct", dest="instruct", action="store_false", help="Do not format prompt as instruction")
    parser.add_argument("--greedy", action="store_true", help="Use deterministic greedy decoding (zero hallucination)")
    parser.add_argument("--max_tokens", type=int, default=120, help="Maximum new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature (0.0 to 2.0, where 0.0 is greedy)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling cutoff")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling threshold")
    parser.add_argument("--rep_penalty", type=float, default=1.05, help="Repetition penalty factor")

    args = parser.parse_args()

    # If greedy requested, override temperature to 0.0
    effective_temp = 0.0 if args.greedy else args.temp
    effective_top_k = 1 if args.greedy else args.top_k

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl, tok, meta = load_model(args.checkpoint, device=dev)

    if args.instruct:
        prompt_text = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
    else:
        prompt_text = args.prompt

    print("\n" + "=" * 60)
    print(f"Task: \"{args.prompt}\"")
    print("=" * 60)
    print("Generated Answer:\n")

    for chunk in generate_stream(
        mdl, tok, prompt_text,
        max_new_tokens=args.max_tokens,
        temperature=effective_temp,
        top_k=effective_top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
        device=dev
    ):
        if "###" in chunk:
            break
        print(chunk, end="", flush=True)

    print("\n" + "=" * 60)
