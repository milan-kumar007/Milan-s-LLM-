import os
import json
import torch
from transformers import AutoTokenizer

from download_wikipedia import download_wikipedia_corpus
from dataset import create_dataloaders

def run_preprocessing_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    block_size: int = 256,
    batch_size: int = 8,
    train_ratio: float = 0.8
):
    print("=" * 70)
    print(" LANGUAGE MODEL DATA PREPROCESSING PIPELINE (WIKIPEDIA / GPT-2 BPE)")
    print("=" * 70)

    # 1. Download / Load Text Corpus from raw_dir
    corpus_files = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.endswith(".txt") and os.path.isfile(os.path.join(raw_dir, f))
    ]
    if not corpus_files:
        corpus_files = download_wikipedia_corpus(output_dir=raw_dir)
    print(f"\n[Step 1] Loaded {len(corpus_files)} modern corpus files from '{raw_dir}'")

    all_texts = []
    total_raw_chars = 0
    for path in corpus_files:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            all_texts.append(content)
            total_raw_chars += len(content)
            print(f"  - {os.path.basename(path):25}: {len(content):,} characters")

    print(f"Total raw text size: {total_raw_chars:,} characters (~{total_raw_chars / 1024 / 1024:.2f} MB)")

    # 2. Initialize GPT-2 Tokenizer
    print("\n[Step 2] Initializing GPT-2 Byte-Level BPE Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer Vocab Size: {tokenizer.vocab_size:,}")
    print(f"  EOS Token           : '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  PAD Token           : '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # 3. Tokenize Text with Document Boundary Separation
    print("\n[Step 3] Tokenizing text documents...")
    all_token_ids = []
    eos_id = tokenizer.eos_token_id

    for idx, text in enumerate(all_texts):
        # Tokenize document
        doc_tokens = tokenizer.encode(text, add_special_tokens=False)
        all_token_ids.extend(doc_tokens)
        all_token_ids.append(eos_id)  # Separate documents with <|endoftext|>

    total_tokens = len(all_token_ids)
    print(f"  Total generated tokens: {total_tokens:,}")
    print(f"  Compression ratio     : {total_raw_chars / total_tokens:.2f} chars/token")

    # Convert to PyTorch Tensor
    token_tensor = torch.tensor(all_token_ids, dtype=torch.long)

    # 4. Save Processed Cache
    os.makedirs(processed_dir, exist_ok=True)
    cache_file = os.path.join(processed_dir, "tokens.pt")
    meta_file = os.path.join(processed_dir, "metadata.json")

    torch.save(token_tensor, cache_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_tokens": total_tokens,
            "vocab_size": tokenizer.vocab_size,
            "eos_token_id": tokenizer.eos_token_id,
            "block_size": block_size,
            "files": [os.path.basename(p) for p in corpus_files]
        }, f, indent=2)

    print(f"\n[Step 4] Saved token cache to '{cache_file}' ({os.path.getsize(cache_file) / 1024 / 1024:.2f} MB)")

    # 5. Build 80/20 Train/Validation DataLoaders
    print("\n[Step 5] Building PyTorch DataLoaders (80% Train / 20% Val)...")
    train_loader, val_loader, stats = create_dataloaders(
        token_tensor,
        train_ratio=train_ratio,
        block_size=block_size,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

    print(f"  Split Ratio        : {int(train_ratio * 100)}% Train / {int((1 - train_ratio) * 100)}% Validation")
    print(f"  Train Tokens       : {stats['train_tokens']:,}")
    print(f"  Validation Tokens  : {stats['val_tokens']:,}")
    print(f"  Train Sequences    : {stats['train_samples']:,} blocks (Block Size: {block_size})")
    print(f"  Val Sequences      : {stats['val_samples']:,} blocks")
    print(f"  Train Batches      : {stats['train_batches']:,} (Batch Size: {batch_size})")
    print(f"  Val Batches        : {stats['val_batches']:,}")

    # 6. Verification and Sample Inspection
    print("\n" + "=" * 70)
    print(" VERIFICATION & SAMPLE TOKENIZED OUTPUT")
    print("=" * 70)

    # Fetch one batch from train_loader
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print(f"Batch Tensor Shapes:")
    print(f"  input_ids : {tuple(input_ids.shape)}  (dtype: {input_ids.dtype})")
    print(f"  labels    : {tuple(labels.shape)}     (dtype: {labels.dtype})")

    # CUDA device transfer verification
    if torch.cuda.is_available():
        gpu_input = input_ids.to("cuda")
        gpu_labels = labels.to("cuda")
        print(f"  CUDA Device Transfer : SUCCESS -> {gpu_input.device}")

    # Inspect first sequence in batch
    sample_seq = input_ids[0]
    sample_target = labels[0]

    print("\nFirst 15 Numerical Token IDs from Sequence 0:")
    print(f"  Input Tokens  (x): {sample_seq[:15].tolist()}")
    print(f"  Target Tokens (y): {sample_target[:15].tolist()}")

    # Decoded text sample
    decoded_sample = tokenizer.decode(sample_seq[:50])
    print("\nDecoded Sample (first 50 tokens):")
    print("-" * 50)
    print(f'"{decoded_sample}..."')
    print("-" * 50)

    # Autoregressive next-token alignment demonstration
    print("\nNext-Token Prediction Alignment Check:")
    print(f"  {'Step':<5} | {'Input Token':<18} | {'Token ID':<10} | {'Target Next Token':<18} | {'Target ID':<10}")
    print("  " + "-" * 68)
    for i in range(8):
        inp_id = sample_seq[i].item()
        tgt_id = sample_target[i].item()
        inp_str = repr(tokenizer.decode([inp_id]))
        tgt_str = repr(tokenizer.decode([tgt_id]))
        print(f"  {i:<5} | {inp_str:<18} | {inp_id:<10} | {tgt_str:<18} | {tgt_id:<10}")

    # Verify shift property: target[i] must equal input[i+1]
    is_shifted = torch.equal(sample_seq[1:], sample_target[:-1])
    print("\nShift Consistency Check (Target[i] == Input[i+1]):", "PASSED" if is_shifted else "FAILED")
    print("=" * 70)

    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    run_preprocessing_pipeline()
