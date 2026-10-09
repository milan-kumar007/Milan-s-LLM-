import os
import json
import urllib.request
import torch
from transformers import AutoTokenizer

CUSTOM_TARGETED_INSTRUCTIONS = [
    {
        "instruction": "Write an email to my class teacher asking for leave.",
        "input": "",
        "output": (
            "Subject: Leave Application - [Your Name]\n\n"
            "Respected Teacher,\n\n"
            "I am writing to inform you that I will be unable to attend classes for two days due to an urgent family matter. "
            "I request you to kindly grant me leave of absence for these days.\n\n"
            "I will ensure that I collect all the notes and complete the pending assignments as soon as I return.\n\n"
            "Thank you for your understanding.\n\n"
            "Sincerely,\n"
            "[Your Name]\n"
            "Roll No: [Your Roll Number]"
        )
    },
    {
        "instruction": "Write a formal leave application to my teacher for fever and sickness.",
        "input": "",
        "output": (
            "Subject: Application for Sick Leave\n\n"
            "Dear Teacher,\n\n"
            "I am writing to request leave from school for two days as I have been suffering from fever and the doctor has advised bed rest. "
            "Please excuse my absence. I will catch up with all missed classwork upon my return.\n\n"
            "Thank you.\n\n"
            "Yours obediently,\n"
            "[Your Name]"
        )
    },
    {
        "instruction": "What does a GPU do?",
        "input": "",
        "output": (
            "A GPU (Graphics Processing Unit) is a specialized processor designed to perform thousands of mathematical calculations simultaneously. "
            "Unlike a CPU which has a few fast cores for general tasks, a GPU has thousands of smaller cores. "
            "This makes it extremely efficient for rendering 3D graphics, processing videos, and training artificial intelligence and deep learning models."
        )
    },
    {
        "instruction": "What is the difference between a CPU and a GPU?",
        "input": "",
        "output": (
            "The main difference is their architecture and purpose:\n"
            "1. CPU (Central Processing Unit): Optimized for sequential processing and complex logic with 4 to 16 powerful cores. It acts as the brain of the computer.\n"
            "2. GPU (Graphics Processing Unit): Optimized for mass parallel computing with thousands of cores designed to execute simple operations simultaneously on large datasets like neural networks and graphics."
        )
    },
    {
        "instruction": "Explain what an algorithm is in computer science.",
        "input": "",
        "output": (
            "An algorithm is a step-by-step procedure or set of rules defined to solve a specific problem or perform a computation. "
            "In software, algorithms take input data, process it through defined logic, and produce the desired output efficiently."
        )
    },
    {
        "instruction": "What is machine learning?",
        "input": "",
        "output": (
            "Machine learning is a subfield of artificial intelligence where computer systems learn from data to identify patterns and make decisions without being explicitly programmed for every scenario."
        )
    },
    {
        "instruction": "Write a Python function to check if a number is prime.",
        "input": "",
        "output": (
            "def is_prime(n):\n"
            "    if n <= 1:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True"
        )
    }
]

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{output}"

def format_prompt(instruction: str, output: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip(), output=output.strip())

def prepare_sft_dataset(
    output_dir: str = "data/processed",
    max_alpaca_samples: int = 2500,
    max_seq_len: int = 256
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print(" PREPARING INSTRUCTION FINE-TUNING (SFT) DATASET")
    print("=" * 65)

    # 1. Download Stanford Alpaca Dataset
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    print("Downloading Stanford Alpaca instruction dataset...")
    headers = {"User-Agent": "CustomLLMSFT/1.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw_alpaca = json.loads(resp.read().decode("utf-8"))

    print(f"Downloaded {len(raw_alpaca):,} raw Alpaca instructions.")

    # Filter concise, high-quality instructions
    filtered_alpaca = []
    for item in raw_alpaca:
        # Prefer direct instructions with reasonably sized answers
        if len(item["instruction"]) < 200 and len(item["output"]) < 400 and len(item["output"]) > 20:
            filtered_alpaca.append(item)
            if len(filtered_alpaca) >= max_alpaca_samples:
                break

    print(f"Selected {len(filtered_alpaca):,} diverse Alpaca instructions.")

    # 2. Combine with Targeted Leave Email & Tech Q&A Instructions (multiplied for weight)
    # Give higher frequency to targeted leave emails and CPU/GPU questions
    all_examples = []
    for _ in range(25):  # Boost targeted Q&A frequency
        all_examples.extend(CUSTOM_TARGETED_INSTRUCTIONS)

    for item in filtered_alpaca:
        inst = item["instruction"]
        if item.get("input"):
            inst += f"\nInput: {item['input']}"
        all_examples.append({
            "instruction": inst,
            "output": item["output"]
        })

    print(f"Total training examples: {len(all_examples):,}")

    # 3. Tokenize with Masking
    print("\nTokenizing with GPT-2 Byte-Level BPE and Prompt Masking...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos_token_id = tokenizer.eos_token_id

    input_ids_list = []
    labels_list = []

    for item in all_examples:
        prompt_text = f"### Instruction:\n{item['instruction'].strip()}\n\n### Response:\n"
        full_text = prompt_text + item["output"].strip() + "<|endoftext|>"

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) > (max_seq_len + 1):
            full_ids = full_ids[:max_seq_len + 1]

        inputs = full_ids[:-1]
        targets = list(full_ids[1:])
        prompt_len = len(prompt_ids)

        # Mask prompt tokens in targets (prompt_len - 1 tokens)
        for i in range(min(prompt_len - 1, len(targets))):
            targets[i] = -100

        # Pad to max_seq_len
        pad_len = max_seq_len - len(inputs)
        if pad_len > 0:
            inputs = inputs + [eos_token_id] * pad_len
            targets = targets + [-100] * pad_len

        input_ids_list.append(inputs)
        labels_list.append(targets)

    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    # 80/20 train/val split
    total = len(input_ids_tensor)
    split = int(0.85 * total)

    train_data = {
        "input_ids": input_ids_tensor[:split],
        "labels": labels_tensor[:split]
    }
    val_data = {
        "input_ids": input_ids_tensor[split:],
        "labels": labels_tensor[split:]
    }

    train_path = os.path.join(output_dir, "sft_train.pt")
    val_path = os.path.join(output_dir, "sft_val.pt")

    torch.save(train_data, train_path)
    torch.save(val_data, val_path)

    print(f"Saved SFT Train Dataset : {train_path} ({len(train_data['input_ids']):,} samples)")
    print(f"Saved SFT Val Dataset   : {val_path} ({len(val_data['input_ids']):,} samples)")
    print("=" * 65)

if __name__ == "__main__":
    prepare_sft_dataset()
