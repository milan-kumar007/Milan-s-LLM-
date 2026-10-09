import torch
from torch.utils.data import Dataset, DataLoader

class CausalLMDataset(Dataset):
    """
    PyTorch Dataset for Autoregressive / Causal Language Modeling.
    Slices a continuous stream of token IDs into fixed-length context windows of length `block_size`.
    
    Each item yields:
      - input_ids: Tokens t_0, t_1, ..., t_{L-1} of length block_size
      - labels   : Tokens t_1, t_2, ..., t_L (next-token prediction targets)
    """
    def __init__(self, token_ids: torch.Tensor, block_size: int = 256):
        super().__init__()
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        else:
            token_ids = token_ids.to(dtype=torch.long)

        self.block_size = block_size
        # Each chunk requires (block_size + 1) tokens so that y = x[1:] has length block_size
        self.chunk_size = block_size + 1
        self.num_chunks = len(token_ids) // self.chunk_size

        if self.num_chunks == 0:
            raise ValueError(f"Total tokens ({len(token_ids)}) is smaller than required chunk size ({self.chunk_size})")

        # Trim to exact multiple of chunk_size and reshape into [num_chunks, chunk_size]
        usable_len = self.num_chunks * self.chunk_size
        self.data = token_ids[:usable_len].view(self.num_chunks, self.chunk_size)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.data[idx]
        input_ids = chunk[:-1]  # shape: (block_size,)
        labels = chunk[1:]      # shape: (block_size,)
        return {
            "input_ids": input_ids,
            "labels": labels
        }


def create_dataloaders(
    token_ids: torch.Tensor,
    train_ratio: float = 0.8,
    block_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Splits token stream into 80/20 train/validation sets and returns PyTorch DataLoaders.
    """
    total_tokens = len(token_ids)
    split_idx = int(total_tokens * train_ratio)

    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    train_dataset = CausalLMDataset(train_tokens, block_size=block_size)
    val_dataset = CausalLMDataset(val_tokens, block_size=block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    stats = {
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "block_size": block_size,
        "batch_size": batch_size,
    }

    return train_loader, val_loader, stats
