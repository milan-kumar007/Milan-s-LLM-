import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class GPTConfig:
    """Configuration class for GPT model architecture."""
    block_size: int = 1024          # Maximum sequence context length
    vocab_size: int = 50257         # GPT-2 vocabulary size
    n_layer: int = 12               # Number of transformer layers
    n_head: int = 12                # Number of attention heads
    n_embd: int = 768               # Hidden embedding dimension (768 / 12 = 64 per head)
    dropout: float = 0.1            # Dropout probability
    bias: bool = True               # Use bias in Linear and LayerNorm layers
    gradient_checkpointing: bool = False  # Enable memory-efficient gradient checkpointing
    tie_weights: bool = True        # Tie word embeddings and output projection weights


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism.
    Uses PyTorch's native scaled_dot_product_attention (FlashAttention / SDPA)
    for high memory efficiency and speed on NVIDIA GPUs.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout

        # Key, query, value projections for all heads packed in a single linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal multi-head self-attention using PyTorch 2.0+ FlashAttention/SDPA
        dropout_p = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True
        )

        # Re-assemble all head outputs side by side: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-Forward Network (MLP) with GELU non-linearity (4x expansion)."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer decoder block with Pre-LayerNorm architecture."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT-style autoregressive Transformer Language Model.
    Supports mixed-precision training, gradient checkpointing, and weight tying.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between token embedding and final projection layer (GPT-2 standard)
        if config.tie_weights:
            self.lm_head.weight = self.transformer.wte.weight

        # Initialize weights following standard GPT normal distribution
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections (as per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for next-token prediction.
        Args:
            input_ids: Tensor of shape (B, T)
            labels: Optional target tensor of shape (B, T)
        Returns:
            logits: Tensor of shape (B, T, vocab_size)
            loss: Cross-entropy scalar loss if labels are provided, else None
        """
        device = input_ids.device
        B, T = input_ids.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block_size is {self.config.block_size}"

        # Position indices: [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Token + positional embeddings
        tok_emb = self.transformer.wte(input_ids)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)        # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks (with optional gradient checkpointing)
        for block in self.transformer.h:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        # Compute logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute next-token prediction loss if labels provided
        loss = None
        if labels is not None:
            # Flatten to compute CrossEntropy across all batch and sequence positions
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        learning_rate: float = 6e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cuda"
    ) -> torch.optim.Optimizer:
        """
        Configures AdamW with weight decay separation.
        Applies weight decay to 2D tensors (weights, embeddings)
        and exempts 1D tensors (biases, layernorms).
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"Optimizer Parameter Groups:")
        print(f"  Decayed tensors (weight decay={weight_decay}) : {len(decay_params)} ({num_decay:,} params)")
        print(f"  Non-decayed tensors (weight decay=0.0)      : {len(no_decay_params)} ({num_no_decay:,} params)")

        use_fused = (device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )
        print(f"  Optimizer: AdamW (fused={use_fused}, lr={learning_rate}, betas={betas})")
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_k: int | None = 40,
        top_p: float | None = None
    ) -> torch.Tensor:
        """
        Autoregressive text generation using temperature, top-k, and top-p filtering.
        Defaults to deterministic greedy decoding (temperature <= 0.05 or top_k == 1)
        for reliable, zero-hallucination outputs.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            # Take logits for the last token position
            next_logits = logits[:, -1, :]

            # Greedy decoding (deterministic - eliminates random hallucinations)
            if temperature <= 0.05 or (top_k is not None and top_k == 1):
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / max(1e-4, temperature)

                # Top-k filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("Inf")

                # Top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_mask = cumulative_probs > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                    next_logits[mask] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)

        return input_ids

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return total parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            if not self.config.tie_weights:
                n_params -= self.transformer.wte.weight.numel()
        return n_params

    def print_model_summary(self):
        """Prints a detailed parameter count breakdown."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        wte_params = self.transformer.wte.weight.numel()
        wpe_params = self.transformer.wpe.weight.numel()
        block_params = sum(p.numel() for p in self.transformer.h[0].parameters())
        all_blocks_params = sum(p.numel() for p in self.transformer.h.parameters())
        ln_f_params = sum(p.numel() for p in self.transformer.ln_f.parameters())
        lm_head_params = self.lm_head.weight.numel() if not self.config.tie_weights else 0

        print("=" * 65)
        print(" GPT MODEL ARCHITECTURE SUMMARY (100-200M Class)")
        print("=" * 65)
        print(f"  Configuration:")
        print(f"    Layers (n_layer)             : {self.config.n_layer}")
        print(f"    Hidden Dimension (n_embd)    : {self.config.n_embd}")
        print(f"    Attention Heads (n_head)     : {self.config.n_head} (head_dim: {self.config.n_embd // self.config.n_head})")
        print(f"    Context Window (block_size)  : {self.config.block_size}")
        print(f"    Vocabulary Size              : {self.config.vocab_size:,}")
        print(f"    Weight Tying (wte == lm_head): {self.config.tie_weights}")
        print(f"    Gradient Checkpointing       : {self.config.gradient_checkpointing}")
        print("-" * 65)
        print(f"  Parameter Breakdown:")
        print(f"    Token Embeddings (wte)       : {wte_params:>12,} ({wte_params / 1e6:.2f}M)")
        print(f"    Position Embeddings (wpe)    : {wpe_params:>12,} ({wpe_params / 1e6:.2f}M)")
        print(f"    Single Layer Block           : {block_params:>12,} ({block_params / 1e6:.2f}M)")
        print(f"    All {self.config.n_layer} Transformer Blocks : {all_blocks_params:>12,} ({all_blocks_params / 1e6:.2f}M)")
        print(f"    Final LayerNorm (ln_f)       : {ln_f_params:>12,} ({ln_f_params / 1e6:.4f}M)")
        if not self.config.tie_weights:
            print(f"    LM Head (Untied)             : {lm_head_params:>12,} ({lm_head_params / 1e6:.2f}M)")
        else:
            print(f"    LM Head (Tied to wte)        : [Shared with wte - 0 extra params]")
        print("-" * 65)
        print(f"  TOTAL PARAMETERS               : {total_params:>12,} ({total_params / 1e6:.2f}M)")
        print(f"  TRAINABLE PARAMETERS           : {trainable_params:>12,} ({trainable_params / 1e6:.2f}M)")
        print("=" * 65)

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2", override_args: dict | None = None):
        """
        Loads pre-trained GPT-2 weights from HuggingFace directly into our custom GPT architecture.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print(f"Loading pre-trained GPT weights ({model_type})...")
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        config_args["tie_weights"] = True
        config_args["gradient_checkpointing"] = True
        if override_args:
            config_args.update(override_args)

        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]
        sd_hf_keys = [k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")]

        # HuggingFace weights for Conv1D are transposed compared to PyTorch nn.Linear
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        for k in sd_hf_keys:
            if k == "transformer.wpe.weight":
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k][:config.block_size])
            elif any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        print(f"Loaded {len(sd_hf_keys)} weight tensors into custom {model_type} architecture!")
        return model


# Class alias for Milan's LLM
MilansLLM = GPT
