# Lara

Lara (Lightweight Advanced Representation Architecture) is a PyTorch encoder-only Transformer for bidirectional text representations. It combines dynamic NTK-scaled RoPE, pre-norm RMSNorm, SwiGLU feed-forward layers, and local/global attention. PyTorch's scaled dot-product attention API dispatches to an optimized backend when the device and inputs support it.

## Install

Lara requires PyTorch 2.0 or newer, NumPy, and Safetensors. Install the PyTorch wheel for your hardware using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install Lara:

```bash
python -m pip install .
```

For CPU-only development, install the CPU wheel and Lara's development tools:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

## Quick Start

Lara accepts token IDs; it does not include a tokenizer. Use IDs from your BPE tokenizer, including its byte-fallback tokens if configured, and set `vocab_size` to match that tokenizer. In `attention_mask`, `1` marks a real token and `0` marks padding.

```python
import torch

from lara import LaraConfig, LaraModel

config = LaraConfig(
    vocab_size=32_000,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    intermediate_size=768,
    local_window_size=128,
    num_global_tokens=1,
)
model = LaraModel(config)

input_ids = torch.randint(1, config.vocab_size, (2, 512))
attention_mask = torch.ones_like(input_ids)
attention_mask[1, 400:] = 0
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print(outputs.last_hidden_state.shape)  # [batch, sequence, hidden]
print(outputs.pooler_output.shape)      # [batch, hidden]
```

For classification, use `LaraForSequenceClassification(config, num_labels=...)`; it returns `logits` and, when `labels` are supplied, cross-entropy `loss`. `LaraForRetrieval(config)` returns L2-normalized pooled vectors suitable for similarity scoring.

Save and reload the base encoder with `config.json` and `model.safetensors`:

```python
model.save_pretrained("lara-checkpoint")
restored_model = LaraModel.from_pretrained("lara-checkpoint")
```

`LaraConfig.save_pretrained()` and `LaraConfig.from_pretrained()` can also save and load the configuration on its own. Task-specific wrappers currently do not provide separate save/load helpers.

## Attention and Context

Each token attends to a sliding window with radius `local_window_size` and the first `num_global_tokens` positions. Those prefix global tokens attend to all non-padding tokens, allowing sequence-wide information to flow through the encoder. With a fixed window and number of global tokens, the attention work grows linearly with sequence length rather than using full dense attention for every token.

RoPE uses dynamic NTK scaling beyond `max_position_embeddings` (32,768 by default). Actual context limits and throughput still depend on memory, hardware, training, and the selected PyTorch attention backend. Variable-length batches are supported through padding masks; bucketing similar lengths can reduce wasted padded-token computation.

## Scope

Lara is a BERT-like architecture, not a drop-in BERT or ModernBERT replacement. It does not include a tokenizer, pretrained weights, a masked-language-modeling head, checkpoint conversion, or benchmark results. The model is randomly initialized and must be trained or fine-tuned for useful representations; compatibility with existing BERT-family checkpoints is not provided.

## Development

Install the project and development tools in editable mode, then run tests and lint:

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check lara.py tests
```

GitHub Actions runs the test and lint checks with CPU PyTorch on Python 3.10, 3.11, and 3.12.
