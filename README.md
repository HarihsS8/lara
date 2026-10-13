# Lara

Lara (Lightweight Advanced Representation Architecture) is a PyTorch encoder-only Transformer for bidirectional text representations. It combines dynamic NTK-scaled RoPE, pre-norm RMSNorm, SwiGLU feed-forward layers, local/global attention, and BERT-style WordPiece tokenization. PyTorch's scaled dot-product attention API dispatches to an optimized backend when the device and inputs support it.

## Install

Lara requires PyTorch 2.0 or newer, NumPy, Safetensors, and Hugging Face Tokenizers. Install the PyTorch wheel for your hardware using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install Lara:

```bash
python -m pip install .
```

For CPU-only development, install the CPU wheel and Lara's development tools:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

## Quick Start

Load a BERT-compatible WordPiece vocabulary. The tokenizer adds `[CLS]`/`[SEP]`, emits segment IDs for sentence pairs, and pads batches. In `attention_mask`, `1` marks a real token and `0` marks padding.

```python
import torch

from lara import LaraConfig, LaraModel
from lara_tokenizer import LaraTokenizer

tokenizer = LaraTokenizer.from_vocab_file("vocab.txt")
batch = tokenizer(
    ["A first sentence.", "Another example."],
    text_pair=["Its paired text.", "A second segment."],
    padding=True,
    truncation=True,
    max_length=128,
)
config = LaraConfig(
    vocab_size=len(tokenizer),
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    intermediate_size=768,
    local_window_size=128,
    num_global_tokens=1,
)
model = LaraModel(config)
outputs = model(**batch)

print(outputs.last_hidden_state.shape)  # [batch, sequence, hidden]
print(outputs.pooler_output.shape)      # [batch, hidden]
```

## BERT Tasks

The encoder supports `token_type_ids` for paired text. Task heads are available for sequence classification (`LaraForSequenceClassification`), token classification (`LaraForTokenClassification`), extractive question answering (`LaraForQuestionAnswering`), and multiple choice (`LaraForMultipleChoice`). `LaraForMaskedLM` provides the tied-weight masked-token prediction head; `LaraForPreTraining` combines masked language modeling with next-sentence prediction. These heads return logits and calculate losses when labels are supplied. `LaraForRetrieval` remains available for normalized embedding vectors.

For MLM batches, `LaraDataCollatorForLanguageModeling` dynamically pads tokenized examples, selects non-special tokens, applies BERT's 80/10/10 mask/random/unchanged corruption, and creates `-100` ignore labels:

```python
from lara import LaraConfig, LaraForMaskedLM
from lara_tokenizer import LaraDataCollatorForLanguageModeling, LaraTokenizer

tokenizer = LaraTokenizer.from_vocab_file("vocab.txt")
encoded = tokenizer(
    ["A training example.", "A second example."],
    padding=False,
    return_tensors=None,
    return_special_tokens_mask=True,
)
features = [
    {name: values[index] for name, values in encoded.items()}
    for index in range(len(encoded["input_ids"]))
]
batch = LaraDataCollatorForLanguageModeling(tokenizer)(features)
model = LaraForMaskedLM(LaraConfig(vocab_size=len(tokenizer)))
loss = model(**batch).loss
```

For next-sentence pretraining, pass paired encodings and `next_sentence_labels` to `LaraForPreTraining`; sentence-pair construction and negative-example sampling are dataset responsibilities.

Save and reload the base encoder with `config.json` and `model.safetensors`:

```python
model.save_pretrained("lara-checkpoint")
restored_model = LaraModel.from_pretrained("lara-checkpoint")
```

Task heads have the same `save_pretrained()` and `from_pretrained()` methods. For example, a fine-tuned classifier saves its label count and head weights alongside the encoder:

```python
classifier.save_pretrained("lara-classifier")
classifier = LaraForSequenceClassification.from_pretrained("lara-classifier")
```

`LaraConfig.save_pretrained()` and `LaraConfig.from_pretrained()` can also save and load the configuration on its own.

## Attention and Context

Each token attends to a sliding window with radius `local_window_size` and the first `num_global_tokens` positions. Those prefix global tokens attend to all non-padding tokens, allowing sequence-wide information to flow through the encoder. With a fixed window and number of global tokens, the attention work grows linearly with sequence length rather than using full dense attention for every token.

RoPE uses dynamic NTK scaling beyond `max_position_embeddings` (32,768 by default). Actual context limits and throughput still depend on memory, hardware, training, and the selected PyTorch attention backend. Variable-length batches are supported through padding masks; bucketing similar lengths can reduce wasted padded-token computation.

## Scope

Lara implements common BERT task heads and a WordPiece workflow, but it is not architecturally or checkpoint compatible with BERT or ModernBERT. Its attention pattern, RoPE, RMSNorm, and SwiGLU differ. It does not include pretrained weights, checkpoint conversion, a full training runner/dataset pipeline, or benchmark results. The model is randomly initialized and must be pretrained or fine-tuned for useful representations.

## Development

Install the project and development tools in editable mode, then run tests and lint:

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check lara.py lara_tokenizer.py tests
```

GitHub Actions runs the test and lint checks with CPU PyTorch on Python 3.10, 3.11, and 3.12.
