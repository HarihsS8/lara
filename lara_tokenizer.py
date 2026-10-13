"""WordPiece tokenization and masked-language-model data preparation for Lara."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import torch
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from torch import Tensor


_SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


def _configure_bert_pipeline(tokenizer: Tokenizer, lowercase: bool) -> None:
    """Install BERT normalization, pre-tokenization, decoding, and pair templates."""
    special_ids = {token: tokenizer.token_to_id(token) for token in _SPECIAL_TOKENS}
    missing = [token for token, token_id in special_ids.items() if token_id is None]
    if missing:
        raise ValueError(f"Vocabulary is missing required special tokens: {', '.join(missing)}")
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=lowercase)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", special_ids["[CLS]"]),
            ("[SEP]", special_ids["[SEP]"]),
        ],
    )


class LaraTokenizer:
    """BERT-style WordPiece tokenizer returning padded PyTorch batches."""

    def __init__(self, backend: Tokenizer, lowercase: bool = True) -> None:
        self.backend = backend
        self.lowercase = lowercase
        self.pad_token_id = self._require_token_id("[PAD]")
        self.unk_token_id = self._require_token_id("[UNK]")
        self.cls_token_id = self._require_token_id("[CLS]")
        self.sep_token_id = self._require_token_id("[SEP]")
        self.mask_token_id = self._require_token_id("[MASK]")
        self.all_special_ids = [self._require_token_id(token) for token in _SPECIAL_TOKENS]

    def _require_token_id(self, token: str) -> int:
        token_id = self.backend.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer vocabulary does not contain {token}")
        return token_id

    @classmethod
    def from_vocab_file(cls, vocab_file: str | Path, lowercase: bool = True) -> LaraTokenizer:
        """Load a BERT-compatible WordPiece vocabulary from ``vocab.txt``."""
        tokenizer = Tokenizer(models.WordPiece.from_file(str(vocab_file), unk_token="[UNK]"))
        _configure_bert_pipeline(tokenizer, lowercase)
        return cls(tokenizer, lowercase=lowercase)

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> LaraTokenizer:
        """Load ``tokenizer.json`` written by :meth:`save_pretrained`."""
        directory = Path(directory)
        tokenizer_path = directory / "tokenizer.json"
        config_path = directory / "tokenizer_config.json"
        config: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open(encoding="utf-8") as config_file:
                loaded_config = json.load(config_file)
            if isinstance(loaded_config, dict):
                config = loaded_config
        backend = Tokenizer.from_file(str(tokenizer_path))
        return cls(backend, lowercase=bool(config.get("do_lower_case", True)))

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int = 30_000,
        min_frequency: int = 2,
        lowercase: bool = True,
    ) -> LaraTokenizer:
        """Train a WordPiece vocabulary from an iterator of text strings."""
        if vocab_size < len(_SPECIAL_TOKENS):
            raise ValueError("vocab_size must fit all required special tokens")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        backend = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        backend.normalizer = normalizers.BertNormalizer(lowercase=lowercase)
        backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(_SPECIAL_TOKENS),
            show_progress=False,
        )
        backend.train_from_iterator(texts, trainer=trainer)
        _configure_bert_pipeline(backend, lowercase)
        return cls(backend, lowercase=lowercase)

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size(with_added_tokens=True)

    def __len__(self) -> int:
        return self.vocab_size

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save the tokenizer JSON, standard ``vocab.txt``, and settings."""
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(directory / "tokenizer.json"))
        vocabulary = self.backend.get_vocab()
        ordered_tokens = sorted(vocabulary, key=vocabulary.__getitem__)
        with (directory / "vocab.txt").open("w", encoding="utf-8") as vocab_file:
            vocab_file.write("\n".join(ordered_tokens) + "\n")
        with (directory / "tokenizer_config.json").open(encoding="utf-8", mode="w") as config_file:
            json.dump({"do_lower_case": self.lowercase}, config_file, indent=2)
            config_file.write("\n")

    def __call__(
        self,
        text: str | Sequence[str],
        text_pair: Optional[str | Sequence[str]] = None,
        *,
        padding: bool | str = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt",
        return_special_tokens_mask: bool = False,
    ) -> dict[str, Tensor] | dict[str, list[list[int]]]:
        """Tokenize one or more strings, optionally forming BERT text pairs."""
        single_input = isinstance(text, str)
        texts = [text] if single_input else list(text)
        if not texts:
            raise ValueError("text batch cannot be empty")
        if text_pair is None:
            pairs: list[Optional[str]] = [None] * len(texts)
        elif isinstance(text_pair, str):
            if not single_input:
                raise ValueError("text_pair must contain one string per input")
            pairs = [text_pair]
        else:
            pairs = list(text_pair)
            if len(pairs) != len(texts):
                raise ValueError("text_pair must contain one string per input")

        if padding not in (True, False, "longest", "max_length"):
            raise ValueError("padding must be True, False, 'longest', or 'max_length'")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be at least 2 to preserve [CLS] and [SEP]")
        if max_length == 2 and any(pair is not None for pair in pairs):
            raise ValueError("paired inputs need max_length of at least 3 for both [SEP] markers")
        if padding == "max_length" and max_length is None:
            raise ValueError("max_length is required when padding='max_length'")
        inputs: list[str | tuple[str, str]] = [
            (item, pair) if pair is not None else item for item, pair in zip(texts, pairs)
        ]
        encodings = self.backend.encode_batch(inputs, add_special_tokens=True)
        rows: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "special_tokens_mask": [],
        }
        for encoding, pair in zip(encodings, pairs):
            sequence_ids = list(encoding.sequence_ids)
            values = {
                "input_ids": list(encoding.ids),
                "attention_mask": list(encoding.attention_mask),
                "token_type_ids": list(encoding.type_ids),
                "special_tokens_mask": list(encoding.special_tokens_mask),
            }
            if max_length is not None and len(values["input_ids"]) > max_length:
                if not truncation:
                    raise ValueError("encoded input exceeds max_length and truncation is disabled")
                while len(values["input_ids"]) > max_length:
                    sequence_lengths = {
                        sequence_id: sum(item == sequence_id for item in sequence_ids)
                        for sequence_id in (0, 1)
                    }
                    available_sequences = [
                        sequence_id
                        for sequence_id, length in sequence_lengths.items()
                        if length > 0
                    ]
                    if not available_sequences:
                        raise ValueError("max_length is too short for required special tokens")
                    sequence_to_truncate = max(
                        available_sequences, key=sequence_lengths.__getitem__
                    )
                    remove_index = max(
                        index
                        for index, sequence_id in enumerate(sequence_ids)
                        if sequence_id == sequence_to_truncate
                    )
                    for name in values:
                        values[name].pop(remove_index)
                    sequence_ids.pop(remove_index)
            for name, value in values.items():
                rows[name].append(value)

        if padding:
            target_length = (
                max_length if padding == "max_length" else max(map(len, rows["input_ids"]))
            )
            for row_index in range(len(texts)):
                amount = target_length - len(rows["input_ids"][row_index])
                if amount < 0:
                    raise ValueError("encoded input exceeds max_length")
                rows["input_ids"][row_index].extend([self.pad_token_id] * amount)
                rows["attention_mask"][row_index].extend([0] * amount)
                rows["token_type_ids"][row_index].extend([0] * amount)
                rows["special_tokens_mask"][row_index].extend([1] * amount)
        if not return_special_tokens_mask:
            rows.pop("special_tokens_mask")
        if return_tensors is None:
            return rows
        if return_tensors != "pt":
            raise ValueError("return_tensors must be 'pt' or None")
        if len({len(row) for row in rows["input_ids"]}) > 1:
            raise ValueError("padding is required to return tensors for variable-length inputs")
        return {name: torch.tensor(value, dtype=torch.long) for name, value in rows.items()}

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Convert a sequence of token IDs back to normalized text."""
        return self.backend.decode(list(token_ids), skip_special_tokens=skip_special_tokens)


class LaraDataCollatorForLanguageModeling:
    """Create dynamic padding and BERT's 80/10/10 MLM corruption targets."""

    def __init__(self, tokenizer: LaraTokenizer, mlm_probability: float = 0.15) -> None:
        if not 0.0 <= mlm_probability <= 1.0:
            raise ValueError("mlm_probability must be in [0, 1]")
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    @staticmethod
    def _feature_tensor(value: Any) -> Tensor:
        tensor = torch.as_tensor(value, dtype=torch.long)
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 1:
            raise ValueError("each collator feature must contain one unbatched sequence")
        return tensor

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
        if not features:
            raise ValueError("features cannot be empty")
        input_rows = [self._feature_tensor(feature["input_ids"]) for feature in features]
        max_length = max(row.numel() for row in input_rows)
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        special_tokens_mask = torch.ones_like(input_ids, dtype=torch.bool)
        special_token_ids = torch.tensor(self.tokenizer.all_special_ids, dtype=torch.long)
        for index, (feature, row) in enumerate(zip(features, input_rows)):
            length = row.numel()
            input_ids[index, :length] = row
            attention_mask[index, :length] = self._feature_tensor(
                feature.get("attention_mask", torch.ones_like(row))
            )
            token_type_ids[index, :length] = self._feature_tensor(
                feature.get("token_type_ids", torch.zeros_like(row))
            )
            if "special_tokens_mask" in feature:
                special_tokens_mask[index, :length] = self._feature_tensor(
                    feature["special_tokens_mask"]
                ).bool()
            else:
                special_tokens_mask[index, :length] = torch.isin(row, special_token_ids)

        labels = input_ids.clone()
        selected = (
            (torch.rand(input_ids.shape) < self.mlm_probability)
            & attention_mask.bool()
            & ~special_tokens_mask
        )
        labels[~selected] = -100

        replacement_draw = torch.rand(input_ids.shape)
        mask_positions = selected & (replacement_draw < 0.8)
        random_positions = selected & (replacement_draw >= 0.8) & (replacement_draw < 0.9)
        input_ids[mask_positions] = self.tokenizer.mask_token_id
        random_words = torch.randint(self.tokenizer.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[random_positions] = random_words[random_positions]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
