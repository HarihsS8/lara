"""Lara: lightweight bidirectional encoder with local/global attention.

Tokenization is deliberately delegated to a production tokenizer (for example,
a BPE tokenizer configured with byte fallback). ``input_ids`` must use the same
vocabulary as ``LaraConfig.vocab_size``. Attention uses PyTorch SDPA, allowing
PyTorch to select FlashAttention or another optimized backend when supported.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


@dataclass
class LaraConfig:
    """Configuration for a Lara encoder.

    ``local_window_size`` is the radius on either side of each token. The first
    ``num_global_tokens`` positions are global: they attend to the full sequence
    and are visible to every local query.
    """

    vocab_size: int
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 2048
    local_window_size: int = 256
    num_global_tokens: int = 1
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    rope_ntk_scaling_factor: float = 2.0
    rms_norm_eps: float = 1e-6
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    pad_token_id: int = 0
    type_vocab_size: int = 2
    pooler_type: str = "mean"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside the configured vocabulary")
        if self.hidden_size <= 0 or self.num_hidden_layers <= 0 or self.num_attention_heads <= 0:
            raise ValueError(
                "hidden_size, num_hidden_layers, and num_attention_heads must be positive"
            )
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.intermediate_size <= 0 or self.max_position_embeddings <= 0:
            raise ValueError("intermediate_size and max_position_embeddings must be positive")
        if self.type_vocab_size <= 0:
            raise ValueError("type_vocab_size must be positive")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("rope_theta and rms_norm_eps must be positive")
        if self.local_window_size < 0 or self.num_global_tokens < 0:
            raise ValueError("local_window_size and num_global_tokens cannot be negative")
        if self.rope_ntk_scaling_factor < 1.0:
            raise ValueError("rope_ntk_scaling_factor must be at least 1.0")
        if not 0.0 <= self.hidden_dropout_prob < 1.0:
            raise ValueError("hidden_dropout_prob must be in [0, 1)")
        if not 0.0 <= self.attention_dropout_prob < 1.0:
            raise ValueError("attention_dropout_prob must be in [0, 1)")
        if self.pooler_type not in {"mean", "cls"}:
            raise ValueError("pooler_type must be 'mean' or 'cls'")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible copy of this configuration."""
        return asdict(self)

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save this configuration as ``config.json``."""
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(directory / "config.json", self.to_dict())

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> LaraConfig:
        """Load and validate a configuration saved by :meth:`save_pretrained`."""
        config_path = Path(directory) / "config.json"
        with config_path.open(encoding="utf-8") as config_file:
            config_data = json.load(config_file)
        if not isinstance(config_data, dict):
            raise ValueError(f"Expected a JSON object in {config_path}")
        return cls(**config_data)


@dataclass
class LaraOutput:
    """Base encoder outputs."""

    last_hidden_state: Tensor
    pooler_output: Tensor


@dataclass
class LaraTaskOutput:
    """Output from a task-specific head."""

    logits: Tensor
    loss: Optional[Tensor] = None


@dataclass
class LaraPreTrainingOutput:
    """Outputs for masked-token and next-sentence pretraining."""

    prediction_logits: Tensor
    next_sentence_logits: Tensor
    loss: Optional[Tensor] = None


@dataclass
class LaraQuestionAnsweringOutput:
    """Start/end logits and optional span loss for extractive QA."""

    start_logits: Tensor
    end_logits: Tensor
    loss: Optional[Tensor] = None


class LaraRMSNorm(nn.Module):
    """Bias-free RMS normalization with accumulation in float32."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float() * torch.rsqrt(
            hidden_states.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(input_dtype)


class LaraRoPE(nn.Module):
    """Rotary embeddings with dynamic NTK scaling beyond the training context."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 32768,
        theta: float = 10000.0,
        ntk_scaling_factor: float = 2.0,
    ) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        if max_position_embeddings <= 0 or theta <= 0 or ntk_scaling_factor < 1.0:
            raise ValueError("invalid RoPE context, theta, or NTK scaling factor")
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self.ntk_scaling_factor = ntk_scaling_factor

    def forward(self, sequence_length: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return cosine and sine tables shaped ``[1, 1, sequence, head_dim]``."""
        scale = 1.0
        if sequence_length > self.max_position_embeddings and self.ntk_scaling_factor > 1.0:
            scale = self.ntk_scaling_factor * sequence_length / self.max_position_embeddings
            scale -= self.ntk_scaling_factor - 1.0
        base = (
            self.theta * scale ** (self.head_dim / (self.head_dim - 2))
            if self.head_dim > 2
            else self.theta
        )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim
            )
        )
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]

    @staticmethod
    def _rotate_half(tensor: Tensor) -> Tensor:
        first, second = tensor.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def apply(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        """Apply rotary position encoding to query and key tensors."""
        cos, sin = self(query.shape[-2], query.device)
        cos = cos.to(dtype=query.dtype)
        sin = sin.to(dtype=query.dtype)
        return (
            query * cos + self._rotate_half(query) * sin,
            key * cos.to(dtype=key.dtype) + self._rotate_half(key) * sin.to(dtype=key.dtype),
        )


class LaraSwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, value = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.dropout(self.down_proj(F.silu(gate) * value))


class LaraSelfAttention(nn.Module):
    """Sliding-window attention with full-sequence prefix global tokens.

    Local keys and values are gathered into a bounded window, so the local path
    uses O(sequence_length * window_size) attention memory. Each layer also
    computes full attention for only the configured global query positions.
    Both paths use ``scaled_dot_product_attention`` for optimized backend
    dispatch, including FlashAttention where its constraints are satisfied.
    """

    def __init__(self, config: LaraConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.local_window_size = config.local_window_size
        self.num_global_tokens = config.num_global_tokens
        self.attention_dropout = config.attention_dropout_prob
        self.qkv_proj = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = LaraRoPE(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rope_ntk_scaling_factor,
        )

    def _local_attention(
        self, query: Tensor, key: Tensor, value: Tensor, token_mask: Tensor
    ) -> Tensor:
        batch_size, heads, sequence_length, head_dim = query.shape
        radius = self.local_window_size
        offsets = torch.arange(-radius, radius + 1, device=query.device)
        positions = torch.arange(sequence_length, device=query.device)
        raw_indices = positions[:, None] + offsets[None, :]
        in_bounds = (raw_indices >= 0) & (raw_indices < sequence_length)
        indices = raw_indices.clamp(0, max(sequence_length - 1, 0))

        local_key = key[:, :, indices, :]
        local_value = value[:, :, indices, :]
        local_mask = token_mask[:, indices] & in_bounds[None, :, :]

        global_count = min(self.num_global_tokens, sequence_length)
        if global_count:
            global_key = (
                key[:, :, :global_count, :].unsqueeze(2).expand(-1, -1, sequence_length, -1, -1)
            )
            global_value = (
                value[:, :, :global_count, :].unsqueeze(2).expand(-1, -1, sequence_length, -1, -1)
            )
            local_key = torch.cat((local_key, global_key), dim=3)
            local_value = torch.cat((local_value, global_value), dim=3)
            global_mask = token_mask[:, None, :global_count].expand(-1, sequence_length, -1)
            local_mask = torch.cat((local_mask, global_mask), dim=-1)

        key_count = local_key.shape[3]
        # An all-padding row needs one harmless key to avoid all-masked SDPA rows.
        local_mask = local_mask.clone()
        local_mask[:, :, 0] |= ~token_mask
        flat_query = query.permute(0, 2, 1, 3).reshape(
            batch_size * sequence_length, heads, 1, head_dim
        )
        flat_key = local_key.permute(0, 2, 1, 3, 4).reshape(
            batch_size * sequence_length, heads, key_count, head_dim
        )
        flat_value = local_value.permute(0, 2, 1, 3, 4).reshape(
            batch_size * sequence_length, heads, key_count, head_dim
        )
        flat_mask = local_mask.reshape(batch_size * sequence_length, 1, 1, key_count)
        attended = F.scaled_dot_product_attention(
            flat_query,
            flat_key,
            flat_value,
            attn_mask=flat_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.reshape(batch_size, sequence_length, heads, head_dim).permute(
            0, 2, 1, 3
        )
        return attended * token_mask[:, None, :, None].to(attended.dtype)

    def _global_attention(
        self, query: Tensor, key: Tensor, value: Tensor, token_mask: Tensor
    ) -> Tensor:
        global_count = min(self.num_global_tokens, query.shape[-2])
        if global_count == 0:
            return query[:, :, :0, :]
        global_query = query[:, :, :global_count, :]
        full_mask = token_mask[:, None, None, :].expand(-1, 1, global_count, -1).clone()
        # Keep SDPA well-defined for examples containing only padding.
        full_mask[:, :, :, 0] |= ~token_mask.any(dim=-1)[:, None, None]
        attended = F.scaled_dot_product_attention(
            global_query,
            key,
            value,
            attn_mask=full_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        return attended * token_mask[:, None, :global_count, None].to(attended.dtype)

    def forward(self, hidden_states: Tensor, token_mask: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query, key = self.rope.apply(query, key)

        attended = self._local_attention(query, key, value, token_mask)
        global_count = min(self.num_global_tokens, sequence_length)
        if global_count:
            global_attended = self._global_attention(query, key, value, token_mask)
            attended = torch.cat((global_attended, attended[:, :, global_count:, :]), dim=2)

        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_size)
        )
        return self.out_proj(attended)


class LaraEncoderLayer(nn.Module):
    """Pre-norm encoder block with local/global attention and SwiGLU."""

    def __init__(self, config: LaraConfig) -> None:
        super().__init__()
        self.attention_norm = LaraRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = LaraSelfAttention(config)
        self.ffn_norm = LaraRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = LaraSwiGLUFFN(
            config.hidden_size, config.intermediate_size, config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, token_mask: Tensor) -> Tensor:
        hidden_states = hidden_states + self.dropout(
            self.attention(self.attention_norm(hidden_states), token_mask)
        )
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states * token_mask[:, :, None].to(hidden_states.dtype)


class LaraEmbeddings(nn.Module):
    """Token and segment embeddings for tokenizer-produced IDs."""

    def __init__(self, config: LaraConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: Tensor, token_type_ids: Optional[Tensor] = None) -> Tensor:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if token_type_ids.shape != input_ids.shape:
            raise ValueError("token_type_ids must have the same shape as input_ids")
        if torch.any(token_type_ids < 0) or torch.any(
            token_type_ids >= self.token_type_embeddings.num_embeddings
        ):
            raise ValueError("token_type_ids contain IDs outside type_vocab_size")
        return self.dropout(
            self.word_embeddings(input_ids) + self.token_type_embeddings(token_type_ids)
        )


class LaraPreTrainedModel(nn.Module):
    """Base class for task models with safe, self-describing checkpoints."""

    def __init__(self, config: LaraConfig) -> None:
        super().__init__()
        self.config = config

    def _head_config(self) -> dict[str, object]:
        return {}

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save task metadata and weights as JSON and Safetensors files."""
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(directory)
        _write_json_atomic(
            directory / "model_config.json",
            {"model_type": type(self).__name__, "head_config": self._head_config()},
        )
        state_dict = {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in self.state_dict().items()
        }
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".safetensors", dir=directory, delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            save_file(state_dict, str(temporary_path))
            os.replace(temporary_path, directory / "model.safetensors")
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> LaraPreTrainedModel:
        """Load a saved task model and validate its recorded model type."""
        directory = Path(directory)
        with (directory / "model_config.json").open(encoding="utf-8") as config_file:
            model_config = json.load(config_file)
        if not isinstance(model_config, dict) or model_config.get("model_type") != cls.__name__:
            raise ValueError(f"Checkpoint does not contain a {cls.__name__} model")
        head_config = model_config.get("head_config", {})
        if not isinstance(head_config, dict):
            raise ValueError("Invalid task-head configuration in checkpoint")
        model = cls(LaraConfig.from_pretrained(directory), **head_config)
        state_dict = load_file(
            str(directory / "model.safetensors"),
            device=str(map_location),
        )
        model.load_state_dict(state_dict, strict=strict)
        return model


class LaraModel(nn.Module):
    """Bidirectional encoder producing token representations and pooled output."""

    def __init__(self, config: LaraConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = LaraEmbeddings(config)
        self.layers = nn.ModuleList(
            LaraEncoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.final_norm = LaraRMSNorm(config.hidden_size, config.rms_norm_eps)

    def _pool(self, hidden_states: Tensor, token_mask: Tensor) -> Tensor:
        if self.config.pooler_type == "cls":
            return hidden_states[:, 0]
        weights = token_mask[:, :, None].to(hidden_states.dtype)
        return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save config and weights in JSON and non-executable Safetensors files."""
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(directory)
        state_dict = {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in self.state_dict().items()
        }
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".safetensors", dir=directory, delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            save_file(state_dict, str(temporary_path))
            os.replace(temporary_path, directory / "model.safetensors")
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> LaraModel:
        """Load a Lara checkpoint without unpickling executable data."""
        directory = Path(directory)
        config = LaraConfig.from_pretrained(directory)
        model = cls(config)
        state_dict = load_file(
            str(directory / "model.safetensors"),
            device=str(map_location),
        )
        model.load_state_dict(state_dict, strict=strict)
        return model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> LaraOutput:
        """Encode ``input_ids``; mask values of one indicate non-padding tokens."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[0] == 0:
            raise ValueError("input_ids must contain at least one sequence per batch")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token per sequence")
        if attention_mask is None:
            token_mask = input_ids.ne(self.config.pad_token_id)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            token_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        hidden_states = self.embeddings(input_ids, token_type_ids)
        hidden_states = hidden_states * token_mask[:, :, None].to(hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, token_mask)
        hidden_states = self.final_norm(hidden_states)
        hidden_states = hidden_states * token_mask[:, :, None].to(hidden_states.dtype)
        return LaraOutput(hidden_states, self._pool(hidden_states, token_mask))


class LaraForSequenceClassification(LaraPreTrainedModel):
    """Sequence-classification head over masked-mean or CLS pooled features."""

    def __init__(self, config: LaraConfig, num_labels: int) -> None:
        super().__init__(config)
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        self.num_labels = num_labels
        self.encoder = LaraModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def _head_config(self) -> dict[str, object]:
        return {"num_labels": self.num_labels}

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> LaraTaskOutput:
        pooled = self.encoder(input_ids, attention_mask, token_type_ids).pooler_output
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            if logits.shape[-1] == 1:
                loss = F.mse_loss(logits.squeeze(-1), labels.to(logits.dtype).view(-1))
            elif labels.is_floating_point():
                loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
            else:
                loss = F.cross_entropy(logits, labels.long())
        return LaraTaskOutput(logits=logits, loss=loss)


class LaraForTokenClassification(LaraPreTrainedModel):
    """Token classification head for tasks such as NER and tagging."""

    def __init__(self, config: LaraConfig, num_labels: int) -> None:
        super().__init__(config)
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        self.num_labels = num_labels
        self.encoder = LaraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def _head_config(self) -> dict[str, object]:
        return {"num_labels": self.num_labels}

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> LaraTaskOutput:
        hidden_states = self.encoder(input_ids, attention_mask, token_type_ids).last_hidden_state
        logits = self.classifier(self.dropout(hidden_states))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("token-classification labels must match input_ids shape")
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1).long())
        return LaraTaskOutput(logits=logits, loss=loss)


class LaraForQuestionAnswering(LaraPreTrainedModel):
    """Extractive QA head predicting start and end token positions."""

    def __init__(self, config: LaraConfig) -> None:
        super().__init__(config)
        self.encoder = LaraModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> LaraQuestionAnsweringOutput:
        hidden_states = self.encoder(input_ids, attention_mask, token_type_ids).last_hidden_state
        start_logits, end_logits = self.qa_outputs(hidden_states).unbind(dim=-1)
        loss = None
        if (start_positions is None) != (end_positions is None):
            raise ValueError("start_positions and end_positions must be provided together")
        if start_positions is not None and end_positions is not None:
            ignored_index = input_ids.shape[1]
            start_positions = start_positions.reshape(-1).long().clamp(0, ignored_index)
            end_positions = end_positions.reshape(-1).long().clamp(0, ignored_index)
            loss = (
                F.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
                + F.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            ) / 2
        return LaraQuestionAnsweringOutput(start_logits, end_logits, loss)


class LaraForMultipleChoice(LaraPreTrainedModel):
    """Multiple-choice classification over candidate sequences."""

    def __init__(self, config: LaraConfig) -> None:
        super().__init__(config)
        self.encoder = LaraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> LaraTaskOutput:
        if input_ids.ndim != 3:
            raise ValueError("multiple-choice input_ids must have shape [batch, choices, sequence]")
        batch_size, num_choices, sequence_length = input_ids.shape
        flat_input_ids = input_ids.reshape(batch_size * num_choices, sequence_length)
        flat_attention_mask = (
            attention_mask.reshape(batch_size * num_choices, sequence_length)
            if attention_mask is not None
            else None
        )
        flat_token_type_ids = (
            token_type_ids.reshape(batch_size * num_choices, sequence_length)
            if token_type_ids is not None
            else None
        )
        pooled = self.encoder(
            flat_input_ids, flat_attention_mask, flat_token_type_ids
        ).pooler_output
        logits = self.classifier(self.dropout(pooled)).reshape(batch_size, num_choices)
        loss = None
        if labels is not None:
            if labels.shape != (batch_size,):
                raise ValueError("multiple-choice labels must have shape [batch]")
            loss = F.cross_entropy(logits, labels.long())
        return LaraTaskOutput(logits=logits, loss=loss)


class _LaraMLMHead(nn.Module):
    """BERT-style dense/activation/norm transform before tied vocabulary logits."""

    def __init__(self, config: LaraConfig, embeddings: LaraEmbeddings) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder_weight = embeddings.word_embeddings.weight
        self.decoder_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.layer_norm(F.gelu(self.dense(hidden_states)))
        return F.linear(hidden_states, self.decoder_weight, self.decoder_bias)


class LaraForMaskedLM(LaraPreTrainedModel):
    """Masked language modeling head with embeddings tied to the decoder."""

    def __init__(self, config: LaraConfig) -> None:
        super().__init__(config)
        self.encoder = LaraModel(config)
        self.mlm_head = _LaraMLMHead(config, self.encoder.embeddings)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> LaraTaskOutput:
        hidden_states = self.encoder(input_ids, attention_mask, token_type_ids).last_hidden_state
        logits = self.mlm_head(hidden_states)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("masked-language-model labels must match input_ids shape")
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1).long())
        return LaraTaskOutput(logits=logits, loss=loss)


class LaraForPreTraining(LaraPreTrainedModel):
    """BERT-style masked language modeling and next-sentence prediction."""

    def __init__(self, config: LaraConfig) -> None:
        super().__init__(config)
        self.encoder = LaraModel(config)
        self.mlm_head = _LaraMLMHead(config, self.encoder.embeddings)
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )
        self.next_sentence_head = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        next_sentence_labels: Optional[Tensor] = None,
    ) -> LaraPreTrainingOutput:
        encoded = self.encoder(input_ids, attention_mask, token_type_ids)
        prediction_logits = self.mlm_head(encoded.last_hidden_state)
        next_sentence_logits = self.next_sentence_head(self.pooler(encoded.last_hidden_state[:, 0]))
        losses = []
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("masked-language-model labels must match input_ids shape")
            losses.append(
                F.cross_entropy(
                    prediction_logits.reshape(-1, prediction_logits.shape[-1]),
                    labels.reshape(-1).long(),
                )
            )
        if next_sentence_labels is not None:
            if next_sentence_labels.shape != (input_ids.shape[0],):
                raise ValueError("next_sentence_labels must have shape [batch]")
            losses.append(F.cross_entropy(next_sentence_logits, next_sentence_labels.long()))
        loss = torch.stack(losses).sum() if losses else None
        return LaraPreTrainingOutput(prediction_logits, next_sentence_logits, loss)


class LaraForRetrieval(LaraPreTrainedModel):
    """Encoder wrapper returning L2-normalized sentence embeddings."""

    def __init__(self, config: LaraConfig) -> None:
        super().__init__(config)
        self.encoder = LaraModel(config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        pooled = self.encoder(input_ids, attention_mask, token_type_ids).pooler_output
        return F.normalize(pooled, p=2, dim=-1)


if __name__ == "__main__":
    # Requires PyTorch: python lara.py
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
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    print(output.last_hidden_state.shape, output.pooler_output.shape)
