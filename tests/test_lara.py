import pytest
import torch

from lara import (
    LaraConfig,
    LaraForRetrieval,
    LaraForSequenceClassification,
    LaraModel,
)


@pytest.fixture
def config() -> LaraConfig:
    return LaraConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        local_window_size=2,
        num_global_tokens=1,
        max_position_embeddings=8,
    )


def test_encoder_outputs_and_zeroes_padding(config: LaraConfig) -> None:
    model = LaraModel(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
    attention_mask = input_ids.ne(config.pad_token_id)

    outputs = model(input_ids, attention_mask)

    assert outputs.last_hidden_state.shape == (2, 5, config.hidden_size)
    assert outputs.pooler_output.shape == (2, config.hidden_size)
    assert torch.count_nonzero(outputs.last_hidden_state[0, 3:]) == 0
    assert torch.isfinite(outputs.last_hidden_state).all()


def test_masked_token_ids_do_not_affect_valid_representations(config: LaraConfig) -> None:
    model = LaraModel(config).eval()
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
    first = torch.tensor([[1, 2, 3, 0, 0]])
    second = torch.tensor([[1, 2, 3, 17, 18]])

    first_output = model(first, attention_mask).last_hidden_state
    second_output = model(second, attention_mask).last_hidden_state

    torch.testing.assert_close(first_output[:, :3], second_output[:, :3])


def test_all_padding_batch_is_finite(config: LaraConfig) -> None:
    model = LaraModel(config).eval()
    input_ids = torch.zeros((2, 5), dtype=torch.long)

    outputs = model(input_ids, torch.zeros_like(input_ids))

    assert torch.isfinite(outputs.last_hidden_state).all()
    assert torch.count_nonzero(outputs.last_hidden_state) == 0
    assert torch.count_nonzero(outputs.pooler_output) == 0


def test_encoder_supports_backward(config: LaraConfig) -> None:
    model = LaraModel(config)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    model(input_ids).last_hidden_state.square().mean().backward()

    gradient = model.embeddings.word_embeddings.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_classification_head_returns_loss(config: LaraConfig) -> None:
    model = LaraForSequenceClassification(config, num_labels=3)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    labels = torch.tensor([0, 2])

    outputs = model(input_ids, input_ids.ne(config.pad_token_id), labels)

    assert outputs.logits.shape == (2, 3)
    assert outputs.loss is not None
    assert outputs.loss.ndim == 0
    assert torch.isfinite(outputs.loss)


def test_retrieval_embeddings_are_normalized(config: LaraConfig) -> None:
    model = LaraForRetrieval(config).eval()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])

    embeddings = model(input_ids)

    assert embeddings.shape == (2, config.hidden_size)
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(2))


def test_pretrained_checkpoint_round_trip(config: LaraConfig, tmp_path) -> None:
    model = LaraModel(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(input_ids).last_hidden_state

    model.save_pretrained(tmp_path)
    restored = LaraModel.from_pretrained(tmp_path).eval()
    actual = restored(input_ids).last_hidden_state

    assert restored.config == config
    torch.testing.assert_close(actual, expected)


def test_empty_sequences_are_rejected(config: LaraConfig) -> None:
    model = LaraModel(config)

    with pytest.raises(ValueError, match="at least one token"):
        model(torch.empty((1, 0), dtype=torch.long))


def test_empty_batches_are_rejected(config: LaraConfig) -> None:
    model = LaraModel(config)

    with pytest.raises(ValueError, match="at least one sequence"):
        model(torch.empty((0, 1), dtype=torch.long))


def test_invalid_config_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        LaraConfig(vocab_size=16, num_attention_heads=0)

    with pytest.raises(ValueError, match="dropout_prob"):
        LaraConfig(vocab_size=16, hidden_dropout_prob=1.0)
