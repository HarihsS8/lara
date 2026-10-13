import torch

from lara import (
    LaraConfig,
    LaraForMaskedLM,
    LaraForMultipleChoice,
    LaraForPreTraining,
    LaraForQuestionAnswering,
    LaraForTokenClassification,
)


def make_config() -> LaraConfig:
    return LaraConfig(
        vocab_size=40,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        local_window_size=2,
        num_global_tokens=1,
    )


def test_masked_lm_uses_tied_embeddings_and_loss() -> None:
    model = LaraForMaskedLM(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, -100, -100]])

    outputs = model(input_ids, labels=labels)

    assert outputs.logits.shape == (1, 4, 40)
    assert outputs.loss is not None and torch.isfinite(outputs.loss)
    assert model.mlm_head.decoder_weight is model.encoder.embeddings.word_embeddings.weight


def test_pretraining_combines_mlm_and_next_sentence_losses() -> None:
    model = LaraForPreTraining(make_config())
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    labels = torch.tensor([[-100, 2, -100], [-100, -100, 6]])
    next_sentence_labels = torch.tensor([0, 1])

    outputs = model(
        input_ids,
        token_type_ids=torch.tensor([[0, 0, 1], [0, 1, 1]]),
        labels=labels,
        next_sentence_labels=next_sentence_labels,
    )

    assert outputs.prediction_logits.shape == (2, 3, 40)
    assert outputs.next_sentence_logits.shape == (2, 2)
    assert outputs.loss is not None and torch.isfinite(outputs.loss)
    outputs.loss.backward()


def test_token_classification_returns_per_token_logits_and_loss() -> None:
    model = LaraForTokenClassification(make_config(), num_labels=5)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, 1, -100]])

    outputs = model(input_ids, labels=labels)

    assert outputs.logits.shape == (1, 4, 5)
    assert outputs.loss is not None and torch.isfinite(outputs.loss)


def test_question_answering_returns_span_logits_and_loss() -> None:
    model = LaraForQuestionAnswering(make_config())
    input_ids = torch.tensor([[1, 2, 3, 4]])

    outputs = model(input_ids, start_positions=torch.tensor([1]), end_positions=torch.tensor([2]))

    assert outputs.start_logits.shape == (1, 4)
    assert outputs.end_logits.shape == (1, 4)
    assert outputs.loss is not None and torch.isfinite(outputs.loss)


def test_multiple_choice_flattens_candidate_batch() -> None:
    model = LaraForMultipleChoice(make_config())
    input_ids = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    token_type_ids = torch.zeros_like(input_ids)

    outputs = model(input_ids, token_type_ids=token_type_ids, labels=torch.tensor([0, 1]))

    assert outputs.logits.shape == (2, 2)
    assert outputs.loss is not None and torch.isfinite(outputs.loss)


def test_task_checkpoint_restores_head_configuration_and_weights(tmp_path) -> None:
    model = LaraForTokenClassification(make_config(), num_labels=5).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    expected = model(input_ids).logits

    model.save_pretrained(tmp_path)
    restored = LaraForTokenClassification.from_pretrained(tmp_path).eval()
    actual = restored(input_ids).logits

    assert restored.num_labels == 5
    torch.testing.assert_close(actual, expected)
