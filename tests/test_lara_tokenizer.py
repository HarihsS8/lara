import torch

from lara import LaraConfig, LaraForMaskedLM
from lara_tokenizer import LaraDataCollatorForLanguageModeling, LaraTokenizer


def test_wordpiece_pairs_padding_mlm_collation_and_persistence(tmp_path) -> None:
    tokenizer = LaraTokenizer.train_from_iterator(
        ["Hello there friend", "Hello world friend", "world there"],
        vocab_size=48,
        min_frequency=1,
    )
    encoded = tokenizer(
        ["Hello world", "Hello"],
        text_pair=["there friend", "world"],
        padding="max_length",
        max_length=8,
        return_special_tokens_mask=True,
    )

    assert encoded["input_ids"].shape == (2, 8)
    assert encoded["token_type_ids"][0].max() == 1
    assert encoded["attention_mask"][0].sum() <= 8

    collator = LaraDataCollatorForLanguageModeling(tokenizer, mlm_probability=1.0)
    features = [
        {name: value[index] for name, value in encoded.items()}
        for index in range(encoded["input_ids"].shape[0])
    ]
    batch = collator(features)

    assert batch["input_ids"].shape == encoded["input_ids"].shape
    assert (batch["labels"][encoded["attention_mask"] == 0] == -100).all()
    assert (batch["labels"][encoded["special_tokens_mask"].bool()] == -100).all()
    assert (batch["labels"] != -100).any()

    model = LaraForMaskedLM(
        LaraConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
        )
    )
    assert torch.isfinite(model(**batch).loss)

    tokenizer.save_pretrained(tmp_path)
    restored = LaraTokenizer.from_pretrained(tmp_path)
    torch.testing.assert_close(
        tokenizer("Hello world")["input_ids"], restored("Hello world")["input_ids"]
    )
    vocab_tokenizer = LaraTokenizer.from_vocab_file(tmp_path / "vocab.txt")
    torch.testing.assert_close(
        tokenizer("Hello world")["input_ids"], vocab_tokenizer("Hello world")["input_ids"]
    )


def test_pair_truncation_preserves_bert_special_tokens() -> None:
    tokenizer = LaraTokenizer.train_from_iterator(
        ["Hello there friend", "world friend there"],
        vocab_size=48,
        min_frequency=1,
    )

    encoded = tokenizer(
        "Hello there friend",
        text_pair="world friend there",
        padding="max_length",
        truncation=True,
        max_length=5,
    )

    valid_ids = encoded["input_ids"][0][encoded["attention_mask"][0].bool()].tolist()
    assert valid_ids[0] == tokenizer.cls_token_id
    assert valid_ids[-1] == tokenizer.sep_token_id
    assert valid_ids.count(tokenizer.sep_token_id) == 2
    assert encoded["token_type_ids"][0].max() == 1
