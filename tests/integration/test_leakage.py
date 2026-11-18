import pytest

from scamdetector import config
from scamdetector.data import load_all_splits, prepare_dataset, verify_no_leakage


@pytest.fixture(scope="module")
def splits():
    if not config.TRAIN_CSV.exists():
        prepare_dataset()
    return load_all_splits()


@pytest.mark.integration
def test_no_dedupe_key_appears_in_two_splits(splits):
    assert verify_no_leakage(splits) == {
        "train__val": 0,
        "train__test": 0,
        "val__test": 0,
    }


@pytest.mark.integration
def test_dedupe_keys_are_unique_within_each_split(splits):
    for name, frame in splits.items():
        assert frame["dedupe_key"].is_unique, f"{name} contains a repeated key"


@pytest.mark.integration
def test_normalised_text_never_repeats_across_splits(splits):
    seen: dict[str, str] = {}
    for name, frame in splits.items():
        for text in frame["text"]:
            from scamdetector.features import normalise_text

            key = normalise_text(text).lower()
            assert key not in seen, f"text leaked from {seen[key]} into {name}"
            seen[key] = name


@pytest.mark.integration
def test_label_vocabulary_is_binary(splits):
    for frame in splits.values():
        assert set(frame["label"]) <= set(config.LABELS)
