"""Comparison engine request-validation and alignment logic.

These do not load any model weights; the full three-model run is exercised by
selfcheck3.py. Here we pin down the input contract and the pure helpers.
"""
import compare3
import pytest


@pytest.fixture
def store():
    return compare3.Compare3Store()


def test_empty_prompt_rejected(store):
    with pytest.raises(ValueError):
        store.run_job({"prompt": "   ", "slots": ["a"]})


def test_missing_prompt_rejected(store):
    with pytest.raises(ValueError):
        store.run_job({"slots": ["a"]})


def test_too_many_slots_rejected(store):
    with pytest.raises(ValueError):
        store.run_job({"prompt": "hi", "slots": ["a", "b", "c", "a"]})


def test_zero_slots_rejected(store):
    with pytest.raises(ValueError):
        store.run_job({"prompt": "hi", "slots": []})


def test_unknown_slot_rejected(store):
    with pytest.raises(KeyError):
        store.run_job({"prompt": "hi", "slots": ["q"]})


def test_mean_ignores_none():
    assert compare3._mean([1.0, None, 3.0]) == 2.0
    assert compare3._mean([None]) is None
    assert compare3._mean([]) is None


def test_alignment_identical(store):
    s = {"a": {"generated_token_ids": [10, 11, 12]},
         "b": {"generated_token_ids": [10, 11, 12]}}
    a = store._alignment(["a", "b"], s)
    assert a["comparable"] is True
    assert a["shared_generated"] == 3
    assert a["first_divergence_step"] is None


def test_alignment_divergence(store):
    s = {"a": {"generated_token_ids": [10, 11, 99]},
         "b": {"generated_token_ids": [10, 11, 12, 13]}}
    a = store._alignment(["a", "b"], s)
    assert a["shared_generated"] == 2
    assert a["first_divergence_step"] == 2


def test_alignment_three_models(store):
    s = {"a": {"generated_token_ids": [1, 2, 3]},
         "b": {"generated_token_ids": [1, 2, 9]},
         "c": {"generated_token_ids": [1, 5, 3]}}
    a = store._alignment(["a", "b", "c"], s)
    assert a["shared_generated"] == 1  # diverges at step 1 (b,c differ from a? a=2,b=2,c=5)
