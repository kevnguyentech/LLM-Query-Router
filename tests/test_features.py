"""Unit tests for data/features.py's feature-extraction functions.

Run with: python -m pytest tests/test_features.py -v
"""
from data.features import (
    token_count,
    sentence_count,
    avg_word_length,
    has_math,
    has_code,
    digit_count,
    unique_word_ratio,
    named_entity_count,
    flesch_reading_ease,
    question_word_flags,
    choice_count,
    extract,
)


def test_token_count_empty_string_is_zero():
    assert token_count("") == 0


def test_token_count_grows_with_text_length():
    short = token_count("cat")
    long = token_count("cat cat cat cat cat cat cat cat cat cat")
    assert long > short


def test_sentence_count_counts_sentences():
    assert sentence_count("This is one. This is two. Three.") == 3


def test_sentence_count_single_sentence():
    assert sentence_count("Just one sentence here.") == 1


def test_avg_word_length_basic():
    assert avg_word_length("cat dog elephant") == (3 + 3 + 8) / 3


def test_avg_word_length_strips_punctuation():
    # "cat," and "dog." should count as 3 chars each, not 4
    assert avg_word_length("cat, dog.") == 3.0


def test_avg_word_length_empty_string_is_zero():
    assert avg_word_length("") == 0.0


def test_has_math_detects_operators():
    assert has_math("what is 2 + 2") == 1


def test_has_math_detects_function_names():
    assert has_math("compute sin of the angle") == 1


def test_has_math_negative_case():
    assert has_math("describe the moon") == 0


def test_has_code_detects_def():
    assert has_code("def foo(): pass") == 1


def test_has_code_negative_case():
    assert has_code("no code here at all") == 0


def test_digit_count():
    assert digit_count("I have 3 cats and 42 dogs") == 3


def test_digit_count_no_digits():
    assert digit_count("no numbers here") == 0


def test_unique_word_ratio():
    assert unique_word_ratio("the the the cat") == 0.5


def test_unique_word_ratio_all_unique():
    assert unique_word_ratio("cat dog bird") == 1.0


def test_unique_word_ratio_empty_string_is_zero():
    assert unique_word_ratio("") == 0.0


def test_choice_count_four_options():
    assert choice_count("A. one B. two C. three D. four") == 4


def test_choice_count_no_options():
    assert choice_count("no MCQ options here") == 0


def test_question_word_flags_what():
    flags = question_word_flags("What is the capital of France?")
    assert flags["starts_with_what"] == 1
    assert flags["starts_with_why"] == 0
    assert flags["starts_with_how"] == 0
    assert flags["starts_with_which"] == 0
    assert flags["starts_with_calc"] == 0


def test_question_word_flags_calculate():
    flags = question_word_flags("Calculate the area of the triangle.")
    assert flags["starts_with_calc"] == 1


def test_named_entity_count_finds_entities():
    assert named_entity_count("Barack Obama visited Paris in 2015.") == 3


def test_named_entity_count_no_entities():
    assert named_entity_count("the small cat sat quietly") == 0


def test_flesch_reading_ease_clamps_low_values(monkeypatch):
    import data.features as features
    monkeypatch.setattr(features.textstat, "flesch_reading_ease", lambda text: -999.0)
    assert features.flesch_reading_ease("irrelevant") == -50.0


def test_flesch_reading_ease_clamps_high_values(monkeypatch):
    import data.features as features
    monkeypatch.setattr(features.textstat, "flesch_reading_ease", lambda text: 999.0)
    assert features.flesch_reading_ease("irrelevant") == 120.0


def test_flesch_reading_ease_passes_through_normal_values(monkeypatch):
    import data.features as features
    monkeypatch.setattr(features.textstat, "flesch_reading_ease", lambda text: 55.5)
    assert features.flesch_reading_ease("irrelevant") == 55.5


def test_extract_returns_all_fifteen_features():
    feats = extract("What is 2 + 2? Calculate the answer.")
    expected_keys = {
        "token_count", "sentence_count", "avg_word_length", "has_math",
        "has_code", "digit_count", "unique_word_ratio", "named_entity_count",
        "flesch_reading_ease", "choice_count", "starts_with_what",
        "starts_with_why", "starts_with_how", "starts_with_which",
        "starts_with_calc",
    }
    assert set(feats.keys()) == expected_keys
    assert len(feats) == 15

    def test_has_math_does_not_fire_on_hyphens():
        assert has_math("a well-known state-of-the-art approach") == 0

    def test_has_math_still_catches_arithmetic_minus():
        assert has_math("3-2") == 1