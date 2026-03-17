import re
import pytest
from unittest.mock import patch

from graders import (
    StrictStringInclusion,
    SoftOverlap,
    RegexMatch,
    LLMGrader,
    CustomGrader,
    CompositeGrader,
    grader_from_config,
)


class TestStrictStringInclusion:
    def test_exact_match(self):
        g = StrictStringInclusion("Au")
        assert g("The symbol is Au.") == 1.0

    def test_no_match(self):
        g = StrictStringInclusion("Au")
        assert g("The symbol is gold.") == 0.0

    def test_case_sensitive_no_match(self):
        g = StrictStringInclusion("Au", case_sensitive=True)
        assert g("The symbol is au.") == 0.0

    def test_case_insensitive_match(self):
        g = StrictStringInclusion("Au", case_sensitive=False)
        assert g("The symbol is au.") == 1.0

    def test_case_insensitive_no_match(self):
        g = StrictStringInclusion("Au", case_sensitive=False)
        assert g("The symbol is gold.") == 0.0

    def test_substring_in_larger_text(self):
        g = StrictStringInclusion("206")
        assert g("The human body has 206 bones.") == 1.0


class TestSoftOverlap:
    def test_exact_substring_match(self):
        g = SoftOverlap("hello")
        assert g("hello world") == 1.0

    def test_below_threshold_returns_zero(self):
        g = SoftOverlap("hello", similarity_threshold=0.8)
        assert g("completely different text zzz") == 0.0

    def test_empty_reference_returns_zero(self):
        g = SoftOverlap("")
        assert g("anything") == 0.0

    def test_zero_threshold_always_scores(self):
        g = SoftOverlap("hello", similarity_threshold=0.0)
        # Any response should yield a non-negative score
        assert g("zzzzz") >= 0.0

    def test_substring_edit_distance_exact(self):
        assert SoftOverlap._substring_edit_distance("hello world", "hello") == 0

    def test_substring_edit_distance_at_end(self):
        assert SoftOverlap._substring_edit_distance("hello world", "world") == 0

    def test_substring_edit_distance_one_edit(self):
        # "helo" requires 1 deletion to match "hello" substring
        assert SoftOverlap._substring_edit_distance("hello world", "helo") == 1

    def test_substring_edit_distance_target_longer_than_source(self):
        dist = SoftOverlap._substring_edit_distance("hi", "hello")
        assert dist >= 3  # at least 3 insertions needed


class TestRegexMatch:
    def test_digit_match(self):
        g = RegexMatch(r"\d+")
        assert g("The answer is 42.") == 1.0

    def test_no_match(self):
        g = RegexMatch(r"\d+")
        assert g("No numbers here.") == 0.0

    def test_case_insensitive_flag(self):
        g = RegexMatch(r"hello", flags=re.IGNORECASE)
        assert g("HELLO WORLD") == 1.0

    def test_complex_pattern(self):
        g = RegexMatch(r"\bAu\b")
        assert g("Element Au is gold.") == 1.0
        assert g("Because (auto)") == 0.0


class TestLLMGrader:
    @patch("graders.requests.post")
    def test_parses_score(self, mock_post):
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Looks good. Score: 0.85"}}]
        }
        g = LLMGrader(api_key="test-key")
        score = g("The answer is Au.")
        assert score == pytest.approx(0.85)

    @patch("graders.requests.post")
    def test_clamps_above_one(self, mock_post):
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "1.5"}}]
        }
        g = LLMGrader(api_key="test-key")
        assert g("test") == 1.0

    @patch("graders.requests.post")
    def test_clamps_below_zero(self, mock_post):
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "-0.1 which rounds to 0"}}]
        }
        g = LLMGrader(api_key="test-key")
        # -0.1 gets clamped; but re.findall picks "0.1" from "-0.1", then last float is "0"
        score = g("test")
        assert 0.0 <= score <= 1.0

    @patch("graders.requests.post")
    def test_returns_zero_no_float(self, mock_post):
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "No score here."}}]
        }
        g = LLMGrader(api_key="test-key")
        assert g("test") == 0.0

    @patch("graders.requests.post")
    def test_uses_reference_in_prompt(self, mock_post):
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "0.9"}}]
        }
        g = LLMGrader(reference="Au", api_key="test-key")
        g("The chemical symbol is Au.")
        called_body = mock_post.call_args[1]["json"]
        assert "Au" in called_body["messages"][0]["content"]


class TestCustomGrader:
    def test_passthrough(self):
        g = CustomGrader(lambda r: 0.5)
        assert g("anything") == 0.5

    def test_uses_response_content(self):
        g = CustomGrader(lambda r: 1.0 if "correct" in r else 0.0)
        assert g("this is correct") == 1.0
        assert g("this is wrong") == 0.0


class TestCompositeGrader:
    def test_and_all_pass(self):
        g = CompositeGrader(
            [StrictStringInclusion("Au"), StrictStringInclusion("gold")], logic="AND"
        )
        assert g("Au is the symbol for gold") == 1.0

    def test_and_one_fails(self):
        g = CompositeGrader(
            [StrictStringInclusion("Au"), StrictStringInclusion("silver")], logic="AND"
        )
        assert g("Au is the symbol for gold") == 0.0

    def test_or_one_passes(self):
        g = CompositeGrader(
            [StrictStringInclusion("Au"), StrictStringInclusion("silver")], logic="OR"
        )
        assert g("Au is the symbol for gold") == 1.0

    def test_or_none_pass(self):
        g = CompositeGrader(
            [StrictStringInclusion("silver"), StrictStringInclusion("platinum")], logic="OR"
        )
        assert g("Au is the symbol for gold") == 0.0

    def test_and_partial_credit(self):
        g = CompositeGrader(
            [StrictStringInclusion("Au"), StrictStringInclusion("silver")],
            logic="AND",
            partial_credit=True,
        )
        assert g("Au is the symbol for gold") == pytest.approx(0.5)

    def test_or_partial_credit_returns_max(self):
        g = CompositeGrader(
            [StrictStringInclusion("Au"), StrictStringInclusion("silver")],
            logic="OR",
            partial_credit=True,
        )
        # max of [1.0, 0.0] = 1.0
        assert g("Au is the symbol for gold") == 1.0

    def test_raises_on_llm_grader(self):
        with pytest.raises(ValueError):
            CompositeGrader([LLMGrader(api_key="test")])

    def test_empty_graders_and(self):
        g = CompositeGrader([], logic="AND")
        assert g("anything") == 0.0


class TestGraderFromConfig:
    def test_strict_string_inclusion(self):
        g = grader_from_config({"type": "StrictStringInclusion", "args": {"substring": "Au"}})
        assert isinstance(g, StrictStringInclusion)
        assert g("Au") == 1.0

    def test_soft_overlap(self):
        g = grader_from_config({"type": "SoftOverlap", "args": {"reference": "hello"}})
        assert isinstance(g, SoftOverlap)
        assert g("hello") == 1.0

    def test_regex_match(self):
        g = grader_from_config({"type": "RegexMatch", "args": {"pattern": r"\d+"}})
        assert isinstance(g, RegexMatch)
        assert g("42") == 1.0

    def test_composite_grader_recursive(self):
        config = {
            "type": "CompositeGrader",
            "args": {
                "graders": [
                    {"type": "StrictStringInclusion", "args": {"substring": "Au"}},
                    {"type": "RegexMatch", "args": {"pattern": r"\d+"}},
                ],
                "logic": "AND",
                "partial_credit": False,
            },
        }
        g = grader_from_config(config)
        assert isinstance(g, CompositeGrader)
        assert g("Au has atomic number 79") == 1.0
        assert g("Au has no numbers") == 0.0

    def test_custom_grader_raises(self):
        with pytest.raises(ValueError):
            grader_from_config({"type": "CustomGrader"})

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            grader_from_config({"type": "UnknownGrader"})

    def test_no_args_key(self):
        g = grader_from_config({"type": "StrictStringInclusion", "args": {"substring": "x"}})
        assert isinstance(g, StrictStringInclusion)
