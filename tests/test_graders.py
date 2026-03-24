import re
import pytest
from unittest.mock import patch

from graders import (
    StrictStringInclusion,
    SoftOverlap,
    RegexMatch,
    IntegerMatch,
    NumericMatch,
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


class TestIntegerMatch:
    # ── digit form ──────────────────────────────────────────────────────────
    def test_digit_in_sentence(self):
        assert IntegerMatch(4)("Arsenal have lost 4 consecutive finals") == 1.0

    def test_digit_with_punctuation(self):
        assert IntegerMatch(4)("The answer is 4.") == 1.0

    def test_digit_ordinal_suffix(self):
        # "4th" still contains a standalone digit 4
        assert IntegerMatch(4)("4th consecutive final") == 1.0

    def test_digit_no_match_larger(self):
        assert IntegerMatch(4)("14 finals") == 0.0

    def test_digit_no_match_prefix(self):
        assert IntegerMatch(4)("24") == 0.0

    def test_digit_no_match_suffix(self):
        assert IntegerMatch(4)("40") == 0.0

    # ── word form ────────────────────────────────────────────────────────────
    def test_word_lowercase(self):
        assert IntegerMatch(4)("They have lost four consecutive finals") == 1.0

    def test_word_capitalized(self):
        assert IntegerMatch(4)("Four finals have been lost") == 1.0

    def test_word_with_punctuation(self):
        assert IntegerMatch(4)("The count is four.") == 1.0

    def test_word_no_match_fourteen(self):
        assert IntegerMatch(4)("fourteen") == 0.0

    def test_word_no_match_twenty_four(self):
        # "four" in "twenty-four" should NOT match when looking for 4
        assert IntegerMatch(4)("twenty-four") == 0.0

    def test_word_no_match_forty_four(self):
        assert IntegerMatch(4)("forty-four") == 0.0

    def test_word_no_match_fourth(self):
        # ordinal "fourth" should not match cardinal "four"
        assert IntegerMatch(4)("the fourth final") == 0.0

    # ── compound word (value=24) ─────────────────────────────────────────────
    def test_compound_digit(self):
        assert IntegerMatch(24)("24 hours") == 1.0

    def test_compound_word(self):
        assert IntegerMatch(24)("twenty-four hours") == 1.0

    def test_compound_word_case_insensitive(self):
        assert IntegerMatch(24)("Twenty-Four") == 1.0

    def test_compound_no_match_in_larger(self):
        # "twenty-four" in "one-hundred-twenty-four" (=124) should NOT match 24
        assert IntegerMatch(24)("one-hundred-twenty-four") == 0.0

    def test_compound_no_match_just_four(self):
        assert IntegerMatch(24)("four") == 0.0

    def test_compound_no_match_digit_four(self):
        assert IntegerMatch(24)("only 4 here") == 0.0

    # ── value=7 ─────────────────────────────────────────────────────────────
    def test_seven_digit(self):
        assert IntegerMatch(7)("7 days a week") == 1.0

    def test_seven_word(self):
        assert IntegerMatch(7)("seven days") == 1.0

    def test_seventeen_no_match(self):
        assert IntegerMatch(7)("seventeen") == 0.0

    def test_seventy_no_match(self):
        # "seventy" contains "seven" but lookahead blocks it ("ty" follows)
        assert IntegerMatch(7)("seventy") == 0.0

    def test_seventy_seven_digit_no_match(self):
        assert IntegerMatch(7)("77") == 0.0

    # ── _int_to_words ────────────────────────────────────────────────────────
    def test_words_zero(self):
        assert IntegerMatch._int_to_words(0) == "zero"

    def test_words_four(self):
        assert IntegerMatch._int_to_words(4) == "four"

    def test_words_twenty(self):
        assert IntegerMatch._int_to_words(20) == "twenty"

    def test_words_twenty_four(self):
        assert IntegerMatch._int_to_words(24) == "twenty-four"

    def test_words_ninety_nine(self):
        assert IntegerMatch._int_to_words(99) == "ninety-nine"

    def test_words_hundred_is_none(self):
        assert IntegerMatch._int_to_words(100) is None

    def test_words_negative_is_none(self):
        assert IntegerMatch._int_to_words(-1) is None


class TestNumericMatch:
    # ── init from fraction string ────────────────────────────────────────────
    def test_init_fraction_string_2_3(self):
        g = NumericMatch("2/3")
        assert abs(g.value - 2 / 3) < 1e-12

    def test_init_fraction_string_2_7(self):
        g = NumericMatch("2/7")
        assert abs(g.value - 2 / 7) < 1e-12

    def test_init_float(self):
        g = NumericMatch(0.5)
        assert g.value == 0.5

    def test_init_int(self):
        g = NumericMatch(4)
        assert g.value == 4.0

    # ── exact decimal (default tolerance ≈ 0) ────────────────────────────────
    def test_exact_decimal_match(self):
        assert NumericMatch(0.5)("The answer is 0.5") == 1.0

    def test_exact_decimal_no_match(self):
        assert NumericMatch(0.5)("The answer is 0.6") == 0.0

    def test_exact_integer_match(self):
        assert NumericMatch(4)("The count is 4") == 1.0

    # ── 2/7 ≈ 0.28571 with tolerance=1e-3 ───────────────────────────────────
    def test_2_7_exact_fraction(self):
        assert NumericMatch("2/7")("ratio is 2/7") == 1.0

    def test_2_7_latex_fraction(self):
        assert NumericMatch("2/7")("ratio is \\frac{2}{7}") == 1.0

    def test_2_7_decimal_approx(self):
        # 0.286 differs from 2/7 by ~3.3e-4 < 1e-3
        assert NumericMatch("2/7", tolerance=1e-3)("approximately 0.286") == 1.0

    def test_2_7_percentage(self):
        # 28.6% = 0.286
        assert NumericMatch("2/7", tolerance=1e-3)("about 28.6%") == 1.0

    def test_2_7_percentage_too_rounded(self):
        # 29% = 0.29, differs by ~4.3e-3 > 1e-3
        assert NumericMatch("2/7", tolerance=1e-3)("about 29%") == 0.0

    def test_2_7_wrong_fraction(self):
        assert NumericMatch("2/7", tolerance=1e-3)("the answer is 2/3") == 0.0

    def test_2_7_wrong_decimal(self):
        assert NumericMatch("2/7", tolerance=1e-3)("the answer is 0.5") == 0.0

    # ── 2/3 ≈ 0.6667 with tolerance=1e-3 ────────────────────────────────────
    def test_2_3_exact_fraction(self):
        assert NumericMatch("2/3")("answer is 2/3") == 1.0

    def test_2_3_latex(self):
        assert NumericMatch("2/3")("answer is \\frac{2}{3}") == 1.0

    def test_2_3_decimal_approx(self):
        # 0.667 differs from 2/3 by ~3.3e-4 < 1e-3
        assert NumericMatch("2/3", tolerance=1e-3)("approximately 0.667") == 1.0

    def test_2_3_percentage(self):
        assert NumericMatch("2/3", tolerance=1e-3)("about 66.7%") == 1.0

    def test_2_3_wrong_decimal(self):
        assert NumericMatch("2/3", tolerance=1e-3)("the answer is 0.5") == 0.0

    def test_2_3_wrong_fraction(self):
        assert NumericMatch("2/3", tolerance=1e-3)("the answer is 2/7") == 0.0

    # ── edge cases ────────────────────────────────────────────────────────────
    def test_denominator_zero_no_crash(self):
        # "4/0" is silently skipped (division by zero guard); "4" is also blocked
        # by the lookahead (?![/%]) since it precedes "/". No crash, returns 0.0.
        assert NumericMatch(4.0)("the value is 4/0") == 0.0

    def test_latex_components_not_double_counted(self):
        # \frac{2}{3}: the "2" and "3" inside braces should not be extracted as
        # plain numbers, so NumericMatch(2) should NOT match \frac{2}{3}
        assert NumericMatch(2.0)("\\frac{2}{3}") == 0.0

    def test_fraction_components_not_double_counted(self):
        # "2/3": neither "2" nor "3" should be extracted as standalone plain numbers
        assert NumericMatch(2.0)("the fraction 2/3 is the answer") == 0.0
        assert NumericMatch(3.0)("the fraction 2/3 is the answer") == 0.0

    def test_percentage_component_not_double_counted(self):
        # "66.7%": "66.7" should not also appear as a plain number
        assert NumericMatch(66.7)("the answer is 66.7%") == 0.0


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

    def test_integer_match(self):
        g = grader_from_config({"type": "IntegerMatch", "args": {"value": 4}})
        assert isinstance(g, IntegerMatch)
        assert g("four losses") == 1.0
        assert g("fourteen") == 0.0

    def test_numeric_match(self):
        g = grader_from_config({"type": "NumericMatch", "args": {"value": "2/7", "tolerance": 1e-3}})
        assert isinstance(g, NumericMatch)
        assert g("0.286") == 1.0
        assert g("0.5") == 0.0

    def test_no_args_key(self):
        g = grader_from_config({"type": "StrictStringInclusion", "args": {"substring": "x"}})
        assert isinstance(g, StrictStringInclusion)
