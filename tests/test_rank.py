import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from rank import load_all_results, rank_results, rank_problems, render_markdown, _total_tokens, _truncate


def make_results(model, mean_score, std_score=0.0, num_scored=2, total_problems=2,
                 total_cost=None, total_input=0, total_output=0, mean_searches=1.0,
                 problems=None):
    return {
        "model": model,
        "problems": problems or {},
        "aggregate": {
            "mean_score": mean_score,
            "std_score": std_score,
            "num_scored": num_scored,
            "total_problems": total_problems,
            "total_cost": total_cost,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "mean_search_calls": mean_searches,
        },
    }


def make_problem_entry(text, score):
    return {"problem": text, "score": score, "status": "success"}


def write_results(tmpdir, short_name, data):
    model_dir = os.path.join(tmpdir, short_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(data, f)


class TestParseArgs:
    def test_default_rankings_path(self):
        import sys
        from rank import parse_args
        old_argv = sys.argv
        sys.argv = ["rank.py", "--results_dir", "./my_output"]
        try:
            config = parse_args()
            assert config["results_dir"] == "./my_output"
            assert config["rankings"] == "./my_output/rankings.md"
        finally:
            sys.argv = old_argv

    def test_explicit_rankings_path(self):
        import sys
        from rank import parse_args
        old_argv = sys.argv
        sys.argv = ["rank.py", "--results_dir", "./my_output", "--rankings", "/tmp/custom.md"]
        try:
            config = parse_args()
            assert config["rankings"] == "/tmp/custom.md"
        finally:
            sys.argv = old_argv


class TestLoadAllResults:
    def test_loads_subdirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(tmpdir, "model-a", make_results("a/model-a", 0.8))
            write_results(tmpdir, "model-b", make_results("b/model-b", 0.6))
            results = load_all_results(tmpdir)
        assert len(results) == 2
        models = {r["model"] for r in results}
        assert models == {"a/model-a", "b/model-b"}

    def test_skips_dirs_without_results_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "empty-model"))
            write_results(tmpdir, "model-a", make_results("a/model-a", 0.9))
            results = load_all_results(tmpdir)
        assert len(results) == 1

    def test_skips_files_in_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "stray.json"), "w") as f:
                f.write("{}")
            write_results(tmpdir, "model-a", make_results("a/model-a", 0.5))
            results = load_all_results(tmpdir)
        assert len(results) == 1

    def test_empty_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = load_all_results(tmpdir)
        assert results == []

    def test_alphabetical_load_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(tmpdir, "zzz", make_results("z/zzz", 0.1))
            write_results(tmpdir, "aaa", make_results("a/aaa", 0.9))
            results = load_all_results(tmpdir)
        assert results[0]["model"] == "a/aaa"
        assert results[1]["model"] == "z/zzz"


class TestTotalTokens:
    def test_sums_input_and_output(self):
        agg = {"total_input_tokens": 1000, "total_output_tokens": 200}
        assert _total_tokens(agg) == 1200

    def test_missing_fields_default_zero(self):
        assert _total_tokens({}) == 0


class TestTruncate:
    def test_short_string_unchanged(self):
        assert _truncate("hello", 60) == "hello"

    def test_exact_length_unchanged(self):
        s = "x" * 60
        assert _truncate(s, 60) == s

    def test_long_string_truncated(self):
        s = "x" * 61
        result = _truncate(s, 60)
        assert result.endswith("…")
        assert len(result) == 61  # 60 chars + ellipsis


class TestRankResults:
    def test_higher_score_ranked_first(self):
        r = [
            make_results("a/low", 0.5),
            make_results("b/high", 0.9),
        ]
        ranked = rank_results(r)
        assert ranked[0]["model"] == "b/high"
        assert ranked[1]["model"] == "a/low"

    def test_tie_broken_by_cost(self):
        r = [
            make_results("a/expensive", 0.8, total_cost=1.0),
            make_results("b/cheap", 0.8, total_cost=0.1),
        ]
        ranked = rank_results(r)
        assert ranked[0]["model"] == "b/cheap"

    def test_tie_broken_by_tokens_when_cost_equal(self):
        r = [
            make_results("a/heavy", 0.8, total_cost=0.5, total_input=5000, total_output=500),
            make_results("b/light", 0.8, total_cost=0.5, total_input=1000, total_output=100),
        ]
        ranked = rank_results(r)
        assert ranked[0]["model"] == "b/light"

    def test_none_cost_ranked_last(self):
        r = [
            make_results("a/no-cost", 0.8, total_cost=None),
            make_results("b/has-cost", 0.8, total_cost=0.01),
        ]
        ranked = rank_results(r)
        assert ranked[0]["model"] == "b/has-cost"
        assert ranked[1]["model"] == "a/no-cost"

    def test_single_model(self):
        r = [make_results("a/only", 0.7)]
        ranked = rank_results(r)
        assert len(ranked) == 1


class TestRankProblems:
    def _results_with_problems(self, model, prob_scores: dict):
        """prob_scores: {prob_id: score}"""
        problems = {
            pid: make_problem_entry(f"Question {pid}", score)
            for pid, score in prob_scores.items()
        }
        return make_results(model, 0.5, problems=problems)

    def test_avg_score_computed(self):
        results = [
            self._results_with_problems("a/m1", {"000": 1.0, "001": 0.5}),
            self._results_with_problems("b/m2", {"000": 0.5, "001": 0.5}),
        ]
        ranked = rank_problems(results)
        by_id = {p["id"]: p for p in ranked}
        assert by_id["000"]["avg_score"] == pytest.approx(0.75)
        assert by_id["001"]["avg_score"] == pytest.approx(0.5)

    def test_higher_avg_score_ranked_first(self):
        results = [
            self._results_with_problems("a/m1", {"000": 0.5, "001": 1.0}),
            self._results_with_problems("b/m2", {"000": 0.5, "001": 1.0}),
        ]
        ranked = rank_problems(results)
        assert ranked[0]["id"] == "001"
        assert ranked[1]["id"] == "000"

    def test_tie_broken_by_fully_correct(self):
        results = [
            self._results_with_problems("a/m1", {"000": 1.0, "001": 0.5}),
            self._results_with_problems("b/m2", {"000": 0.5, "001": 1.0}),
        ]
        ranked = rank_problems(results)
        # Both have avg 0.75, but "000" has 1 fully correct, "001" has 1 too — tie on fully_correct
        # then by id: "000" < "001"
        assert ranked[0]["id"] == "000"

    def test_tie_broken_by_id_when_fully_correct_equal(self):
        results = [
            self._results_with_problems("a/m1", {"001": 0.5, "000": 0.5}),
        ]
        ranked = rank_problems(results)
        assert ranked[0]["id"] == "000"
        assert ranked[1]["id"] == "001"

    def test_fully_correct_count(self):
        results = [
            self._results_with_problems("a/m1", {"000": 1.0}),
            self._results_with_problems("b/m2", {"000": 1.0}),
            self._results_with_problems("c/m3", {"000": 0.5}),
        ]
        ranked = rank_problems(results)
        assert ranked[0]["fully_correct"] == 2
        assert ranked[0]["num_attempted"] == 3

    def test_skips_none_scores(self):
        results = [{
            "model": "a/m",
            "problems": {
                "000": {"problem": "Q0", "score": None, "status": "api_error"},
                "001": {"problem": "Q1", "score": 1.0, "status": "success"},
            },
            "aggregate": {},
        }]
        ranked = rank_problems(results)
        assert len(ranked) == 1
        assert ranked[0]["id"] == "001"

    def test_empty_results(self):
        assert rank_problems([]) == []

    def test_problem_text_preserved(self):
        results = [self._results_with_problems("a/m1", {"000": 0.8})]
        ranked = rank_problems(results)
        assert ranked[0]["problem"] == "Question 000"


class TestRenderMarkdown:
    def _render(self, ranked_models, ranked_problems=None):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
        render_markdown(ranked_models, ranked_problems or [], path)
        content = open(path).read()
        os.unlink(path)
        return content

    def test_output_file_created(self):
        ranked = [make_results("a/model", 0.8, total_cost=0.5, total_input=1000, total_output=200)]
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            render_markdown(ranked, [], path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_contains_model_name(self):
        ranked = [make_results("openai/gpt-test", 0.75, total_cost=0.42,
                               total_input=4800, total_output=2200)]
        content = self._render(ranked)
        assert "openai/gpt-test" in content
        assert "0.750" in content
        assert "$0.4200" in content
        assert "7,000" in content  # 4800 + 2200

    def test_model_rank_column(self):
        ranked = [make_results("a/first", 0.9), make_results("b/second", 0.5)]
        content = self._render(ranked)
        lines = content.splitlines()
        model_rows = [l for l in lines if l.startswith("| ") and "Rank" not in l
                      and "---" not in l and "Problem Rankings" not in l
                      and "Problem |" not in l and "Question" not in l]
        assert model_rows[0].startswith("| 1 |")
        assert model_rows[1].startswith("| 2 |")

    def test_no_cost_renders_dash(self):
        ranked = [make_results("a/model", 0.5, total_cost=None)]
        assert "| — |" in self._render(ranked)

    def test_zero_tokens_renders_dash(self):
        ranked = [make_results("a/model", 0.5, total_input=0, total_output=0)]
        assert "| — |" in self._render(ranked)

    def test_problem_rankings_section_present(self):
        problems = [{"id": "000", "problem": "What is 2+2?", "avg_score": 0.9,
                     "fully_correct": 2, "num_attempted": 3}]
        content = self._render([make_results("a/m", 0.9)], problems)
        assert "# Problem Rankings" in content
        assert "What is 2+2?" in content
        assert "0.900" in content

    def test_problem_text_truncated(self):
        long_q = "A" * 70
        problems = [{"id": "000", "problem": long_q, "avg_score": 1.0,
                     "fully_correct": 1, "num_attempted": 1}]
        content = self._render([make_results("a/m", 1.0)], problems)
        assert long_q not in content
        assert "…" in content

    def test_problem_rank_column(self):
        problems = [
            {"id": "000", "problem": "Q0", "avg_score": 1.0, "fully_correct": 2, "num_attempted": 2},
            {"id": "001", "problem": "Q1", "avg_score": 0.5, "fully_correct": 0, "num_attempted": 2},
        ]
        content = self._render([make_results("a/m", 0.75)], problems)
        lines = content.splitlines()
        prob_rows = [l for l in lines if "| Q" in l]
        assert prob_rows[0].startswith("| 1 |")
        assert prob_rows[1].startswith("| 2 |")
