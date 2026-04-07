import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from rank import load_all_results, rank_results, render_markdown, _total_tokens


def make_results(model, mean_score, std_score=0.0, num_scored=2, total_problems=2,
                 total_cost=None, total_input=0, total_output=0, mean_searches=1.0):
    return {
        "model": model,
        "problems": {},
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
        sys.argv = ["rank.py", "--output", "./my_output"]
        try:
            config = parse_args()
            assert config["rankings"] == "./my_output/rankings.md"
        finally:
            sys.argv = old_argv

    def test_explicit_rankings_path(self):
        import sys
        from rank import parse_args
        old_argv = sys.argv
        sys.argv = ["rank.py", "--output", "./my_output", "--rankings", "/tmp/custom.md"]
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


class TestRenderMarkdown:
    def test_output_file_created(self):
        ranked = [make_results("a/model", 0.8, total_cost=0.5, total_input=1000, total_output=200)]
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            render_markdown(ranked, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_contains_model_name(self):
        ranked = [make_results("openai/gpt-test", 0.75, total_cost=0.42,
                               total_input=4800, total_output=2200)]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".md", delete=False) as f:
            path = f.name
        try:
            render_markdown(ranked, path)
            content = open(path).read()
            assert "openai/gpt-test" in content
            assert "0.750" in content
            assert "$0.4200" in content
            assert "7,000" in content  # 4800 + 2200
        finally:
            os.unlink(path)

    def test_rank_column(self):
        ranked = [
            make_results("a/first", 0.9),
            make_results("b/second", 0.5),
        ]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".md", delete=False) as f:
            path = f.name
        try:
            render_markdown(ranked, path)
            lines = open(path).read().splitlines()
            data_lines = [l for l in lines if l.startswith("| ") and "Rank" not in l and "---" not in l]
            assert data_lines[0].startswith("| 1 |")
            assert data_lines[1].startswith("| 2 |")
        finally:
            os.unlink(path)

    def test_no_cost_renders_dash(self):
        ranked = [make_results("a/model", 0.5, total_cost=None)]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".md", delete=False) as f:
            path = f.name
        try:
            render_markdown(ranked, path)
            content = open(path).read()
            assert "| — |" in content
        finally:
            os.unlink(path)

    def test_zero_tokens_renders_dash(self):
        ranked = [make_results("a/model", 0.5, total_input=0, total_output=0)]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".md", delete=False) as f:
            path = f.name
        try:
            render_markdown(ranked, path)
            content = open(path).read()
            assert "| — |" in content
        finally:
            os.unlink(path)
