import json
import os
import sys
import tempfile

import pytest
from unittest.mock import patch, MagicMock

# Add repo root to path so we can import eval
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import (
    extract_short_name,
    load_model_list,
    load_problems,
    extract_text_from_response,
    count_search_calls,
    run_model_eval,
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)


class TestExtractShortName:
    def test_no_provider(self):
        assert extract_short_name("openai/gpt-5.4") == "gpt-5.4"

    def test_with_provider(self):
        assert extract_short_name("openai/gpt-5.4,prov") == "gpt-5.4"

    def test_complex_provider(self):
        assert extract_short_name("moonshotai/kimi-k2.5,moonshotai/int4") == "kimi-k2.5"

    def test_no_slash(self):
        assert extract_short_name("mymodel") == "mymodel"

    def test_free_suffix(self):
        assert extract_short_name("stepfun/step-3.5-flash:free") == "step-3.5-flash:free"


class TestLoadModelList:
    def test_loads_tiny_model_list(self):
        path = os.path.join(TESTS_DIR, "tiny_model_list.txt")
        models = load_model_list(path)
        assert len(models) == 3

    def test_first_model_no_provider(self):
        path = os.path.join(TESTS_DIR, "tiny_model_list.txt")
        models = load_model_list(path)
        assert models[0]["model"] == "x-ai/grok-4.1-fast"
        assert models[0]["provider"] is None
        assert models[0]["short_name"] == "grok-4.1-fast"

    def test_model_with_provider(self):
        path = os.path.join(TESTS_DIR, "tiny_model_list.txt")
        models = load_model_list(path)
        assert models[1]["model"] == "minimax/minimax-m2.5"
        assert models[1]["provider"] == "minimax/highspeed"
        assert models[1]["short_name"] == "minimax-m2.5"

    def test_skips_blank_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("openai/gpt-4\n\nopenai/gpt-3.5\n\n")
            tmp = f.name
        try:
            models = load_model_list(tmp)
            assert len(models) == 2
        finally:
            os.unlink(tmp)


class TestLoadProblems:
    def test_loads_tiny_problems(self):
        path = os.path.join(TESTS_DIR, "tiny_problems.jsonl")
        problems = load_problems(path)
        assert len(problems) == 2

    def test_problem_structure(self):
        path = os.path.join(TESTS_DIR, "tiny_problems.jsonl")
        problems = load_problems(path)
        p = problems[0]
        assert "problem" in p
        assert "reference" in p
        assert "grader" in p

    def test_carabao_cup_problem(self):
        path = os.path.join(TESTS_DIR, "tiny_problems.jsonl")
        problems = load_problems(path)
        assert "Carabao Cup" in problems[0]["problem"]
        assert problems[0]["reference"] == "Manchester City"
        assert problems[0]["grader"]["type"] == "StrictStringInclusion"
        assert problems[0]["grader"]["args"]["substring"] == "Manchester City"

    def test_arsenal_problem(self):
        path = os.path.join(TESTS_DIR, "tiny_problems.jsonl")
        problems = load_problems(path)
        assert "Arsenal" in problems[1]["problem"]
        assert problems[1]["reference"] == "4"
        assert problems[1]["grader"]["type"] == "CompositeGrader"
        graders = problems[1]["grader"]["args"]["graders"]
        assert len(graders) == 2
        assert problems[1]["grader"]["args"]["logic"] == "OR"


class TestParseArgs:
    def test_cli_args(self):
        from eval import parse_args
        old_argv = sys.argv
        sys.argv = [
            "eval.py",
            "--problems", "tests/tiny_problems.jsonl",
            "--model_list", "tests/tiny_model_list.txt",
        ]
        try:
            config = parse_args()
            assert config["problems"] == "tests/tiny_problems.jsonl"
            assert config["model_list"] == "tests/tiny_model_list.txt"
            assert config["output"] == "./output"
            assert config["timeout"] == 120
            assert config["engine"] == "exa"
        finally:
            sys.argv = old_argv

    def test_engine_flag(self):
        from eval import parse_args
        old_argv = sys.argv
        sys.argv = [
            "eval.py",
            "--problems", "tests/tiny_problems.jsonl",
            "--model_list", "tests/tiny_model_list.txt",
            "--engine", "google",
        ]
        try:
            config = parse_args()
            assert config["engine"] == "google"
        finally:
            sys.argv = old_argv

    def test_yaml_config(self):
        import yaml
        from eval import parse_args
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "problems": "yaml_problems.jsonl",
                "model_list": "yaml_models.txt",
                "timeout": 60,
            }, f)
            yaml_path = f.name

        old_argv = sys.argv
        sys.argv = ["eval.py", "--config", yaml_path]
        try:
            config = parse_args()
            assert config["problems"] == "yaml_problems.jsonl"
            assert config["timeout"] == 60
        finally:
            sys.argv = old_argv
            os.unlink(yaml_path)

    def test_cli_overrides_yaml(self):
        import yaml
        from eval import parse_args
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "problems": "yaml_problems.jsonl",
                "model_list": "yaml_models.txt",
                "timeout": 60,
            }, f)
            yaml_path = f.name

        old_argv = sys.argv
        sys.argv = [
            "eval.py",
            "--config", yaml_path,
            "--problems", "cli_problems.jsonl",  # should override YAML
        ]
        try:
            config = parse_args()
            assert config["problems"] == "cli_problems.jsonl"   # CLI wins
            assert config["model_list"] == "yaml_models.txt"    # YAML retained
            assert config["timeout"] == 60                      # YAML retained
        finally:
            sys.argv = old_argv
            os.unlink(yaml_path)


class TestExtractTextFromResponse:
    def test_extracts_message_text(self):
        response = {
            "output": [
                {"type": "web_search_call", "queries": ["test"]},
                {"type": "message", "content": [{"type": "output_text", "text": "The answer is Au."}]},
            ]
        }
        assert extract_text_from_response(response) == "The answer is Au."

    def test_empty_output(self):
        assert extract_text_from_response({}) == ""
        assert extract_text_from_response({"output": []}) == ""

    def test_multiple_messages(self):
        response = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Part 1"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "Part 2"}]},
            ]
        }
        text = extract_text_from_response(response)
        assert "Part 1" in text
        assert "Part 2" in text

    def test_ignores_non_output_text(self):
        response = {
            "output": [
                {"type": "message", "content": [{"type": "input_text", "text": "ignored"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "visible"}]},
            ]
        }
        assert extract_text_from_response(response) == "visible"


class TestCountSearchCalls:
    def test_counts_search_calls(self):
        response = {
            "output": [
                {"type": "web_search_call"},
                {"type": "web_search_call"},
                {"type": "message", "content": []},
            ]
        }
        assert count_search_calls(response) == 2

    def test_zero_search_calls(self):
        response = {"output": [{"type": "message", "content": []}]}
        assert count_search_calls(response) == 0

    def test_empty_response(self):
        assert count_search_calls({}) == 0


class TestResumeLogic:
    def test_skips_successful_problems(self):
        problems = [
            {"problem": "Q1", "grader": {"type": "StrictStringInclusion", "args": {"substring": "A"}}},
        ]
        model_entry = {"model": "test/model", "provider": None, "short_name": "model"}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(os.path.join(model_dir, "responses"))
            os.makedirs(os.path.join(model_dir, "interactions"))

            existing = {
                "model": "test/model",
                "problems": {
                    "000": {
                        "problem": "Q1",
                        "score": 1.0,
                        "search_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "status": "success",
                    }
                },
                "aggregate": {},
            }
            with open(os.path.join(model_dir, "results.json"), "w") as f:
                json.dump(existing, f)

            config = {"output": tmpdir, "timeout": 10, "max_results": 5}
            with patch("eval.requests.post") as mock_post:
                run_model_eval(model_entry, problems, config)
                mock_post.assert_not_called()

    def test_does_not_skip_failed_problems(self):
        problems = [
            {"problem": "Q1", "grader": {"type": "StrictStringInclusion", "args": {"substring": "A"}}},
        ]
        model_entry = {"model": "test/model", "provider": None, "short_name": "model"}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(os.path.join(model_dir, "responses"))
            os.makedirs(os.path.join(model_dir, "interactions"))

            existing = {
                "model": "test/model",
                "problems": {
                    "000": {
                        "problem": "Q1",
                        "score": None,
                        "search_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "status": "api_error",
                    }
                },
                "aggregate": {},
            }
            with open(os.path.join(model_dir, "results.json"), "w") as f:
                json.dump(existing, f)

            config = {"output": tmpdir, "timeout": 10, "max_results": 5, "api_key": "test-key"}
            canned = {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "A is correct"}]}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            mock_resp = MagicMock()
            mock_resp.json.return_value = canned
            mock_resp.raise_for_status = MagicMock()

            with patch("eval.requests.post", return_value=mock_resp):
                results = run_model_eval(model_entry, problems, config)

            assert results["problems"]["000"]["status"] == "success"


class TestIntegration:
    def test_run_model_eval_carabao_cup_problem(self):
        problems_path = os.path.join(TESTS_DIR, "tiny_problems.jsonl")
        problems = load_problems(problems_path)[:1]  # just Carabao Cup problem

        model_entry = {"model": "test/model", "provider": None, "short_name": "test-model"}

        canned_response = {
            "output": [
                {"type": "web_search_call", "queries": ["2025-2026 Carabao Cup winner"]},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Manchester City won the 2025-26 Carabao Cup."}],
                },
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "output": tmpdir,
                "timeout": 10,
                "max_results": 5,
                "api_key": "test-key",
            }

            mock_resp = MagicMock()
            mock_resp.json.return_value = canned_response
            mock_resp.raise_for_status = MagicMock()

            with patch("eval.requests.post", return_value=mock_resp):
                results = run_model_eval(model_entry, problems, config)

            assert results["problems"]["000"]["status"] == "success"
            assert results["problems"]["000"]["score"] == 1.0
            assert results["problems"]["000"]["search_calls"] == 1
            assert results["problems"]["000"]["input_tokens"] == 100
            assert results["problems"]["000"]["output_tokens"] == 50

            # Verify results.json was written
            results_path = os.path.join(tmpdir, "test-model", "results.json")
            assert os.path.exists(results_path)
            with open(results_path) as f:
                saved = json.load(f)
            assert saved["problems"]["000"]["score"] == 1.0
            assert saved["aggregate"]["mean_score"] == 1.0
            assert saved["aggregate"]["num_scored"] == 1

            # Verify raw response and interaction files were written
            resp_file = os.path.join(tmpdir, "test-model", "responses", "problem_000.json")
            assert os.path.exists(resp_file)
            inter_file = os.path.join(tmpdir, "test-model", "interactions", "problem_000.txt")
            assert os.path.exists(inter_file)

    def test_engine_passed_to_api(self):
        problems = load_problems(os.path.join(TESTS_DIR, "tiny_problems.jsonl"))[:1]
        model_entry = {"model": "test/model", "provider": None, "short_name": "engine-test"}

        canned = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Au"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = canned
        mock_resp.raise_for_status = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"output": tmpdir, "timeout": 10, "max_results": 5,
                      "api_key": "test-key", "engine": "google"}
            with patch("eval.requests.post", return_value=mock_resp) as mock_post:
                run_model_eval(model_entry, problems, config)
            plugin = mock_post.call_args[1]["json"]["plugins"][0]
            assert plugin["engine"] == "google"

    def test_run_model_eval_timeout(self):
        import requests as req_lib

        problems = [
            {"problem": "Q1", "grader": {"type": "StrictStringInclusion", "args": {"substring": "A"}}},
        ]
        model_entry = {"model": "test/model", "provider": None, "short_name": "timeout-model"}

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"output": tmpdir, "timeout": 1, "max_results": 5, "api_key": "test-key"}

            with patch("eval.requests.post", side_effect=req_lib.Timeout("timed out")):
                results = run_model_eval(model_entry, problems, config)

            assert results["problems"]["000"]["status"] == "timeout"
            assert results["problems"]["000"]["score"] is None
