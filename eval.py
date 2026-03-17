import argparse
import json
import os
import statistics
import sys

import requests
import yaml

from graders import grader_from_config


def extract_short_name(model_str: str) -> str:
    """'openai/gpt-5.4,prov' -> 'gpt-5.4'"""
    model_part = model_str.split(",", 1)[0] if "," in model_str else model_str
    return model_part.split("/", 1)[1] if "/" in model_part else model_part


def load_model_list(path) -> list[dict]:
    """Returns [{"model": "openai/gpt-5.4", "provider": None, "short_name": "gpt-5.4"}]"""
    models = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            provider = None
            if "," in line:
                model_part, provider = line.split(",", 1)
            else:
                model_part = line
            short_name = extract_short_name(line)
            models.append({"model": model_part, "provider": provider, "short_name": short_name})
    return models


def load_problems(path) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Nano Agentic Search Eval")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--problems", default=None, help="Path to problems JSONL file")
    parser.add_argument("--model_list", default=None, help="Path to model list file")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--timeout", type=int, default=None, help="API timeout in seconds")
    parser.add_argument("--max_results", type=int, default=None, help="Max web search results")
    parser.add_argument("--api_key", default=None, help="OpenRouter API key")
    parser.add_argument("--engine", default=None, help="Web search engine (default: exa)")

    args = parser.parse_args()

    config = {
        "output": "./output",
        "timeout": 120,
        "max_results": 5,
        "engine": "exa",
    }

    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)

    # CLI overrides YAML
    if args.problems is not None:
        config["problems"] = args.problems
    if args.model_list is not None:
        config["model_list"] = args.model_list
    if args.output is not None:
        config["output"] = args.output
    if args.timeout is not None:
        config["timeout"] = args.timeout
    if args.max_results is not None:
        config["max_results"] = args.max_results
    if args.api_key is not None:
        config["api_key"] = args.api_key
    if args.engine is not None:
        config["engine"] = args.engine

    if "problems" not in config:
        parser.error("--problems is required (or set in YAML config)")
    if "model_list" not in config:
        parser.error("--model_list is required (or set in YAML config)")

    return config


def call_responses_api(model_entry: dict, prompt: str, config: dict) -> dict:
    api_key = config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    max_results = config.get("max_results", 5)
    timeout = config.get("timeout", 120)
    engine = config.get("engine", "exa")

    body = {
        "model": model_entry["model"],
        "input": prompt,
        "plugins": [{"id": "web", "max_results": max_results, "engine": engine}],
    }

    if model_entry.get("provider"):
        body["provider"] = {"order": [model_entry["provider"]]}

    resp = requests.post(
        "https://openrouter.ai/api/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def extract_text_from_response(response: dict) -> str:
    texts = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))
    return "\n".join(texts)


def count_search_calls(response: dict) -> int:
    return sum(1 for item in response.get("output", []) if item.get("type") == "web_search_call")


def format_interaction_text(problem: str, response: dict) -> str:
    lines = [f"PROBLEM:\n{problem}\n"]
    for item in response.get("output", []):
        t = item.get("type", "")
        if t == "web_search_call":
            queries = item.get("queries", [])
            lines.append(f"[WEB SEARCH: {', '.join(queries)}]")
        elif t == "message":
            text = extract_text_from_response({"output": [item]})
            lines.append(f"\nRESPONSE:\n{text}")
    return "\n".join(lines)


def compute_aggregates(results: dict) -> dict:
    problems = results.get("problems", {})
    scored = [
        p for p in problems.values()
        if p.get("status") == "success" and p.get("score") is not None
    ]

    scores = [p["score"] for p in scored]
    search_calls = [p.get("search_calls", 0) for p in scored]
    input_tokens = [p.get("input_tokens", 0) for p in scored]
    output_tokens = [p.get("output_tokens", 0) for p in scored]

    return {
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "total_problems": len(problems),
        "num_scored": len(scored),
        "mean_search_calls": statistics.mean(search_calls) if search_calls else 0.0,
        "mean_input_tokens": statistics.mean(input_tokens) if input_tokens else 0.0,
        "mean_output_tokens": statistics.mean(output_tokens) if output_tokens else 0.0,
    }


def _write_results(path: str, results: dict) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def run_model_eval(model_entry: dict, problems: list, config: dict) -> dict:
    output_dir = config["output"]
    short_name = model_entry["short_name"]

    model_dir = os.path.join(output_dir, short_name)
    responses_dir = os.path.join(model_dir, "responses")
    interactions_dir = os.path.join(model_dir, "interactions")
    os.makedirs(responses_dir, exist_ok=True)
    os.makedirs(interactions_dir, exist_ok=True)

    results_path = os.path.join(model_dir, "results.json")

    # Load existing results for resume
    existing_problems = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
            existing_problems = data.get("problems", {})

    results = {
        "model": model_entry["model"],
        "problems": dict(existing_problems),
        "aggregate": {},
    }

    api_key = config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")

    for idx, problem in enumerate(problems):
        prob_id = f"{idx:03d}"

        # Resume: skip already successful problems
        if results["problems"].get(prob_id, {}).get("status") == "success":
            continue

        print(f"  Problem {prob_id}: {problem['problem'][:60]}...")

        # Call API
        try:
            raw_response = call_responses_api(model_entry, problem["problem"], config)
        except requests.Timeout:
            results["problems"][prob_id] = {
                "problem": problem["problem"],
                "score": None,
                "search_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "status": "timeout",
            }
            _write_results(results_path, results)
            continue
        except Exception as e:
            print(f"    API error: {e}", file=sys.stderr)
            results["problems"][prob_id] = {
                "problem": problem["problem"],
                "score": None,
                "search_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "status": "api_error",
            }
            _write_results(results_path, results)
            continue

        # Save raw response
        resp_path = os.path.join(responses_dir, f"problem_{prob_id}.json")
        with open(resp_path, "w") as f:
            json.dump(raw_response, f, indent=2)

        # Save interaction text
        interaction_text = format_interaction_text(problem["problem"], raw_response)
        inter_path = os.path.join(interactions_dir, f"problem_{prob_id}.txt")
        with open(inter_path, "w") as f:
            f.write(interaction_text)

        # Grade
        response_text = extract_text_from_response(raw_response)
        try:
            grader = grader_from_config(problem["grader"], api_key=api_key)
            score = grader(response_text)
        except Exception as e:
            print(f"    Grader error: {e}", file=sys.stderr)
            usage = raw_response.get("usage", {})
            results["problems"][prob_id] = {
                "problem": problem["problem"],
                "score": None,
                "search_calls": count_search_calls(raw_response),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "status": "grader_error",
            }
            _write_results(results_path, results)
            continue

        usage = raw_response.get("usage", {})
        results["problems"][prob_id] = {
            "problem": problem["problem"],
            "score": score,
            "search_calls": count_search_calls(raw_response),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "status": "success",
        }
        _write_results(results_path, results)

    results["aggregate"] = compute_aggregates(results)
    _write_results(results_path, results)
    return results


def main():
    config = parse_args()
    problems = load_problems(config["problems"])
    models = load_model_list(config["model_list"])

    print(f"Running eval on {len(problems)} problems with {len(models)} models")

    for model_entry in models:
        print(f"\nEvaluating {model_entry['model']}...")
        results = run_model_eval(model_entry, problems, config)
        agg = results.get("aggregate", {})
        print(
            f"  Mean score: {agg.get('mean_score', 0):.3f} "
            f"({agg.get('num_scored', 0)}/{agg.get('total_problems', 0)} problems)"
        )


if __name__ == "__main__":
    main()
