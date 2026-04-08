import argparse
import json
import os
import statistics


def load_all_results(results_dir: str) -> list[dict]:
    """Load all results.json files from subdirectories of results_dir."""
    results = []
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        results_path = os.path.join(entry.path, "results.json")
        if not os.path.exists(results_path):
            continue
        with open(results_path) as f:
            data = json.load(f)
        results.append(data)
    return results


def _total_tokens(agg: dict) -> int:
    """Sum of input + output tokens (cached is a subset of input, not double-counted)."""
    return agg.get("total_input_tokens", 0) + agg.get("total_output_tokens", 0)


def rank_results(results: list[dict]) -> list[dict]:
    """Sort by mean_score desc, total_cost asc (None last), total_tokens asc."""
    def sort_key(r):
        agg = r.get("aggregate", {})
        score = agg.get("mean_score", 0.0)
        cost = agg.get("total_cost")
        tokens = _total_tokens(agg)
        return (-score, cost if cost is not None else float("inf"), tokens)

    return sorted(results, key=sort_key)


def rank_problems(all_results: list[dict]) -> list[dict]:
    """Aggregate per-problem stats across all model results.

    Sorted by avg_score desc, fully_correct desc, problem id asc.
    Only problems with at least one scored attempt are included.
    """
    problem_data: dict[str, dict] = {}

    for result in all_results:
        for prob_id, prob in result.get("problems", {}).items():
            score = prob.get("score")
            if score is None:
                continue
            if prob_id not in problem_data:
                problem_data[prob_id] = {
                    "id": prob_id,
                    "problem": prob.get("problem", ""),
                    "scores": [],
                    "fully_correct": 0,
                }
            problem_data[prob_id]["scores"].append(score)
            if score == 1.0:
                problem_data[prob_id]["fully_correct"] += 1

    problems = []
    for data in problem_data.values():
        scores = data["scores"]
        problems.append({
            "id": data["id"],
            "problem": data["problem"],
            "avg_score": statistics.mean(scores),
            "fully_correct": data["fully_correct"],
            "num_attempted": len(scores),
        })

    problems.sort(key=lambda p: (-p["avg_score"], -p["fully_correct"], p["id"]))
    return problems


def _truncate(text: str, max_len: int = 60) -> str:
    return text if len(text) <= max_len else text[:max_len] + "…"


def render_markdown(ranked_models: list[dict], ranked_problems: list[dict], output_path: str) -> None:
    lines = [
        "# Model Rankings",
        "",
        "| Rank | Model | Mean Score | Problems | Total Cost | Total Tokens | Avg Searches |",
        "|------|-------|-----------|----------|-----------|-------------|-------------|",
    ]

    for rank, r in enumerate(ranked_models, start=1):
        model = r.get("model", "unknown")
        agg = r.get("aggregate", {})

        mean_score = agg.get("mean_score", 0.0)
        std_score = agg.get("std_score", 0.0)
        num_scored = agg.get("num_scored", 0)
        total_problems = agg.get("total_problems", 0)

        cost = agg.get("total_cost")
        cost_str = f"${cost:.4f}" if cost is not None else "—"

        tokens = _total_tokens(agg)
        tokens_str = f"{tokens:,}" if tokens else "—"

        mean_searches = agg.get("mean_search_calls")
        searches_str = f"{mean_searches:.1f}" if mean_searches is not None else "—"

        score_str = f"{mean_score:.3f} ± {std_score:.3f}"
        problems_str = f"{num_scored}/{total_problems}"

        lines.append(
            f"| {rank} | `{model}` | {score_str} | {problems_str} "
            f"| {cost_str} | {tokens_str} | {searches_str} |"
        )

    lines += [
        "",
        "# Problem Rankings",
        "",
        "| Rank | Problem | Avg Score | Fully Correct | Models Attempted |",
        "|------|---------|-----------|--------------|-----------------|",
    ]

    for rank, p in enumerate(ranked_problems, start=1):
        problem_str = _truncate(p["problem"])
        avg_score = p["avg_score"]
        fully_correct = p["fully_correct"]
        num_attempted = p["num_attempted"]
        lines.append(
            f"| {rank} | {problem_str} | {avg_score:.3f} | {fully_correct} | {num_attempted} |"
        )

    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Rank evaluated models by score")
    parser.add_argument("--results_dir", default="./output", help="Directory containing per-model results")
    parser.add_argument("--rankings", default=None, help="Path to write rankings Markdown (default: {results_dir}/rankings.md)")
    args = parser.parse_args()
    rankings = args.rankings if args.rankings is not None else os.path.join(args.results_dir, "rankings.md")
    return {"results_dir": args.results_dir, "rankings": rankings}


def main():
    config = parse_args()
    results = load_all_results(config["results_dir"])
    if not results:
        print(f"No results found in {config['results_dir']}")
        return
    ranked_models = rank_results(results)
    ranked_problems = rank_problems(results)
    render_markdown(ranked_models, ranked_problems, config["rankings"])
    print(f"Ranked {len(ranked_models)} models, {len(ranked_problems)} problems → {config['rankings']}")


if __name__ == "__main__":
    main()
