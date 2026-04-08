import argparse
import json
import os


def load_all_results(output_dir: str) -> list[dict]:
    """Load all results.json files from subdirectories of output_dir."""
    results = []
    for entry in sorted(os.scandir(output_dir), key=lambda e: e.name):
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


def render_markdown(ranked: list[dict], output_path: str) -> None:
    lines = [
        "# Model Rankings",
        "",
        "| Rank | Model | Mean Score | Problems | Total Cost | Total Tokens | Avg Searches |",
        "|------|-------|-----------|----------|-----------|-------------|-------------|",
    ]

    for rank, r in enumerate(ranked, start=1):
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
    ranked = rank_results(results)
    render_markdown(ranked, config["rankings"])
    print(f"Ranked {len(ranked)} models → {config['rankings']}")


if __name__ == "__main__":
    main()
