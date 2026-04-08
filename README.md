# nano_agentic_search_eval

A nano-scale LLM evaluation framework for agentic web search. Evaluates models that use web search (via [OpenRouter's Responses API](https://openrouter.ai/docs/responses)) against a set of problems with expected answers.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --extra dev
```

This creates a `.venv` and installs all dependencies including `pytest`.

## Running an Evaluation

```bash
OPENROUTER_API_KEY=xxx uv run python eval.py \
  --problems tests/tiny_problems.jsonl \
  --model_list tests/tiny_model_list.txt \
  --output ./output
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | YAML config file (CLI flags override YAML values) |
| `--problems` | required | Path to problems JSONL file |
| `--model_list` | required | Path to model list file |
| `--output` | `./output` | Output directory |
| `--timeout` | `120` | API timeout in seconds |
| `--max_results` | `5` | Max web search results per query |
| `--engine` | `exa` | Web search engine (`exa`, `google`, etc.) |
| `--max_trials` | `5` | Max API attempts per problem on error before giving up |
| `--api_key` | env `OPENROUTER_API_KEY` | OpenRouter API key |

### YAML Config

```yaml
problems: problems.jsonl
model_list: model_list.txt
output: ./output
timeout: 120
max_results: 5
max_trials: 5
```

```bash
uv run python eval.py --config config.yaml
```

## Model List Format

One model per line. Optional provider suffix after a comma:

```
openai/gpt-5.4
z-ai/glm-5,z-ai
moonshotai/kimi-k2.5,moonshotai/int4
stepfun/step-3.5-flash:free
```

## Problems Format (JSONL)

Each line is a JSON object:

```json
{"problem": "Which team won the 2025-2026 Carabao Cup?", "reference": "Manchester City", "rationale": "Manchester City won the 2025-26 Carabao Cup final.", "grader": {"type": "StrictStringInclusion", "args": {"substring": "Manchester City", "case_sensitive": true}}}
```

Fields:
- `problem` — the question to ask the model
- `reference` — expected answer
- `rationale` — optional explanation
- `grader` — grader config (see below)

## Problem Editor

Open `problem_editor.html` in any browser — no server required. Supports importing and exporting JSONL files.

## Grader Types

All graders are callable: `grader(response: str) -> float` returning a score in `[0.0, 1.0]`.

| Type | Args | Description |
|------|------|-------------|
| `StrictStringInclusion` | `substring`, `case_sensitive` | 1.0 if substring found in response |
| `IntegerMatch` | `value` | Matches an integer in digit or English word form (0–99); blocks "fourteen", "twenty-four", "fourth" when matching 4 |
| `NumericMatch` | `value`, `tolerance` | Extracts and compares numbers in decimal, fraction (`2/3`), LaTeX (`\frac{2}{3}`), or percentage form |
| `SoftOverlap` | `reference`, `similarity_threshold` | Edit-distance based partial match |
| `RegexMatch` | `pattern`, `flags` | 1.0 if regex matches response |
| `LLMGrader` | `model`, `prompt`, `reference` | Calls OpenRouter chat completions to score |
| `CompositeGrader` | `graders`, `logic`, `partial_credit` | Combines multiple graders (no LLMGrader) |
| `CustomGrader` | `func` | Python callable, code-only |

### IntegerMatch

Matches an integer answer in both digit and English word form (0–99):

```json
{"type": "IntegerMatch", "args": {"value": 4}}
```

Matches: `"4 finals"`, `"four consecutive"`, `"Four"` — but not `"fourteen"`, `"twenty-four"`, or `"fourth"`.

### NumericMatch

Matches numeric answers in any common form. Default tolerance `1e-9` (exact); set higher for rounded answers:

```json
{"type": "NumericMatch", "args": {"value": "2/3", "tolerance": 1e-3}}
```

Matches: `"0.667"`, `"66.7%"`, `"\\frac{2}{3}"`, `"2/3"` — all within tolerance of 2/3.

### CompositeGrader Logic

| `logic` | `partial_credit` | Behavior |
|---------|-----------------|----------|
| `AND` | `false` | 1.0 only if all scores > 0 |
| `AND` | `true` | mean of all scores |
| `OR` | `false` | 1.0 if any score > 0 |
| `OR` | `true` | max of all scores |

## Output Structure

```
{output}/
  {short_name}/
    responses/problem_000.json     # raw API response
    interactions/problem_000.txt   # human-readable transcript
    results.json                   # scores + aggregates
```

### results.json Schema

```json
{
  "model": "openai/gpt-5.4",
  "problems": {
    "000": {
      "problem": "...",
      "score": 0.85,
      "search_calls": 2,
      "input_tokens": 512,
      "cached_tokens": 0,
      "output_tokens": 256,
      "cost": 0.0206,
      "status": "success"
    }
  },
  "aggregate": {
    "mean_score": 0.72,
    "std_score": 0.15,
    "total_problems": 10,
    "num_scored": 10,
    "mean_search_calls": 1.8,
    "total_input_tokens": 4800,
    "mean_input_tokens": 480,
    "total_cached_tokens": 120,
    "mean_cached_tokens": 12,
    "total_output_tokens": 2200,
    "mean_output_tokens": 220,
    "total_cost": 0.42,
    "mean_cost": 0.042,
    "max_cost": 0.12
  }
}
```

Problem statuses: `success`, `timeout`, `api_error`, `grader_error`.

**Resume logic:** The runner skips problems with `status: "success"` and writes `results.json` after each problem for crash-safety.

## Ranking Models

After running evaluations, compile a ranking across all models in the output directory:

```bash
uv run python rank.py --results_dir ./output
```

This writes `./output/rankings.md` (Markdown table sorted by score, with cost and token breakdowns). Override the output path with `--rankings`:

```bash
uv run python rank.py --results_dir ./output --rankings ./output/rankings.md
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--results_dir` | `./output` | Directory containing per-model result subdirectories |
| `--rankings` | `{results_dir}/rankings.md` | Path to write the rankings Markdown file |

### Rankings Format

| Rank | Model | Mean Score | Problems | Total Cost | Total Tokens | Avg Searches |
|------|-------|-----------|----------|-----------|-------------|-------------|
| 1 | `openai/gpt-5.4` | 0.850 ± 0.120 | 10/10 | $0.4200 | 7,000 | 1.8 |
| 2 | `x-ai/grok-4` | 0.720 ± 0.200 | 9/10 | $0.2100 | 5,500 | 2.1 |

Ties in mean score are broken by total cost (ascending), then total tokens (ascending). Models without cost data rank below models with cost data.

## Running Tests

```bash
uv run pytest tests/ -v
```
