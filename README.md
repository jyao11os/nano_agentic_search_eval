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
| `--api_key` | env `OPENROUTER_API_KEY` | OpenRouter API key |

### YAML Config

```yaml
problems: problems.jsonl
model_list: model_list.txt
output: ./output
timeout: 120
max_results: 5
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
{"problem": "What is the chemical symbol for gold?", "reference": "Au", "rationale": "From Latin 'aurum'", "grader": {"type": "StrictStringInclusion", "args": {"substring": "Au", "case_sensitive": true}}}
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
| `SoftOverlap` | `reference`, `similarity_threshold` | Edit-distance based partial match |
| `RegexMatch` | `pattern`, `flags` | 1.0 if regex matches response |
| `LLMGrader` | `model`, `prompt`, `reference` | Calls OpenRouter chat completions to score |
| `CompositeGrader` | `graders`, `logic`, `partial_credit` | Combines multiple graders (no LLMGrader) |
| `CustomGrader` | `func` | Python callable, code-only |

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
      "output_tokens": 256,
      "status": "success"
    }
  },
  "aggregate": {
    "mean_score": 0.72,
    "std_score": 0.15,
    "total_problems": 10,
    "num_scored": 10,
    "mean_search_calls": 1.8,
    "mean_input_tokens": 480,
    "mean_output_tokens": 220
  }
}
```

Problem statuses: `success`, `timeout`, `api_error`, `grader_error`.

**Resume logic:** The runner skips problems with `status: "success"` and writes `results.json` after each problem for crash-safety.

## Running Tests

```bash
uv run pytest tests/ -v
```
