import re
import os
import requests

DEFAULT_PROMPT = """\
You are grading a model's response to a question.

Reference answer: {reference}
Model response: {response}

Does the model response correctly answer the question? Consider the reference answer.
Output a score as a float between 0.0 (completely wrong) and 1.0 (completely correct).
Output only the score as the last line of your response.
"""


class BaseGrader:
    def __call__(self, response: str) -> float:
        raise NotImplementedError


class StrictStringInclusion(BaseGrader):
    def __init__(self, substring: str, case_sensitive: bool = True):
        self.substring = substring
        self.case_sensitive = case_sensitive

    def __call__(self, response: str) -> float:
        if self.case_sensitive:
            return 1.0 if self.substring in response else 0.0
        return 1.0 if self.substring.lower() in response.lower() else 0.0


class SoftOverlap(BaseGrader):
    def __init__(self, reference: str, similarity_threshold: float = 0.2):
        self.reference = reference
        self.similarity_threshold = similarity_threshold

    def __call__(self, response: str) -> float:
        if not self.reference:
            return 0.0
        dist = self._substring_edit_distance(response, self.reference)
        score = 1.0 - dist / len(self.reference)
        return score if score >= self.similarity_threshold else 0.0

    @staticmethod
    def _substring_edit_distance(source: str, target: str) -> int:
        """Semi-global DP: find min edit distance to match target as substring of source."""
        m, n = len(source), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # dp[i][0] = 0 for all i: can start matching target at any position in source for free
        for j in range(n + 1):
            dp[0][j] = j  # no source chars consumed: must delete j target chars
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if source[i - 1] == target[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
        return min(dp[i][n] for i in range(m + 1))


class RegexMatch(BaseGrader):
    def __init__(self, pattern: str, flags: int = 0):
        self.pattern = pattern
        self.flags = flags

    def __call__(self, response: str) -> float:
        return 1.0 if re.search(self.pattern, response, self.flags) else 0.0


class LLMGrader(BaseGrader):
    def __init__(self, model: str = "openai/gpt-4.1-mini", prompt: str = DEFAULT_PROMPT,
                 reference: str = "", api_key: str = None):
        self.model = model
        self.prompt = prompt
        self.reference = reference
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

    def __call__(self, response: str) -> float:
        formatted = self.prompt
        formatted = formatted.replace("{reference}", self.reference)
        formatted = formatted.replace("{response}", response)

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": formatted}],
            },
        )
        content = resp.json()["choices"][0]["message"]["content"]
        floats = re.findall(r"\d+\.?\d*", content)
        if not floats:
            return 0.0
        score = float(floats[-1])
        return max(0.0, min(1.0, score))


class CustomGrader(BaseGrader):
    def __init__(self, func: callable):
        self.func = func

    def __call__(self, response: str) -> float:
        return self.func(response)


class CompositeGrader(BaseGrader):
    def __init__(self, graders: list, logic: str = "AND", partial_credit: bool = False):
        for g in graders:
            if isinstance(g, LLMGrader):
                raise ValueError("CompositeGrader does not support LLMGrader")
        self.graders = graders
        self.logic = logic
        self.partial_credit = partial_credit

    def __call__(self, response: str) -> float:
        scores = [g(response) for g in self.graders]
        if not scores:
            return 0.0
        if self.logic == "AND":
            if self.partial_credit:
                return sum(scores) / len(scores)
            return 1.0 if all(s > 0 for s in scores) else 0.0
        elif self.logic == "OR":
            if self.partial_credit:
                return max(scores)
            return 1.0 if any(s > 0 for s in scores) else 0.0
        raise ValueError(f"Unknown logic: {self.logic}")


def grader_from_config(config: dict, api_key: str = None) -> BaseGrader:
    """Factory: config = {"type": "StrictStringInclusion", "args": {...}}"""
    grader_type = config["type"]
    args = config.get("args", {})

    if grader_type == "StrictStringInclusion":
        return StrictStringInclusion(**args)
    elif grader_type == "SoftOverlap":
        return SoftOverlap(**args)
    elif grader_type == "RegexMatch":
        return RegexMatch(**args)
    elif grader_type == "LLMGrader":
        llm_args = dict(args)
        if api_key and "api_key" not in llm_args:
            llm_args["api_key"] = api_key
        return LLMGrader(**llm_args)
    elif grader_type == "CustomGrader":
        raise ValueError("CustomGrader cannot be created from config")
    elif grader_type == "CompositeGrader":
        sub_graders = [grader_from_config(g, api_key) for g in args.get("graders", [])]
        return CompositeGrader(
            graders=sub_graders,
            logic=args.get("logic", "AND"),
            partial_credit=args.get("partial_credit", False),
        )
    raise ValueError(f"Unknown grader type: {grader_type}")
