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


_ONES = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen',
]
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']


class IntegerMatch(BaseGrader):
    """Matches an integer answer in both digit form and English word form (0–99).

    The digit pattern rejects adjacent digits (so 4 won't match 14, 24, 40).
    The word pattern rejects compound words via lookahead/lookbehind (so "four"
    won't match "fourteen", "twenty-four", or "fourth").
    """

    def __init__(self, value: int):
        self.value = value
        self._patterns = self._build_patterns(value)

    @staticmethod
    def _int_to_words(n: int) -> str | None:
        """Return English word for n (0–99), or None if out of range."""
        if n < 0 or n > 99:
            return None
        if n < 20:
            return _ONES[n]
        if n % 10 == 0:
            return _TENS[n // 10]
        return f"{_TENS[n // 10]}-{_ONES[n % 10]}"

    @staticmethod
    def _build_patterns(n: int) -> list[tuple[str, int]]:
        patterns = []
        # Digit form: the number not adjacent to other digits
        patterns.append((rf'(?<!\d){re.escape(str(n))}(?!\d)', 0))
        # English word form (0–99 only)
        word = IntegerMatch._int_to_words(n)
        if word:
            # Lookbehind blocks "twenty-four"→"four" and "fourteen"→"four"
            # Lookahead  blocks "fourth", "fourteen", "forty-four"
            patterns.append((
                rf'(?<![a-zA-Z\d\-]){re.escape(word)}(?![a-zA-Z\d\-])',
                re.IGNORECASE,
            ))
        return patterns

    def __call__(self, response: str) -> float:
        for pattern, flags in self._patterns:
            if re.search(pattern, response, flags):
                return 1.0
        return 0.0


class NumericMatch(BaseGrader):
    """Matches a numeric answer expressed in any common form: plain decimal, fraction
    (a/b), LaTeX \\frac{a}{b}, or percentage (e.g. 66.7%).

    Args:
        value: Target value as a float, int, or fraction string like ``"2/3"``.
        tolerance: Maximum absolute difference to accept. Default 1e-9 (exact).
                   Use e.g. ``1e-3`` to accept rounded decimals such as ``"0.286"``
                   when the reference is ``"2/7"``.
    """

    def __init__(self, value: float | str, tolerance: float = 1e-9):
        if isinstance(value, str) and '/' in value:
            num, den = value.split('/', 1)
            self.value = float(num) / float(den)
        else:
            self.value = float(value)
        self.tolerance = tolerance

    def __call__(self, response: str) -> float:
        for candidate in self._extract_values(response):
            if abs(candidate - self.value) <= self.tolerance:
                return 1.0
        return 0.0

    @staticmethod
    def _extract_values(text: str) -> list[float]:
        """Extract all numeric values from text in all recognised forms."""
        values = []
        # 1. LaTeX fractions: \frac{a}{b}
        for m in re.finditer(r'\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}', text):
            den = float(m.group(2))
            if den != 0:
                values.append(float(m.group(1)) / den)
        # 2. Regular fractions: a/b
        for m in re.finditer(r'(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)', text):
            den = float(m.group(2))
            if den != 0:
                values.append(float(m.group(1)) / den)
        # 3. Percentages: 66.7% → 0.667
        for m in re.finditer(r'(-?\d+(?:\.\d+)?)\s*%', text):
            values.append(float(m.group(1)) / 100.0)
        # 4. Plain numbers — skip those adjacent to /, %, or LaTeX braces to
        #    avoid double-counting fractions and percentages already extracted above
        for m in re.finditer(r'(?<![/\d\{])-?\d+(?:\.\d+)?(?![/%\d\}])', text):
            values.append(float(m.group()))
        return values


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
    elif grader_type == "IntegerMatch":
        return IntegerMatch(**args)
    elif grader_type == "NumericMatch":
        return NumericMatch(**args)
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
