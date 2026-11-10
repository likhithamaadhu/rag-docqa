"""Parse citation markers out of LLM answers. Pure, dependency-free."""
import re

# Models don't always follow the requested format exactly: besides [1] we see
# full-width brackets 【1】, ranges/lists like [1, 3] or [2-4], and "†" suffixes
# like 【1†source】. Parse defensively instead of trusting the format.
_CITATION_RE = re.compile(r"[\[【]\s*([\d\s,–\-]+?)\s*(?:†[^\]】]*)?[\]】]")


def cited_numbers(answer: str, n_passages: int) -> list[int]:
    """Which passage numbers the model actually cited, in order, deduplicated."""
    seen: list[int] = []
    for group in _CITATION_RE.findall(answer):
        for part in re.split(r"\s*,\s*", group.strip()):
            bounds = re.split(r"\s*[–\-]\s*", part)
            if not all(b.isdigit() for b in bounds):
                continue
            lo, hi = int(bounds[0]), int(bounds[-1])
            for i in range(lo, min(hi, n_passages) + 1):
                if i >= 1 and i not in seen:
                    seen.append(i)
    return seen
