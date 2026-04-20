from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional

from get_mapping import get_inverse_mapping


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sanitize_extracted_candidate(value: str) -> str:
    candidate = _collapse_spaces(value).strip("`'\"")
    if "/" in candidate:
        prefix, suffix = candidate.split("/", 1)
        if ":" in prefix:
            candidate = "/" + suffix
    return candidate


def normalize_candidate_path(
    value: str,
    *,
    ignore_extension: bool = True,
    ignore_leading_slash: bool = True,
) -> str:
    normalized = value.strip().strip("`'\"")
    normalized = normalized.replace("\\", "/")
    normalized = _collapse_spaces(normalized)
    normalized = normalized.rstrip(".,;:)]}")

    if ignore_leading_slash:
        normalized = normalized.lstrip("/")

    if ignore_extension:
        normalized = re.sub(r"\.(nvn|nvm)$", "", normalized, flags=re.IGNORECASE)

    return normalized.casefold()


class SourcePathValidator:
    """Lightweight validator for script paths cited in final answers."""

    def __init__(
        self,
        *,
        ignore_extension: bool = True,
        ignore_leading_slash: bool = True,
        allow_partial_suffix_match: bool = True,
        ignore_data_extensions: bool = True,
        ignored_path_extensions: Optional[List[str]] = None,
    ):
        self.ignore_extension = ignore_extension
        self.ignore_leading_slash = ignore_leading_slash
        self.allow_partial_suffix_match = allow_partial_suffix_match
        self.ignore_data_extensions = ignore_data_extensions
        self.ignored_path_extensions = {
            ext.casefold().lstrip(".")
            for ext in (ignored_path_extensions or ["ion", "csv"])
        }

        inverse_mapping = get_inverse_mapping() or {}
        self.canonical_paths = sorted(inverse_mapping.keys())
        self.normalized_map: Dict[str, str] = {}
        for real_path in self.canonical_paths:
            normalized = normalize_candidate_path(
                real_path,
                ignore_extension=self.ignore_extension,
                ignore_leading_slash=self.ignore_leading_slash,
            )
            self.normalized_map[normalized] = real_path

    def extract_candidates(self, answer: str) -> List[str]:
        candidates: List[str] = []
        seen: set[str] = set()

        patterns = [
            r"`([^`\n]+)`",
            r"(?<![\w`])(\/?[^,\n`]*?/[^\n,`]*?\.(?:nvn|nvm|ion|csv|xlsx))\b",
            r"(?<![\w`])(\/?\d[^,\n`:;`]*?/[^,\n`]+(?:/[^,\n`]+)*)",
            r"(?<!\w)(/?\d[^,\n`]*?\.(?:nvn|nvm))(?!\w)",
        ]

        for pattern in patterns:
            for match in re.findall(pattern, answer, flags=re.IGNORECASE):
                candidate = _sanitize_extracted_candidate(match)
                if not candidate:
                    continue
                if "/" not in candidate and not re.search(r"\.(?:nvn|nvm)$", candidate, flags=re.IGNORECASE):
                    continue
                if candidate.count("/") == 1 and not candidate.lstrip("/").startswith(tuple(str(i) for i in range(10))) and not re.search(r"\.(?:nvn|nvm|ion|csv|xlsx)$", candidate, flags=re.IGNORECASE):
                    continue
                normalized_key = normalize_candidate_path(
                    candidate,
                    ignore_extension=self.ignore_extension,
                    ignore_leading_slash=self.ignore_leading_slash,
                )
                if candidate and normalized_key not in seen:
                    seen.add(normalized_key)
                    candidates.append(candidate)

        return candidates

    def _should_ignore_candidate(self, raw: str) -> bool:
        if not self.ignore_data_extensions:
            return False
        cleaned = raw.strip().strip("`'\"").rstrip(".,;:)]}")
        cleaned = cleaned.replace("\\", "/")
        suffix = PurePosixPath(cleaned).suffix.casefold().lstrip(".")
        return bool(suffix) and suffix in self.ignored_path_extensions

    def validate_answer(self, answer: str) -> Dict[str, Any]:
        candidates = self.extract_candidates(answer)
        validated: List[Dict[str, str]] = []
        invalid: List[str] = []
        ignored: List[str] = []

        for raw in candidates:
            if self._should_ignore_candidate(raw):
                ignored.append(raw)
                continue

            normalized = normalize_candidate_path(
                raw,
                ignore_extension=self.ignore_extension,
                ignore_leading_slash=self.ignore_leading_slash,
            )

            exact = self.normalized_map.get(normalized)
            if exact:
                validated.append({"raw": raw, "canonical": exact, "match_type": "exact"})
                continue

            if self.allow_partial_suffix_match:
                suffix_matches = [
                    path
                    for key, path in self.normalized_map.items()
                    if key.endswith(normalized) or normalized.endswith(key)
                ]
                suffix_matches = sorted(set(suffix_matches))
                if len(suffix_matches) == 1:
                    validated.append(
                        {
                            "raw": raw,
                            "canonical": suffix_matches[0],
                            "match_type": "suffix",
                        }
                    )
                    continue

            invalid.append(raw)

        return {
            "candidates": candidates,
            "validated": validated,
            "invalid": invalid,
            "ignored": ignored,
            "has_invalid": bool(invalid),
        }


def build_validation_feedback(report: Dict[str, Any]) -> str:
    invalid_paths = report.get("invalid", [])
    ignored_paths = report.get("ignored", [])
    examples = ", ".join(item["canonical"] for item in report.get("validated", [])[:2])
    examples_line = (
        f"Examples of valid canonical paths from the project mapping: {examples}.\n"
        if examples
        else ""
    )
    ignored_line = (
        "This V1 validator intentionally ignores cited data-file paths such as .ion or .csv, "
        "because mapping.txt only tracks script paths.\n"
        if ignored_paths
        else ""
    )
    invalid_block = "\n".join(f"- {path}" for path in invalid_paths) or "- none"
    return (
        "SOURCE PATH VALIDATION CHECK FAILED.\n"
        "Rule checked: every cited script path must match a real path from mapping.txt.\n"
        "The check is tolerant to an optional leading slash, extra spaces, and .nvn/.nvm/missing extension.\n"
        "Partial suffix matches are allowed when they uniquely identify a mapped file.\n"
        f"{ignored_line}"
        f"{examples_line}"
        "Invalid or unverified cited paths detected:\n"
        f"{invalid_block}\n"
        "Please regenerate the answer and only cite paths that match the project mapping.\n"
        "If unsure, remove the path instead of inventing one."
    )


def append_validation_warning(answer: str, report: Dict[str, Any]) -> str:
    invalid_paths = report.get("invalid", [])
    if not invalid_paths:
        return answer
    warning_lines = "\n".join(f"- {path}" for path in invalid_paths)
    warning_section = (
        "\n\nPotentially invalid cited paths:\n"
        f"{warning_lines}"
    )
    if warning_section.strip() in answer:
        return answer
    return answer.rstrip() + warning_section
