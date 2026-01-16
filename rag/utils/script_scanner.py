"""Advanced script parser for resolving constants and finding structured references"""
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

PLACEHOLDER_RE = re.compile(r"\\?\{([A-Za-z0-9_]+)\}")
CONST_DECL_RE = re.compile(r'^\s*const\s+([A-Za-z0-9_]+)\s*=\s*"(.*)"\s*$')
STRING_LITERAL_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')

def collect_constants(script_content: str) -> Dict[str, str]:
    consts: Dict[str, str] = {}
    lines = script_content.splitlines()
    for raw in lines:
        match = CONST_DECL_RE.match(raw.strip())
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        consts[key] = _resolve_placeholders(value, consts)
    return consts

def _resolve_placeholders(text: str, consts: Dict[str, str], *, depth: int = 0) -> str:
    if depth > 10:
        return text
    replaced = PLACEHOLDER_RE.sub(lambda match: consts.get(match.group(1), ""), text)
    if replaced == text:
        return replaced
    return _resolve_placeholders(replaced, consts, depth=depth + 1)

def scan_string_for_references(
    script_content: str,
    target_path_fragment: str,
    consts: Dict[str, str],
    verbs: List[str] = None,
    need_cleaned_string: bool = False
) -> List[Dict[str, Any]]:
    """
    Scan a script content string for references to target_path, resolving constants.
    """
    if verbs is None:
        verbs = ["read", "write", "import"]
        
    verbs_pattern = "|".join(re.escape(v) for v in verbs)
    statement_re = re.compile(rf"\b({verbs_pattern})\s*:?\s*\"(.*?)\"", re.IGNORECASE)
    
    hits: List[Dict[str, Any]] = []

    lines = script_content.splitlines()
    
    # Pre-calculate variants of the target to match against
    normalized_target = (target_path_fragment or "").strip()
    stripped_target = normalized_target.lstrip("/")
    pattern_regex = re.compile(stripped_target, re.IGNORECASE)
    
    final_content = []


    for line_no, raw_line in enumerate(lines):
        final_line = raw_line
        # 1. Check for explicit verb statements (read: "...", write "...")
        match = statement_re.search(raw_line)
        if match:
            verb = match.group(1).lower()
            literal_value = match.group(2)
            
            # Resolve constants in the path string
            resolved_path = _resolve_placeholders(literal_value, consts).replace("\\", "")
            
            if need_cleaned_string and literal_value != resolved_path:
                final_line = _resolve_placeholders(raw_line, consts)
            
            if pattern_regex.search(resolved_path):
                hits.append({
                    "line": line_no,
                    "verb": verb,
                    "raw": raw_line.strip(),
                    "resolved_path": resolved_path
                })
                final_content.append(final_line)
                continue # If found as a verb statement, we can skip pure literal check for this line? 
                         # Actually user code checks both but maintains uniqueness. Let's keep it simple.

        # 2. Also check for raw string literals just in case (e.g. variable assignment)
        # Only if we want to be very exhaustive. The user user instruction says:
        # "si plus generalement on cherche tous les read ..., write ... ou import ...".
        # But also: "le re.Search sera dans les autres cas".
        # The provided snippet DOES check literals if `keyword` is None.
        # "When keyword is omitted, both verbs are considered and the function also falls back to matching any quoted literal"
        
        # We will keep literal check but ensure uniqueness if already found
        for literal in STRING_LITERAL_RE.findall(raw_line):
            resolved_literal = _resolve_placeholders(literal, consts).replace("\\", "")
            
            if pattern_regex.search(resolved_literal):
                # Check duplication
                if any(h['line'] == line_no for h in hits):
                    continue
                    
                hits.append({
                    "line": line_no,
                    "verb": "literal",
                    "raw": raw_line.strip(),
                    "resolved_path": resolved_literal
                })

            if need_cleaned_string and literal != resolved_literal:
                final_line = _resolve_placeholders(raw_line, consts)
        
        final_content.append(final_line)
            
    if need_cleaned_string:
        return hits, "\n".join(final_content)
    return hits

def scan_script_for_references(
    file_path: Path,
    target_path_fragment: str,
    verbs: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Scan a single script for references to target_path, resolving constants.
    """
    try:
        script_content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    
    consts = collect_constants(script_content)
    
    return scan_string_for_references(
        script_content,
        target_path_fragment,
        consts,
        verbs
    )