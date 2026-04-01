import re
from typing import Dict, List, Any
from collections import Counter

class SymbolExtractor:
    """
    Extracts high-level symbols (Variables, Functions, Constants) from Envision code
    using Regex for rapid "Mini-Map" generation.
    """
    
    # Regex Patterns
    # 1. Variables: Table.Var = ... (and Table.1 for tuples)
    # Matches: Items.Sold = ..., Catalog.0 = ...
    VAR_PATTERN = re.compile(r'\b([A-Z][a-zA-Z0-9_]*\.[a-zA-Z0-9_]+)\s*=', re.MULTILINE)
    
    # 2. Functions: def/process Name(...)
    # Matches: def process StockEvol, export def pure Rate, export def process PurchaseQty
    # Handles optional 'export', 'pure', 'process' keywords
    FUNC_PATTERN = re.compile(r'(?:export\s+)?(?:def|process)(?:\s+pure)?(?:\s+process)?\s+([a-zA-Z0-9_]+)', re.MULTILINE)
    
    # 3. Constants: const Name = ...
    # Matches: const inputPath, const onHandColor (lowercase supported)
    CONST_PATTERN = re.compile(r'\bconst\s+([a-zA-Z0-9_]+)\s*=', re.MULTILINE)
    
    # 4. Table Reads: read "..." as Table
    READ_PATTERN = re.compile(r'\bread\s+["\'].*?["\'](?:\s+unsafe)?\s+as\s+([A-Z][a-zA-Z0-9_]+)', re.MULTILINE)
    
    # 5. IsBoolean: Matches any usage of IsTitleCase (strong signal for boolean logic)
    IS_BOOL_PATTERN = re.compile(r'\b(Is[A-Z][a-zA-Z0-9_]+)\b')

    @classmethod
    def extract(cls, content: str) -> Dict[str, Dict[str, int]]:
        """
        Scans content and returns a frequency map of detected symbols.
        
        Returns:
            {
                "variables": {"Items.Sold": 1, "IsActive": 3},
                "functions": {"StockEvol": 1},
                "tables": {"Orders": 1}
            }
        """
        symbols = {
            "variables": Counter(),
            "functions": Counter(),
            "tables": Counter()
        }
        
        # Strip comments to avoid false positives? 
        # For a "Mini-Map" keeping comments might actually be okay (if a variable is mentioned in docs),
        # but for strict code structure, stripping is safer.
        # However, stripping is expensive. Let's try raw content first, regexes anchor to syntax (=, def).
        
        # 1. Variables (Table.Var assignments)
        for match in cls.VAR_PATTERN.finditer(content):
            symbols["variables"][match.group(1)] += 1
            
        # 2. Boolean Flags (Is...) - Logic usage
        # We classify these as variables too, or separate?
        # User asked for "Variables".
        for match in cls.IS_BOOL_PATTERN.finditer(content):
            symbols["variables"][match.group(1)] += 1
            
        # 3. Functions
        for match in cls.FUNC_PATTERN.finditer(content):
            symbols["functions"][match.group(1)] += 1
            
        # 4. Constants
        for match in cls.CONST_PATTERN.finditer(content):
            symbols["variables"][match.group(1)] += 1 # Constants are variables
            
        # 5. Tables (Reads)
        for match in cls.READ_PATTERN.finditer(content):
            symbols["tables"][match.group(1)] += 1

        # Convert Counters to dicts for JSON serialization
        return {k: dict(v) for k, v in symbols.items() if v}
