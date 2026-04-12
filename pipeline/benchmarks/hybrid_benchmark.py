from pipeline.benchmarks.dual_cross_encoder_benchmark import DualBenchmark
from typing import List, Dict, Any
import numpy as np
import re

class HybridBenchmark:
    """A benchmark that uses both a simple algorithmic check for factual consistency (define as deterministic questions) and a dual cross-encoder benchmark for more high level questions.
    For a given question, if it is deterministic, we will check if the reference answer is contained in the llm response, if it is the case we will give it a score of 1, else 0.
    """
    def __init__(self):
        self.dual_benchmark = DualBenchmark()

    def initialize(self):
        self.dual_benchmark.initialize()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for matching by:
        - Collapsing all whitespace sequences to single space (handles spaces, tabs, newlines, etc.)
        - Converting to lowercase for case-insensitive comparison
        - Stripping punctuation except path separators (/, - and .)
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text ready for comparison
        """
        # Collapse all whitespace sequences (spaces, tabs, newlines, etc.) to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Remove punctuation except forward slash (path separator)
        # Keep: alphanumerics, spaces, forward slashes, hyphens (common in paths), underscores, dots (some vars use them)
        text = re.sub(r'[^\w\s/\-\.]', '', text)
        
        return text

    def run(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Args:
            data: dict list containing :
                - question : the question asked to the answerer llm
                - llm_response : the response given by the answerer llm
                - reference : the exepected answer to the question
                - deterministic : boolean indicating if the question is deterministic or not

        Returns:
            dict containing :
                - results : individual score to each question
                - mean_score : average success rate
        """
        results = []
        for item in data:
            if item.get("deterministic", False):
                # Handle reference as either string or list
                ref_text = item["reference"] if isinstance(item["reference"], str) else "\n".join([str(a) for a in item["reference"]])
                llm_text = item["llm_response"]
                
                # Extract individual lines/items from reference and check coverage
                ref_lines = [line.strip() for line in ref_text.split('\n') if line.strip()]
                
                # Normalize LLM response once
                normalized_response = self._normalize_text(llm_text)
                
                # Check if each normalized reference line appears in normalized LLM response
                matched = 0
                normalized_ref_lines = []
                for ref_line in ref_lines:
                    normalized_ref = self._normalize_text(ref_line)
                    normalized_ref_lines.append(normalized_ref)
                    if normalized_ref and normalized_ref in normalized_response:
                        matched += 1
                
                # Binary scoring: 1.0 if ALL reference items found, 0.0 otherwise
                score = 1.0 if matched == len(ref_lines) and len(ref_lines) > 0 else 0.0
                
                results.append({
                    "question": item["question"],
                    "llm_response": item["llm_response"],
                    "reference": item["reference"],
                    "score": score,
                    "method": "deterministic_check",
                    "matched": f"{matched}/{len(ref_lines)}",
                    "normalized_response": normalized_response,
                    "normalized_reference_lines": normalized_ref_lines
                })
            else:
                # Use the dual benchmark for non-deterministic questions
                scores = self.dual_benchmark.evaluate_generation_comprehensive(item["question"], item["reference"], item["llm_response"])
                results.append({
                    "question": item["question"],
                    "llm_response": item["llm_response"],
                    "reference": item["reference"],
                    "score": scores["final_score"],
                    **scores,
                    "method": "dual_benchmark"
                })
        mean_score = float(np.mean([r["score"] for r in results]))
        return {"results": results, "mean_score": mean_score}