from pipeline.benchmarks.dual_cross_encoder_benchmark import DualBenchmark
from typing import List, Dict, Any
import numpy as np

class HybridBenchmark:
    """A benchmark that uses both a simple algorithmic check for factual consistency (define as deterministic questions) and a dual cross-encoder benchmark for more high level questions.
    For a given question, if it is deterministic, we will check if the reference answer is contained in the llm response, if it is the case we will give it a score of 1, else 0.
    """
    def __init__(self):
        self.dual_benchmark = DualBenchmark()

    def initialize(self):
        self.dual_benchmark.initialize()

    def run(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
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
                # Check if each reference line is in the LLM response (with handling for spacing differences)
                matched = sum(1 for ref_line in ref_lines if ref_line in llm_text or ref_line.replace("  ", " ") in llm_text)
                
                # Score: 1.0 if all items found, else ratio of matched items
                score = 1.0 if matched == len(ref_lines) else (matched / len(ref_lines) if ref_lines else 0.0)
                results.append({
                    "question": item["question"],
                    "llm_response": item["llm_response"],
                    "reference": item["reference"],
                    "score": score,
                    "method": "deterministic_check",
                    "matched": f"{matched}/{len(ref_lines)}"
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