from typing import Dict, List, Any
from sentence_transformers import CrossEncoder
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pipeline.benchmarks.base_benchmark import Benchmark
import numpy as np

class DualBenchmark(Benchmark):
    def initialize(self):
        self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
        self.relevance_model = TextCrossEncoder("jinaai/jina-reranker-v2-base-multilingual")

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def evaluate_generation_comprehensive(self, query: str, reference_answer: str, generated_answer: str) -> dict:
        # --- Step 1: Check for Factual Contradictions ---
        nli_logits = self.nli_model.predict((str(reference_answer), str(generated_answer)))
        nli_probs = self.softmax(nli_logits)
        contradiction_prob = float(nli_probs[0])
        entailment_prob = float(nli_probs[1])
        
        # --- Step 2: Check for Query Relevance ---
        # fastembed expects pairs in a list
        relevance_score = list(self.relevance_model.rerank(query, [generated_answer]))[0]
        # Apply sigmoid to normalize the relevance logit
        relevance_prob = 1 / (1 + np.exp(-relevance_score))
        
        # --- Step 3: Calculate Final Composite Score ---
        # We heavily penalize the score if there is a factual contradiction.
        # Otherwise, the score is a blend of how much it entails the reference AND how relevant it is to the query.
        
        if contradiction_prob > 0.5:
            # It hallucinated or lied. Score is tanked.
            final_score = 0.0
        else:
            # Blend the strict fact-match with the overall helpfulness
            # (You can tweak these weights! 60% fact, 40% relevance is a good start)
            final_score = (entailment_prob * 0.6) + (relevance_prob * 0.4)
            
        return {
            "final_score": final_score,
            "contradiction": contradiction_prob,
            "entailment": entailment_prob,
            "relevance": relevance_prob
        }
    
    def run(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcule les similarités pour chaque paire (LLM ↔ référence).

        Args:
            data: dict list containing :
                - question : the question asked to the answerer llm
                - llm_response : the response given by the answerer llm
                - reference : the exepected answer to the question

        Returns:
            dict containing :
                - results : individual score to each question
                - mean_score : average success rate
        """
        results = []
        for item in data:
            scores = self.evaluate_generation_comprehensive(item["question"], item["reference"], item["llm_response"])
            results.append({
                "question": item["question"],
                "llm_response": item["llm_response"],
                "reference": item["reference"],
                "score": scores["final_score"],
                **scores
            })
        mean_score = float(np.mean([r["score"] for r in results]))
        return {"results": results, "mean_score": mean_score}

if __name__ == "__main__":
    benchmark = DualBenchmark()
    benchmark.initialize()
    
    query = "Do I need to force the refund?"
    ground_truth = "Yes, use force=True."
    
    print("--- Test: Verbose but Correct Answer ---")
    gen_answer = "Yes, you must trigger it manually. To do this, call the process_refund() function and pass the argument force=True, otherwise the system will block it."
    
    scores = benchmark.evaluate_generation_comprehensive(query, ground_truth, gen_answer)
    print(f"Final Benchmark Score: {scores['final_score'] * 100:.1f}/100")
    print(f"  - Contradiction Risk: {scores['contradiction'] * 100:.1f}%")
    print(f"  - Strict Fact Match: {scores['entailment'] * 100:.1f}%")
    print(f"  - Query Relevance: {scores['relevance'] * 100:.1f}%")