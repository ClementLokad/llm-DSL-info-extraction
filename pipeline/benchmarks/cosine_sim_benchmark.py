from typing import List, Dict, Any
import numpy as np
from sentence_transformers import util
from .base_benchmark import Benchmark
from rag.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from config_manager import get_config


class CosineSimBenchmark(Benchmark):
    """
    Benchmark basé sur la similarité cosinus entre les embeddings
    de la réponse du LLM et de la réponse attendue.
    """

    def __init__(self, embedder: SentenceTransformerEmbedder = None):
        """
        Args:
            embedder: instance initialisée de SentenceTransformerEmbedder
        """
        if embedder is None:
            self.embedder = SentenceTransformerEmbedder(get_config().get_embedder_config())
            self.embedder.initialize()
        else:
            self.embedder = embedder

    def compute_similarity(self, llm_response: str, reference: str) -> float:
        """Calcule la similarité cosinus entre deux textes."""
        emb1 = self.embedder.embed_text(llm_response)
        emb2 = self.embedder.embed_text(reference)
        return float(util.cos_sim(emb1, emb2).item())

    def run(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcule les similarités pour chaque paire (LLM ↔ référence).

        Args:
            data: liste de dicts contenant :
                - question
                - llm_response
                - reference

        Returns:
            dict avec :
                - results : scores individuels
                - mean_score : moyenne globale
        """
        results = []
        for item in data:
            score = self.compute_similarity(item["llm_response"], item["reference"])
            results.append({
                "question": item["question"],
                "llm_response": item["llm_response"],
                "reference": item["reference"],
                "score": score
            })

        mean_score = float(np.mean([r["score"] for r in results]))
        return {"results": results, "mean_score": mean_score}
