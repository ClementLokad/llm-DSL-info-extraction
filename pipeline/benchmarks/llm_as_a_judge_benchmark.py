from typing import List, Dict, Any
from .base_benchmark import Benchmark
import numpy as np
import agents.prepare_agent as prepare_agent
import config_manager

default_prompt = "Tu es un évaluateur strict. Lorsque je te soumets une , la vraie réponse et la réponse d'un LLM, tu dois générer UNIQUEMENT un seul caractère : 1 si la réponse du LLM est entièrement correcte, ou 0 si la réponse est incorrecte. Si la réponse du LLM est incomplète et qu'il manque un élément, mets la note de 0. Si la réponse du LLM a le même sens que la vraie réponse, mets la note de 1. Ne génère PAS d'explications, de raisonnement, de balises, d'espaces, de ponctuation ou tout autre texte supplémentaire. Si tu génères autre chose que strictement 1 ou 0, tu as échoué à la tâche. Ta réponse complète doit être exactement d'un caractère : 1 ou 0"

class LLMAsAJudgeBenchmark(Benchmark):
    """
    Benchmark using an LLM as a judge to evaluate the correctness of default LLM's response
    """

    def __init__(self, prompt = default_prompt):
        self.config = config_manager.get_config()
        self.rate_limit_delay = self.config.get('agent.rate_limit_delay', 0)
        self.prompt = default_prompt
        
    def initialize(self):
        self.agent = prepare_agent.prepare_benchmark_agent()

    def judge(self, question: str, llm_response: str, reference: str) -> int:
        """Returns 1 if the llm_response is considered correct by the judge llm, else 0"""
        text_score = self.agent.generate_response(self.prompt + f"\n\nQuestion : {question}\nVraie réponse : {reference}\nRéponse du LLM : {llm_response}")
        # print(text_score)
        if text_score not in ["1", "0"]:
            raise Exception("Score invalide")
        return (int(text_score))

    def run(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
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
        issues = 0 # Nombre d'items pour lesquels le judge a donné un jugement invalide
        for item in data:
            try:
                score = self.judge(item["question"], item["llm_response"], item["reference"])
            except Exception:
                issues +=1
            else:
                results.append({
                    "question": item["question"],
                    "llm_response": item["llm_response"],
                    "reference": item["reference"],
                    "score": score
                })
        mean_score = float(np.mean([r["score"] for r in results]))
        return {"results": results, "mean_score": mean_score, "issues" : issues, "prompt" : self.prompt}
