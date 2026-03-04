from typing import List, Dict, Any
from .base_benchmark import Benchmark
import numpy as np
import time
import re
import json
import agents.prepare_agent as prepare_agent
import config_manager

default_prompt = "Tu es un évaluateur strict. Lorsque je te soumets une question, la vraie réponse et la réponse d'un LLM, tu dois générer UNIQUEMENT un seul caractère : 1 si la réponse du LLM est correcte, ou 0 si la réponse est incorrecte. Si la réponse du LLM est très incomplète et qu'il manque un élément important, mets la note de 0. Si la réponse du LLM contient des éléments en plus par rapport à la vraie réponse, mais qu'elle contient tous les éléments de la vraie réponse, mets la note de 1. Ne génère PAS d'explications, de raisonnement, de balises, d'espaces, de ponctuation ou tout autre texte supplémentaire. Si tu génères autre chose que strictement 1 ou 0, tu as échoué à la tâche. Ta réponse complète doit être exactement d'un caractère : 1 ou 0"

default_prompt2 = """Tu es un juge expert en évaluation de systèmes RAG (Retrieval-Augmented Generation) pour une base de code d'entreprise.
Ton rôle est d'évaluer la 'Réponse du LLM' par rapport à la 'Vraie réponse' (la référence de vérité).

CRITÈRES D'ÉVALUATION (Échelle de 1 à 5) :
- Questions Techniques (ex: "Combien de scripts...", "Quels fichiers lisent...") : Sois impitoyable. Si la réponse omet un script ou invente un chemin de fichier, pénalise sévèrement (Score 1, 2 ou 3 maximum). Un score de 5 nécessite une précision absolue sur les listes et les nombres.
- Questions Conceptuelles (ex: "Comment fonctionne...", "Que fait...") : Évalue la pertinence sémantique et l'exactitude factuelle. Si la réponse du LLM est plus verbeuse mais factuellement correcte et contient tous les concepts clés de la référence, accorde un 4 ou 5.

ÉCHELLE DE NOTATION :
5 : Parfait. Exactitude totale (nombres/listes exacts, ou concept parfaitement expliqué).
4 : Très bon. Légère omission ou détail mineur manquant, mais l'essentiel est correct.
3 : Partiel. Contient des éléments corrects mais a des omissions majeures ou quelques hallucinations de fichiers.
2 : Mauvais. Fortes hallucinations ou répond à côté de la plaque.
1 : Faux. Totalement incorrect ou affirme l'inverse de la référence.

FORMAT DE SORTIE REQUIS :
Tu dois générer UNIQUEMENT un objet JSON valide avec deux clés : "reasoning" (ta justification détaillée) et "score" (un entier entre 1 et 5). Ne génère aucun texte en dehors de ce JSON.
Exemple :
{"reasoning": "Le LLM a identifié 27 scripts au lieu de 28. Il manque le script '/4. Optimization workflow/02 Suppliers'.", "score": 3}
"""

class LLMAsAJudgeBenchmark(Benchmark):
    """
    Benchmark using an LLM as a judge to evaluate the correctness of default LLM's response
    """

    def __init__(self, prompt = default_prompt):
        self.config = config_manager.get_config()
        self.rate_limit_delay = self.config.get('agent.rate_limit_delay', 0)
        self.prompt = prompt
        
    def initialize(self):
        self.agent = prepare_agent.prepare_benchmark_agent()

    def judge(self, question: str, llm_response: str, reference: str) -> int:
        """Returns 1 if the llm_response is considered correct by the judge llm, else 0"""
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        
        text_score = self.agent.generate_response(self.prompt + f"\n\nQuestion : {question}\nVraie réponse : {reference}\nRéponse du LLM : {llm_response}")
        if text_score not in ["1", "0"]:
            raise Exception("Score invalide")
        return (int(text_score))

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

class LLMAsAJudgeBenchmark2(Benchmark):
    """
    Benchmark using an LLM as a judge to evaluate the correctness of an agent's response on a 1-5 scale.
    """

    def __init__(self, prompt=default_prompt2):
        self.config = config_manager.get_config()
        self.rate_limit_delay = self.config.get('agent.rate_limit_delay', 0)
        self.prompt = prompt
        
    def initialize(self):
        self.agent = prepare_agent.prepare_benchmark_agent()

    def judge(self, question: str, llm_response: str, reference: str) -> Dict[str, Any]:
        """Returns the reasoning and a normalized score (0.0 to 1.0)"""
        
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
            
        full_prompt = f"{self.prompt}\n\nQuestion : {question}\nVraie réponse : {reference}\nRéponse du LLM : {llm_response}"
        raw_response = self.agent.generate_response(full_prompt)
        
        # Robust JSON extraction (in case the LLM wraps it in ```json ... ``` tags)
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if not json_match:
            raise ValueError(f"Le LLM n'a pas retourné de JSON valide. Réponse brute: {raw_response}")
            
        parsed_eval = json.loads(json_match.group(0))
        
        score_1_to_5 = int(parsed_eval.get("score", 0))
        if score_1_to_5 < 1 or score_1_to_5 > 5:
            raise ValueError(f"Score hors limites: {score_1_to_5}")
            
        # Normalize the 1-5 scale to a 0.0 - 1.0 percentage for your mean_score
        normalized_score = (score_1_to_5 - 1) / 4.0 
        
        return {
            "score": normalized_score,
            "raw_score_1_to_5": score_1_to_5,
            "reasoning": parsed_eval.get("reasoning", "Pas de justification fournie.")
        }

    def run(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        issues = 0 
        
        for item in data:
            try:
                eval_data = self.judge(item["question"], item["llm_response"], item["reference"])
                results.append({
                    "question": item["question"],
                    "llm_response": item["llm_response"],
                    "reference": item["reference"],
                    "score": eval_data["score"],
                    "raw_score_1_to_5": eval_data["raw_score_1_to_5"],
                    "reasoning": eval_data["reasoning"]
                })
            except Exception as e:
                print(f"Issue evaluating question '{item['question']}': {e}")
                issues += 1

        mean_score = float(np.mean([r["score"] for r in results])) if results else 0.0
        
        return {
            "results": results, 
            "mean_score": mean_score, 
            "issues": issues, 
            "prompt": self.prompt
        }