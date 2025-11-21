from typing import List, Dict, Any
from config_manager import ConfigManager
from .base_benchmark import Benchmark
import numpy as np

default_prompt = "You're an expert at judging answers to questions by answering only a single number. Return JUST 1 if the LLM response to the answer give is correct, else JUST return 0. Don't explain anything just give the number 1 or 0 so it can go into a int() function"

class LLMAsAJudgeBenchmark(Benchmark):
    """
    Benchmark using an LLM as a judge to evaluate the correctness of default LLM's response
    """
    def __init__(self):
        self.config = ConfigManager()
        self.rate_limit_delay = self.config.get('agent.rate_limit_delay', 0)

    def initialize(self, verbose=True):
        agent_type = self.config.get_benchmark_agent()
        if agent_type == 'mistral':
            from agents.mistral_agent import MistralAgent
            self.agent = MistralAgent()
        elif agent_type == 'gemini':
            from agents.gemini_agent import GeminiAgent
            self.agent = GeminiAgent()
        elif agent_type == 'groq':
            from agents.groq_agent import GroqAgent
            self.agent = GroqAgent()
        else:
            from agents.gpt_agent import GPTAgent
            self.agent = GPTAgent()            
        self.agent.initialize()
    
    def judge (self, llm_response: str, reference: str)-> int:
        """Returns 1 if the llm_response is considered correct by the judge llm, else 0"""
        text_score = self.agent.generate_response("You are a strict evaluator. When I give you a question and an LLM’s answer, you must output ONLY a single character: 1 if the answer is fully correct, or 0 if the answer is incorrect. Do NOT output explanations, chain-of-thought, tags, spaces, punctuation, or any extra text. If you output anything other than exactly 1 or 0, you have failed the task. Your entire reply must be exactly one character: 1 or 0." + "Question :" + llm_response + "Expected answer :" + reference)
        return(int(text_score))


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
        for item in data:
            score = self.judge(item["llm_response"], item["reference"])
            results.append({
                "question": item["question"],
                "llm_response": item["llm_response"],
                "reference": item["reference"],
                "score": score
            })
        mean_score = float(np.mean([r["score"] for r in results]))
        return {"results": results, "mean_score": mean_score}

# if __name__ == "__main__":
#     b = LLMAsAJudgeBenchmark()
#     b.initialize()
#     print(b.judge("What is the capital of Germany", "Paris"))