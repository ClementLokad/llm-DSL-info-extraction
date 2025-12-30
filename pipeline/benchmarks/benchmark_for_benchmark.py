from .llm_as_a_judge_benchmark import LLMAsAJudgeBenchmark
import json

# lancer ce script avec python -m pipeline.benchmarks.benchmark_for_benchmark depuis la racine du projet

with open("benchmark_for_benchmark.json", 'r', encoding='utf-8') as file:
    samples = json.load(file)

benchmark = LLMAsAJudgeBenchmark()
benchmark.initialize()
data = benchmark.run([{"question":item["question"], "llm_response":item["llm_answer"], "reference":item["reference"]} for item in samples])
results = data["results"]
issues = data["issues"]
print(issues, "réponses du juge non valides")
n=len(results)
s = 0
for i in range(n):
    while results[i]["question"]!=samples[i]["question"]:
        samples.pop(i)
    if results[i]["score"]==samples[i]["score"]:
        s+=1
        
print(f"note finale : {s}/{n}")
    