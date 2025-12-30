from langgraph_base import BasePipeline, GraphState, BenchmarkState
from main import DSLQuerySystem
import json 

#  Ce script génère un fichier json contenant les questions du benchmark, les vraies réponses et les réponses du LLM

verbose = False

system = DSLQuerySystem()   
system.initialize(verbose=verbose)

with open('questions.json', 'r', encoding='utf-8') as file:
    questions = json.load(file)["answered"]

samples = []

for qa_pair in questions:
    question = qa_pair['question']
    llm_answer = system.query(question)
    samples.append({"question": question, "llm_answer": llm_answer, "reference": qa_pair["answers"]})

# print(samples)

with open("samples.json", 'w', encoding='utf-8') as file:
    json.dump(samples, file, ensure_ascii=False)
    
    