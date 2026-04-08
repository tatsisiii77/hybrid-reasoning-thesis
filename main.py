from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
import os
from src.pipeline import HybridPrologPipeline, HybridPythonPipeline, LLMOnlyPipeline

load_dotenv()

PROVIDER = "deepseek"
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-chat"


# Setup all three pipelines
prolog_pipeline = HybridPrologPipeline(PROVIDER, API_KEY, MODEL)
python_pipeline = HybridPythonPipeline(PROVIDER, API_KEY, MODEL)
llm_only_pipeline = LLMOnlyPipeline(PROVIDER, API_KEY, MODEL)

# Test problem
problem = "Ο Γιώργος είναι σεφ. Όλοι οι σεφ είναι μάγειρες. Είναι ο Γιώργος μάγειρας;"

print("=" * 60)
print("PROBLEM:", problem)
print("=" * 60)

# Pipeline 1: LLM + Prolog
print("\n--- LLM + Prolog ---")
result = prolog_pipeline.run(problem)
print(f"Code:\n{result['prolog_code']}")
print(f"Result: {'TRUE' if result['success'] else 'FALSE'}")
if result['error']:
    print(f"Error: {result['error']}")
print(f"Attempts: {result['total_attempts']}")

# Pipeline 2: LLM + Python
print("\n--- LLM + Python ---")
result = python_pipeline.run(problem)
print(f"Code:\n{result['python_code']}")
print(f"Result: {'TRUE' if result['success'] else 'FALSE'}")
if result['error']:
    print(f"Error: {result['error']}")
print(f"Attempts: {result['total_attempts']}")

# Pipeline 3: LLM Only (Chain-of-Thought)
print("\n--- LLM Only (CoT) ---")
result = llm_only_pipeline.run(problem)
print(f"Response:\n{result['response']}")
print(f"Result: {'TRUE' if result['success'] else 'FALSE' if result['success'] is not None else 'UNKNOWN'}")
if result['error']:
    print(f"Error: {result['error']}")