import json
import time
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
import os
from src.data_loader import FOLIOLoader
from src.pipeline import HybridPrologPipeline, HybridPythonPipeline, LLMOnlyPipeline
from src.utils import extract_result, normalize_label

load_dotenv()

# Configuration
PROVIDER = "gemini"
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"
NUM_PROBLEMS = 204
RESULTS_FILE = "results_folio_gemini_flash.json"
DELAY_BETWEEN_PROBLEMS = 5  # seconds between problems

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)
    start_from = len(results)
    print(f"Resuming from problem {start_from + 1} (found {start_from} existing results)")
else:
    results = []
    start_from = 0

loader = FOLIOLoader("data/folio/folio-validation.jsonl")
print(f"Loaded {loader.get_problem_count()} problems from FOLIO")
print(f"Running problems {start_from + 1} to {min(NUM_PROBLEMS, loader.get_problem_count())}...\n")

prolog_pipeline = HybridPrologPipeline(PROVIDER, API_KEY, MODEL)
python_pipeline = HybridPythonPipeline(PROVIDER, API_KEY, MODEL)
cot_pipeline = LLMOnlyPipeline(PROVIDER, API_KEY, MODEL)

def save_results():
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

for i in range(start_from, min(NUM_PROBLEMS, loader.get_problem_count())):
    problem = loader.get_problem(i)
    print(f"{'='*60}")
    print(f"Problem {i+1}/{NUM_PROBLEMS}")
    print(f"Gold Label: {problem['gold_label']}")
    print(f"Conclusion: {problem['conclusion'][:80]}...")
    print(f"{'='*60}")

    gold = normalize_label(problem["gold_label"])
    entry = {"index": i, "gold_label": gold, "conclusion": problem["conclusion"]}

    print("\n  [Prolog] Running...")
    try:
        prolog_result = prolog_pipeline.run(problem["problem_text"])
        prolog_output = prolog_result.get("output", "") or ""
        prolog_pred = extract_result(prolog_output)
        entry["prolog_pred"] = prolog_pred
        entry["prolog_attempts"] = prolog_result.get("total_attempts", 0)
        entry["prolog_error"] = prolog_result["error"]
        print(f"  [Prolog] Prediction: {prolog_pred} | Correct: {prolog_pred == gold}")
    except Exception as e:
        entry["prolog_pred"] = None
        entry["prolog_error"] = str(e)
        print(f"  [Prolog] Error: {e}")

    time.sleep(DELAY_BETWEEN_PROBLEMS)

    print("  [Python] Running...")
    try:
        python_result = python_pipeline.run(problem["problem_text"])
        python_output = python_result.get("output", "") or ""
        python_pred = extract_result(python_output)
        entry["python_pred"] = python_pred
        entry["python_attempts"] = python_result.get("total_attempts", 0)
        entry["python_error"] = python_result["error"]
        print(f"  [Python] Prediction: {python_pred} | Correct: {python_pred == gold}")
    except Exception as e:
        entry["python_pred"] = None
        entry["python_error"] = str(e)
        print(f"  [Python] Error: {e}")

    time.sleep(DELAY_BETWEEN_PROBLEMS)

    print("  [CoT] Running...")
    try:
        cot_result = cot_pipeline.run(problem["problem_text"])
        cot_pred = extract_result(cot_result.get("response", "") or "")
        entry["cot_pred"] = cot_pred
        entry["cot_error"] = cot_result["error"]
        print(f"  [CoT] Prediction: {cot_pred} | Correct: {cot_pred == gold}")
    except Exception as e:
        entry["cot_pred"] = None
        entry["cot_error"] = str(e)
        print(f"  [CoT] Error: {e}")

    results.append(entry)
    save_results()  # Save after each problem
    print(f"\n  [Saved] Progress saved ({len(results)} problems completed)")

    time.sleep(DELAY_BETWEEN_PROBLEMS)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for pipeline_name in ["prolog", "python", "cot"]:
    pred_key = f"{pipeline_name}_pred"
    correct = sum(1 for r in results if r.get(pred_key) == r["gold_label"])
    total = sum(1 for r in results if r.get(pred_key) is not None)
    failed = sum(1 for r in results if r.get(pred_key) is None)
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"{pipeline_name.upper():>8}: {correct}/{total} correct ({accuracy:.1f}%) | {failed} failed")

print(f"\nResults saved to {RESULTS_FILE}")