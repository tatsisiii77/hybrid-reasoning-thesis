import json
import time
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
import os
from google import genai
from src.utils import extract_result
from src.data_loader import FOLIOLoader

load_dotenv()

RESULTS_FILE = "results_folio_gemini_flash.json"
with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
    results = json.load(f)

failed_indices = [i for i, r in enumerate(results) if r.get("cot_pred") is None]
print(f"Found {len(failed_indices)} failed CoT problems")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
loader = FOLIOLoader("data/folio/folio-validation.jsonl")

STRICT_PROMPT = """Determine whether the conclusion is True, False, or Uncertain based on the premises.

Think step by step, then you MUST end your response with EXACTLY one of these three lines (copy-paste exactly):
RESULT: TRUE
RESULT: FALSE
RESULT: UNCERTAIN

Do NOT end with anything else. The last line of your response must start with "RESULT:"

Problem: """

fixed = 0

for idx in failed_indices:
    problem = loader.get_problem(idx)
    print(f"\nRetrying problem {idx + 1} (gold: {problem['gold_label']})...")

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=STRICT_PROMPT + problem["problem_text"]
            )
            text = response.text
            pred = extract_result(text)

            if pred is not None:
                results[idx]["cot_pred"] = pred
                results[idx]["cot_error"] = None
                print(f"  Fixed! Prediction: {pred} | Correct: {pred == problem['gold_label']}")
                fixed += 1
                break
            else:
                # Show last 100 chars to see what it wrote
                print(f"  Attempt {attempt + 1}: no RESULT found. Ending: ...{text[-100:]}")

        except Exception as e:
            print(f"  Attempt {attempt + 1} error: {e}")

        time.sleep(3)
    else:
        print(f"  Still failed after 3 attempts")

    time.sleep(2)

with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nFixed: {fixed}/{len(failed_indices)}")
print(f"Still failed: {len(failed_indices) - fixed}")
print(f"Results saved to {RESULTS_FILE}")