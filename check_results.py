import json

with open('results_folio_gemini_flash.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Check how many still failed
still_null = sum(1 for r in results if r.get('cot_pred') is None)
print(f"Still null: {still_null}")

# Overall accuracy
for pipeline in ["prolog", "python", "cot"]:
    pred_key = f"{pipeline}_pred"
    correct = sum(1 for r in results if r.get(pred_key) == r["gold_label"])
    total = sum(1 for r in results if r.get(pred_key) is not None)
    failed = sum(1 for r in results if r.get(pred_key) is None)
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"{pipeline.upper():>8}: {correct}/{total} correct ({accuracy:.1f}%) | {failed} failed")