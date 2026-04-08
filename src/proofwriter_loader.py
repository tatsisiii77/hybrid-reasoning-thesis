import json

class ProofWriterLoader:
    """Loads and formats ProofWriter benchmark problems."""

    def __init__(self, filepath, target_depth=None):
        """
        Args:
            filepath: Path to the meta-dev.jsonl file
            target_depth: If set, only include problems with exactly this QDep
        """
        self.rulebases = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                self.rulebases.append(json.loads(line))

        # Flatten: each problem is one (theory, question) pair
        self.problems = []
        for rulebase in self.rulebases:
            theory = rulebase["theory"]
            for q_id, q_data in rulebase["questions"].items():
                depth = q_data.get("QDep", 0)
                
                # Filter by target depth if specified
                if target_depth is not None and depth != target_depth:
                    continue
                
                self.problems.append({
                    "rulebase_id": rulebase["id"],
                    "question_id": q_id,
                    "theory": theory,
                    "question": q_data["question"],
                    "answer": q_data["answer"],
                    "depth": depth,
                })

    def get_problem_count(self):
        return len(self.problems)

    def get_problem(self, index):
        """Get a single problem formatted for our pipeline."""
        item = self.problems[index]

        problem_text = f"""Given the following premises:
{item['theory']}

Conclusion: {item['question']}

Is the conclusion True, False, or Unknown based on the premises?"""

        # Normalize answer to TRUE/FALSE/UNCERTAIN format
        ans = item["answer"]
        if ans is True:
            gold = "TRUE"
        elif ans is False:
            gold = "FALSE"
        else:
            gold = "UNCERTAIN"

        return {
            "index": index,
            "problem_text": problem_text,
            "theory": item["theory"],
            "question": item["question"],
            "gold_label": gold,
            "depth": item["depth"],
            "rulebase_id": item["rulebase_id"],
            "question_id": item["question_id"],
        }