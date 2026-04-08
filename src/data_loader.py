import json

class FOLIOLoader:
    """Loads and formats FOLIO benchmark problems."""

    def __init__(self, filepath):
        self.problems = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                self.problems.append(json.loads(line))

    def get_problem_count(self):
        return len(self.problems)

    def get_problem(self, index):
        """Get a single problem formatted for our pipeline."""
        item = self.problems[index]

        # Format premises as a single text
        premises_text = " ".join(item["premises"])

        # Build the full problem
        problem_text = f"""Given the following premises:
{premises_text}

Conclusion: {item['conclusion']}

Is the conclusion True, False, or Uncertain based on the premises?"""

        return {
            "index": index,
            "problem_text": problem_text,
            "premises": item["premises"],
            "conclusion": item["conclusion"],
            "gold_label": item["label"],
        }

    def get_all_problems(self):
        """Get all problems formatted for our pipeline."""
        return [self.get_problem(i) for i in range(len(self.problems))]