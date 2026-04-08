from src.llm_client import LLMClient
from src.prolog_engine import PrologEngine
from src.python_engine import PythonEngine
from src.utils import clean_prolog_code, clean_python_code, split_program_and_query

class HybridPrologPipeline:
    def __init__(self, provider, api_key, model, max_retries=3):
        self.llm = LLMClient(provider, api_key, model)
        self.prolog = PrologEngine()
        self.max_retries = max_retries

    def run(self, problem):
        raw_code = self.llm.generate_prolog(problem)
        prolog_code = clean_prolog_code(raw_code)
        attempts = []

        for attempt in range(self.max_retries):
            success, output, error = self.prolog.execute(prolog_code)
            attempts.append({"attempt": attempt + 1, "code": prolog_code, "error": error})

            if error is None:
                return {
                    "problem": problem, "prolog_code": prolog_code,
                    "success": success, "output": output, "result": output,
                    "error": None, "attempts": attempts, "total_attempts": attempt + 1
                }

            print(f"  Attempt {attempt + 1} failed: {error}")
            print(f"  Asking LLM to fix...")
            raw_code = self.llm.refine_prolog(problem, prolog_code, error)
            prolog_code = clean_prolog_code(raw_code)

        return {
            "problem": problem, "prolog_code": prolog_code,
            "success": False, "output": None, "result": None,
            "error": f"Failed after {self.max_retries} attempts",
            "attempts": attempts, "total_attempts": self.max_retries
        }


class HybridPythonPipeline:
    def __init__(self, provider, api_key, model, max_retries=3):
        self.llm = LLMClient(provider, api_key, model)
        self.python = PythonEngine()
        self.max_retries = max_retries

    def run(self, problem):
        raw_code = self.llm.generate_python(problem)
        python_code = clean_python_code(raw_code)
        attempts = []

        for attempt in range(self.max_retries):
            success, output, error = self.python.execute(python_code)
            attempts.append({"attempt": attempt + 1, "code": python_code, "error": error})

            if error is None:
                return {
                    "problem": problem, "python_code": python_code,
                    "success": success, "output": output, "error": None,
                    "attempts": attempts, "total_attempts": attempt + 1
                }

            print(f"  Attempt {attempt + 1} failed: {error}")
            print(f"  Asking LLM to fix...")
            raw_code = self.llm.refine_python(problem, python_code, error)
            python_code = clean_python_code(raw_code)

        return {
            "problem": problem, "python_code": python_code,
            "success": False, "output": None,
            "error": f"Failed after {self.max_retries} attempts",
            "attempts": attempts, "total_attempts": self.max_retries
        }


class LLMOnlyPipeline:
    def __init__(self, provider, api_key, model):
        self.llm = LLMClient(provider, api_key, model)

    def run(self, problem):
        import re
        response = self.llm.generate_cot(problem)
        match = re.search(r'RESULT:\s*(TRUE|FALSE|UNCERTAIN)', response, re.IGNORECASE)
        if match:
            success = match.group(1).upper()
        else:
            success = None
        return {
            "problem": problem, "response": response,
            "success": success, "error": None if match else "No RESULT found"
        }