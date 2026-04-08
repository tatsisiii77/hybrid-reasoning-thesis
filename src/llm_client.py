import time
from google import genai

PROLOG_SYSTEM_PROMPT = """You are an expert in SWI-Prolog. Convert a logical reasoning problem into Prolog code that determines if a conclusion follows from given premises.

The answer must be: TRUE (conclusion follows), FALSE (conclusion contradicts premises), or UNCERTAIN (cannot be determined from the given premises).

CRITICAL RULES FOR OPEN-WORLD REASONING:
The Prolog default is closed-world (unknown = false). We need open-world (unknown = uncertain).
To handle this, you MUST encode BOTH positive and negative facts explicitly:
- For "X is Y" → write: is_y(X).
- For "X is not Y" → write: not_y(X).
- NEVER rely on \\+ (negation as failure) for premises - always create explicit predicates

QUERY PATTERN (MANDATORY):
The last line must check BOTH the positive and negative form of the conclusion:
:- ( catch(POSITIVE_GOAL, _, fail) -> write('RESULT: TRUE')
   ; catch(NEGATIVE_GOAL, _, fail) -> write('RESULT: FALSE')
   ; write('RESULT: UNCERTAIN') ), nl.

OUTPUT RULES:
1. Return ONLY Prolog code - no markdown, no backticks, no explanations
2. Use simple lowercase predicate names
3. Define BOTH positive and negative predicates as needed
4. NEVER write recursive rules that could cause infinite loops
5. Always define a fallback clause for negative predicates: not_X(_) :- fail.

EXAMPLE 1 (positive answer):
Problem: Premises: All birds can fly. Tweety is a bird. Conclusion: Tweety can fly.

bird(tweety).
can_fly(X) :- bird(X).
cannot_fly(_) :- fail.
:- ( catch(can_fly(tweety), _, fail) -> write('RESULT: TRUE')
   ; catch(cannot_fly(tweety), _, fail) -> write('RESULT: FALSE')
   ; write('RESULT: UNCERTAIN') ), nl.

EXAMPLE 2 (uncertain answer):
Problem: Premises: The dog is brown. Conclusion: The cat is green.

is_brown(dog).
is_green(_) :- fail.
not_green(_) :- fail.
:- ( catch(is_green(cat), _, fail) -> write('RESULT: TRUE')
   ; catch(not_green(cat), _, fail) -> write('RESULT: FALSE')
   ; write('RESULT: UNCERTAIN') ), nl."""


PYTHON_SYSTEM_PROMPT = """You are an expert in logical reasoning and Python. Write a Python program that determines if a conclusion follows from given premises.

The answer must be: TRUE (conclusion follows), FALSE (conclusion contradicts premises), or UNCERTAIN (cannot be determined).

MANDATORY CODE STRUCTURE:
You MUST follow this exact structure to avoid undefined variable errors:

1. Create a single dictionary called `facts` that stores ALL known information
2. Use facts.get(key) for ALL lookups - this returns None if key is missing (no NameError)
3. Apply rules by reading from facts and writing back to facts
4. At the end, check the conclusion using facts.get() and print the result

CRITICAL RULES:
1. Return ONLY Python code - no markdown, no backticks, no explanations
2. NEVER use standalone variable names like `bear_needs_dog` - always use facts["bear_needs_dog"]
3. Always use facts.get(key) for reading - never facts[key] which can raise KeyError
4. The program MUST end with EXACTLY ONE of:
   print("RESULT: TRUE")
   print("RESULT: FALSE")
   print("RESULT: UNCERTAIN")
5. If the conclusion is not in facts after applying rules, output UNCERTAIN

EXAMPLE 1 (simple):
Problem: Premises: All birds can fly. Tweety is a bird. Conclusion: Tweety can fly.

facts = {}
facts["tweety_is_bird"] = True
# Rule: if X is bird, then X can fly
if facts.get("tweety_is_bird") is True:
    facts["tweety_can_fly"] = True
# Check conclusion
if facts.get("tweety_can_fly") is True:
    print("RESULT: TRUE")
elif facts.get("tweety_can_fly") is False:
    print("RESULT: FALSE")
else:
    print("RESULT: UNCERTAIN")

EXAMPLE 2 (with rule chain):
Problem: Premises: The cat is rough. If something is rough then it likes the dog. Conclusion: The cat likes the dog.

facts = {}
facts["cat_is_rough"] = True
# Rule: if X is rough, then X likes dog
if facts.get("cat_is_rough") is True:
    facts["cat_likes_dog"] = True
# Check conclusion
if facts.get("cat_likes_dog") is True:
    print("RESULT: TRUE")
elif facts.get("cat_likes_dog") is False:
    print("RESULT: FALSE")
else:
    print("RESULT: UNCERTAIN")

EXAMPLE 3 (uncertain):
Problem: Premises: The dog is brown. Conclusion: The cat is green.

facts = {}
facts["dog_is_brown"] = True
# No information about cat being green
if facts.get("cat_is_green") is True:
    print("RESULT: TRUE")
elif facts.get("cat_is_green") is False:
    print("RESULT: FALSE")
else:
    print("RESULT: UNCERTAIN")"""


COT_SYSTEM_PROMPT = """You are a logical reasoning assistant. Given a logical reasoning problem with premises and a conclusion, determine whether the conclusion is True, False, or Uncertain.

Think through the problem step by step, then provide your final answer.

CRITICAL: Your response MUST end with EXACTLY ONE of these lines:
RESULT: TRUE
RESULT: FALSE
RESULT: UNCERTAIN"""


class LLMClient:
    """Handles communication with LLM APIs."""

    def __init__(self, provider, api_key, model, max_api_retries=3):
        self.provider = provider
        self.model = model
        self.max_api_retries = max_api_retries

        if provider == "gemini":
            self.client = genai.Client(api_key=api_key)
        elif provider == "deepseek" or provider == "openai":
            from openai import OpenAI # pyright: ignore[reportMissingImports]
            base_url = "https://api.deepseek.com" if provider == "deepseek" else None
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_llm(self, prompt, system_prompt=None):
        for attempt in range(self.max_api_retries):
            try:
                if self.provider == "gemini":
                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=full_prompt
                    )
                    return response.text
                else:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages
                    )
                    return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = 60 * (attempt + 1)
                    print(f"    Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception(f"Failed after {self.max_api_retries} API retries")

    def generate_prolog(self, problem):
        return self._call_llm(f"Problem: {problem}", PROLOG_SYSTEM_PROMPT)

    def generate_python(self, problem):
        return self._call_llm(f"Problem: {problem}", PYTHON_SYSTEM_PROMPT)

    def generate_cot(self, problem):
        return self._call_llm(f"Problem: {problem}", COT_SYSTEM_PROMPT)

    def refine_prolog(self, problem, failed_code, error_message):
        prompt = f"""The Prolog code below failed. Fix it.

Original problem: {problem}

Failed code:
{failed_code}

Error: {error_message}

CRITICAL FIXES:
1. For open-world reasoning, encode BOTH positive (is_X) and negative (not_X) facts
2. Always define fallback clauses: not_X(_) :- fail.
3. The query must check positive THEN negative THEN default to UNCERTAIN
4. Use catch(GOAL, _, fail) to handle undefined predicates gracefully

Return ONLY corrected Prolog code, no markdown, no explanation."""
        return self._call_llm(prompt, "You are a Prolog debugging expert.")

    def refine_python(self, problem, failed_code, error_message):
        prompt = f"""The Python code below failed. Fix it.

Original problem: {problem}

Failed code:
{failed_code}

Error: {error_message}

CRITICAL FIX - USE THIS STRUCTURE:
1. Create a single `facts = {{}}` dictionary
2. NEVER use standalone variables - always facts["key_name"]
3. Always use facts.get(key) for reading - never facts[key]
4. End with: if facts.get(conclusion) is True: print("RESULT: TRUE") elif facts.get(conclusion) is False: print("RESULT: FALSE") else: print("RESULT: UNCERTAIN")

Return ONLY corrected Python code, no markdown, no explanation."""
        return self._call_llm(prompt, "You are a Python debugging expert.")