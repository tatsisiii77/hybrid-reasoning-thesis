from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
import os
import re
import tempfile
from google import genai
from pyswip import Prolog # pyright: ignore[reportMissingImports]

load_dotenv()

# Step 1: Send problem to LLM
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

problem = """
Αν το σύστημα ασφαλείας ανιχνεύσει κίνηση στο κτίριο, τότε κλειδώνουν αυτόματα όλες οι εξωτερικές πόρτες. Αν κλειδώσουν οι εξωτερικές πόρτες, τότε ακούγεται η κεντρική σειρήνα. Ο φύλακας παρατηρεί ότι η κεντρική σειρήνα δεν χτυπάει. Μπορούμε να γνωρίζουμε με βεβαιότητα αν το σύστημα έχει ανιχνεύσει κίνηση στο κτίριο;
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"""You are a logic programming assistant. Given a logical reasoning problem in natural language, convert it to valid SWI-Prolog code.

Return ONLY the Prolog code, no markdown, no backticks, no explanation.
The code must include facts, rules, and a query on the last line starting with ?-

Problem: {problem}"""
)

prolog_code = response.text.strip()

# Clean markdown formatting if present
prolog_code = re.sub(r'```prolog\s*', '', prolog_code)
prolog_code = re.sub(r'```\s*', '', prolog_code)
prolog_code = prolog_code.strip()

print("=== Generated Prolog Code ===")
print(prolog_code)
print()

# Step 2: Split code into program and query
lines = prolog_code.split('\n')
query_line = None
program_lines = []

for line in lines:
    stripped = line.strip()
    if stripped.startswith('?-'):
        query_line = stripped[2:].strip().rstrip('.')
    elif stripped:
        program_lines.append(line)

program_code = '\n'.join(program_lines)

print(f"=== Query: {query_line} ===")
print()

# Step 3: Write Prolog code to temp file and consult it
prolog = Prolog()

with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False, encoding='utf-8') as f:
    f.write(program_code)
    temp_path = f.name

try:
    prolog.consult(temp_path)
    
    # Run query with timeout
    results = list(prolog.query(query_line, maxresult=1))
    if results:
        print("=== Prolog Result: TRUE ===")
        if results[0]:
            print(f"Bindings: {results}")
    else:
        print("=== Prolog Result: FALSE ===")
except Exception as e:
    print(f"Error: {e}")
finally:
    os.unlink(temp_path)