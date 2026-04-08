from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
import os
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

problem = """
Ο Γιώργος είναι σεφ. Όλοι οι σεφ είναι μάγειρες. 
Είναι ο Γιώργος μάγειρας;
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"""You are a logic programming assistant. Given a logical reasoning problem in natural language, convert it to valid SWI-Prolog code. Return ONLY the Prolog code, nothing else. The code must include all facts, rules, and a query. Format the query as: ?- query.

Problem: {problem}"""
)

prolog_code = response.text
print("=== Generated Prolog Code ===")
print(prolog_code)