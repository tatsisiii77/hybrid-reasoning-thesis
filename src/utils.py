import re

def clean_prolog_code(raw_code):
    code = raw_code.strip()
    code = re.sub(r'```prolog\s*', '', code)
    code = re.sub(r'```\s*', '', code)
    return code.strip()

def clean_python_code(raw_code):
    code = raw_code.strip()
    code = re.sub(r'```python\s*', '', code)
    code = re.sub(r'```\s*', '', code)
    return code.strip()

def split_program_and_query(prolog_code):
    """
    Split Prolog code into program and query.
    With the new prompts, the query is a directive (:-) that runs automatically.
    So we return the entire code as 'program' and None as query.
    """
    return prolog_code, None

def extract_result(text):
    if text is None:
        return None
    match = re.search(r'RESULT:\s*(TRUE|FALSE|UNCERTAIN)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def normalize_label(label):
    label = label.strip().upper()
    if label == "TRUE":
        return "TRUE"
    elif label == "FALSE":
        return "FALSE"
    else:
        return "UNCERTAIN"