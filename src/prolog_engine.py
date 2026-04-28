import tempfile
import os
import subprocess

class PrologEngine:
    """Executes Prolog code via SWI-Prolog subprocess."""

    def execute(self, program, query=None, timeout=10):
        """
        Run Prolog code with directives.
        Returns: (success: bool, output: str, error: str or None)
        """
        wrapper = """
:- set_prolog_flag(unknown, fail).
:- use_module(library(error)).
"""
        full_code = wrapper + program + "\n:- halt.\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False, encoding='utf-8') as f:
            f.write(full_code)
            temp_path = f.name

        try:
            process = subprocess.run(
                ["swipl", "-q", "-f", temp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = process.stdout.strip()
            errors = process.stderr.strip()

            if "RESULT:" in output:
                return True, output, None

            if errors:
                return False, output, errors[:500]  # Truncate long errors

            return True, "RESULT: UNCERTAIN", None

        except subprocess.TimeoutExpired:
            return False, None, "Prolog execution timed out (10s)"
        except Exception as e:
            return False, None, str(e)
        finally:
            os.unlink(temp_path)