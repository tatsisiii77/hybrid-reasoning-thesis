import subprocess
import tempfile
import os
import re

class PythonEngine:
    """Executes Python code via subprocess with timeout."""

    def execute(self, code, timeout=10):
        """
        Write Python code to temp file and run it.
        Returns: (success: bool, output: str, error: str or None)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name

        try:
            process = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if process.returncode != 0:
                return False, None, process.stderr.strip()

            output = process.stdout.strip()

            # Look for RESULT: TRUE/FALSE/UNCERTAIN in output
            match = re.search(r'RESULT:\s*(TRUE|FALSE|UNCERTAIN)', output, re.IGNORECASE)
            if match:
                result = match.group(1).upper()
                return result, output, None
            else:
                return False, output, "No RESULT: TRUE/FALSE/UNCERTAIN found in output"

        except subprocess.TimeoutExpired:
            return False, None, "Execution timed out (10s)"
        except Exception as e:
            return False, None, str(e)
        finally:
            os.unlink(temp_path)