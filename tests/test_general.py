import subprocess
import re
import sys
from pathlib import Path

def test_vle_small_end_gamma_below_one():
    """
    Acceptance test:
    - Runs `main.py` with the `vle-0.80-small.json` configuration.
    - Verifies the script exits successfully.
    - Parses the output for the final gamma value and asserts it's below 1.0.
    """

    # Locate the configuration file relative to the project root
    project_root = Path(__file__).parent.parent
    config_path = project_root / "vle-0.80-small.json"
    assert config_path.exists(), f"Config file not found: {config_path}"

    # Execute the simulation
    result = subprocess.run(
        [sys.executable, str(project_root / "main.py"), "-f", str(config_path)],
        capture_output=True,
        text=True
    )

    # Check that the script completed without errors
    assert result.returncode == 0, (
        f"Script exited with non-zero status {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    # Extract the final gamma value from stdout
    match = re.search(r"Final\s+gamma\s*[:=]\s*([0-9]*\.?[0-9]+)", result.stdout)
    assert match, "Final gamma not found in script output"

    gamma = float(match.group(1))
    # Assert gamma is below 1.0
    assert abs(gamma) < 1.0, f"Final gamma is {gamma}, expected < 1.0"
