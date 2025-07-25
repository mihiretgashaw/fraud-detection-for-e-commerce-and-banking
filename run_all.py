import os
import subprocess
from pathlib import Path

# Run all Python scripts in src/
src_dir = Path("src")
print("🔁 Running Python scripts in src/")
for script in src_dir.glob("*.py"):
    print(f"▶️ Running {script}")
    subprocess.run(["python", str(script)], check=True)

# Run all notebooks in notebooks/
notebooks_dir = Path("notebooks")
output_dir = Path("executed_notebooks")
output_dir.mkdir(exist_ok=True)

print("📘 Executing Jupyter notebooks in notebooks/")
for notebook in notebooks_dir.glob("*.ipynb"):
    print(f"▶️ Executing {notebook}")
    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", str(notebook),
        "--output-dir", str(output_dir),
        "--inplace"
    ], check=True)

print("✅ All scripts and notebooks executed successfully.")
