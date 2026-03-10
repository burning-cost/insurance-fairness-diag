# Databricks notebook source

# COMMAND ----------

import subprocess
import sys
import os
import shutil

# Install dependencies
r1 = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "numpy>=1.24.0", "scipy>=1.11.0", "polars>=1.0.0",
     "scikit-learn>=1.3.0", "jinja2>=3.1.0", "pytest>=7.4.0"],
    capture_output=True, text=True
)
r2 = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e",
     "/Workspace/insurance-fairness-diag/"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-fairness-diag/"
)

# Copy tests to /tmp to avoid Workspace pycache issues
test_src = "/Workspace/insurance-fairness-diag/tests"
test_dst = "/tmp/ifd_tests"
if os.path.exists(test_dst):
    shutil.rmtree(test_dst)
shutil.copytree(test_src, test_dst)

env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"

# Run tests - show only failures (short output)
run = subprocess.run(
    [sys.executable, "-m", "pytest", test_dst,
     "--tb=long",  # full tracebacks for failures
     "--no-header",
     "-q"],  # quiet: only show failures + summary
    capture_output=True, text=True,
    cwd="/tmp",
    env=env,
)
run_out = run.stdout + run.stderr

# Split: show last 7000 chars (summary + failures)
if len(run_out) > 7000:
    display = run_out[:2000] + "\n...[middle omitted]...\n" + run_out[-5000:]
else:
    display = run_out

dbutils.notebook.exit(f"rc={run.returncode}\n\n{display}"[:4096])
