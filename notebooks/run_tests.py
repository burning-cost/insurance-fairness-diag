# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-fairness-diag: Test Runner
# MAGIC
# MAGIC Runs the full pytest suite for insurance-fairness-diag on Databricks.

# COMMAND ----------

# MAGIC %pip install numpy>=1.24.0 scipy>=1.11.0 polars>=1.0.0 scikit-learn>=1.3.0 jinja2>=3.1.0 pytest>=7.4.0 pytest-cov>=4.1.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess
import sys
import os

# Install the package in editable mode from the uploaded workspace files
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-fairness-diag/"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

# COMMAND ----------

# Run pytest
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-fairness-diag/tests/",
        "-v",
        "--tb=short",
        "--no-header",
    ],
    capture_output=True,
    text=True,
    cwd="/Workspace/insurance-fairness-diag/"
)
print(result.stdout[-10000:] if len(result.stdout) > 10000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-3000:])
print("\nReturn code:", result.returncode)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo: Full audit on synthetic data

# COMMAND ----------

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from insurance_fairness_diag import ProxyDiscriminationAudit

# Generate synthetic data with known proxy structure
rng = np.random.default_rng(42)
n = 5000

age_band = rng.integers(0, 5, n).astype(float)
vehicle_group = rng.integers(0, 5, n).astype(float)
ncd_years = rng.integers(0, 5, n).astype(float)
postcode = rng.integers(0, 2, n).astype(float)  # sensitive: North/South

# proxy_feature is correlated with postcode (proxy discrimination source)
proxy_feature = 0.7 * postcode + np.sqrt(1 - 0.7**2) * rng.normal(0, 1, n)
proxy_feature = np.clip(
    np.digitize(proxy_feature, np.percentile(proxy_feature, [20, 40, 60, 80])), 0, 4
).astype(float)

# True fair price (no postcode effect)
y = 200 + 50 * age_band + 30 * vehicle_group + 20 * ncd_years + rng.normal(0, 40, n)

# Fit model WITHOUT postcode but WITH proxy_feature (creates proxy discrimination)
X_fit = np.column_stack([age_band, vehicle_group, ncd_years, proxy_feature])
model = Ridge(alpha=1.0)
model.fit(X_fit, y)

# Build DataFrame with ALL columns including sensitive
X = pl.DataFrame({
    "age_band": age_band,
    "vehicle_group": vehicle_group,
    "ncd_years": ncd_years,
    "proxy_feature": proxy_feature,
    "postcode_area": postcode,
})

# Run audit
audit = ProxyDiscriminationAudit(
    model=model,
    X=X,
    y=y,
    sensitive_col="postcode_area",
    rating_factors=["age_band", "vehicle_group", "ncd_years", "proxy_feature"],
    n_perms=128,
    subsample_n=2000,
    random_state=42,
)

result = audit.fit()
print(result.summary())

# COMMAND ----------

print(f"\nD_proxy = {result.d_proxy:.4f}")
print(f"95% CI: {result.d_proxy_ci}")
print(f"Monetary: £{result.d_proxy_monetary:.2f}")
print(f"RAG: {result.rag.upper()}")

print("\nShapley effects:")
for name, se in result.shapley_effects.items():
    print(f"  {se.rank}. {name}: phi={se.phi:.4f} (£{se.phi_monetary:.2f}) [{se.rag}]")

print(f"\nLocal scores preview:")
print(result.local_scores.head(10))

print(f"\nRAG distribution:")
print(result.local_scores.group_by("rag").agg(pl.len().alias("count")))

# COMMAND ----------

# Export reports
result.to_json("/tmp/proxy_audit.json")
result.to_html("/tmp/proxy_audit.html")
print("Reports written to /tmp/proxy_audit.json and /tmp/proxy_audit.html")

import json
with open("/tmp/proxy_audit.json") as f:
    report = json.load(f)
print(json.dumps(report, indent=2))
