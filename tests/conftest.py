"""
Shared fixtures for insurance-fairness-diag tests.

Generates synthetic insurance datasets with known proxy discrimination
structure so that tests have analytical ground truth.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder


def make_synthetic_dataset(
    n: int = 2000,
    proxy_strength: float = 0.3,
    random_state: int = 42,
) -> tuple[pl.DataFrame, np.ndarray, object]:
    """
    Generate a synthetic insurance pricing dataset with known proxy structure.

    The dataset has:
      - age_band: legitimate rating factor (0..4)
      - vehicle_group: legitimate rating factor (0..4)
      - ncd_years: legitimate rating factor (0..4)
      - postcode_area: SENSITIVE attribute (binary: 0 or 1)
      - proxy_feature: feature that is correlated with postcode_area
        by *proxy_strength* (this creates proxy discrimination)

    The true price is:
      true_price = 200 + 50*age_band + 30*vehicle_group + 20*ncd_years

    The model is fitted WITHOUT postcode_area, but WITH proxy_feature,
    which introduces proxy discrimination.

    Parameters
    ----------
    n:
        Number of policyholders.
    proxy_strength:
        Correlation between proxy_feature and postcode_area (0 = no proxy).
    random_state:
        Random seed.

    Returns
    -------
    (X, h, model) where:
      X is the Polars feature DataFrame (includes all columns)
      h is the fitted premium array
      model is the fitted sklearn Ridge model
    """
    rng = np.random.default_rng(random_state)

    # Sensitive attribute: binary postcode_area (0 = North, 1 = South)
    postcode = rng.integers(0, 2, size=n).astype(float)

    # Legitimate factors
    age_band = rng.integers(0, 5, size=n).astype(float)
    vehicle_group = rng.integers(0, 5, size=n).astype(float)
    ncd_years = rng.integers(0, 5, size=n).astype(float)

    # Proxy feature: correlated with postcode but not identical
    noise = rng.normal(0, 1, size=n)
    proxy_feature = proxy_strength * postcode + np.sqrt(1 - proxy_strength**2) * noise
    # Discretise to 5 bands
    proxy_feature = np.clip(
        np.digitize(proxy_feature, np.percentile(proxy_feature, [20, 40, 60, 80])),
        0, 4
    ).astype(float)

    # True price (no proxy discrimination)
    true_price = 200.0 + 50.0 * age_band + 30.0 * vehicle_group + 20.0 * ncd_years

    # Observed claims with noise
    y = true_price + rng.normal(0, 50, size=n)

    # Feature matrix for model: does NOT include postcode_area, but includes proxy_feature
    X_model = np.column_stack([age_band, vehicle_group, ncd_years, proxy_feature])

    # Fit a Ridge model
    model = Ridge(alpha=1.0)
    model.fit(X_model, y)
    h = model.predict(X_model)

    # Build Polars DataFrame with ALL columns (including sensitive)
    X = pl.DataFrame({
        "age_band": age_band,
        "vehicle_group": vehicle_group,
        "ncd_years": ncd_years,
        "proxy_feature": proxy_feature,
        "postcode_area": postcode,
    })

    return X, h, model


def make_zero_proxy_dataset(
    n: int = 1000,
    random_state: int = 42,
) -> tuple[pl.DataFrame, np.ndarray, object]:
    """
    Generate a dataset where the model has NO proxy discrimination.

    The proxy_feature is independent of postcode_area (proxy_strength=0).
    D_proxy should be approximately 0 in this case.
    """
    return make_synthetic_dataset(n=n, proxy_strength=0.0, random_state=random_state)


def make_high_proxy_dataset(
    n: int = 2000,
    random_state: int = 42,
) -> tuple[pl.DataFrame, np.ndarray, object]:
    """
    Generate a dataset with strong proxy discrimination.

    proxy_feature is strongly correlated (0.8) with postcode_area.
    D_proxy should be clearly above 0.
    """
    return make_synthetic_dataset(n=n, proxy_strength=0.8, random_state=random_state)


@pytest.fixture
def synthetic_data():
    """Standard synthetic dataset with moderate proxy discrimination."""
    return make_synthetic_dataset(n=2000, proxy_strength=0.3, random_state=42)


@pytest.fixture
def zero_proxy_data():
    """Synthetic dataset with zero proxy discrimination."""
    return make_zero_proxy_dataset(n=1000, random_state=42)


@pytest.fixture
def high_proxy_data():
    """Synthetic dataset with strong proxy discrimination."""
    return make_high_proxy_dataset(n=2000, proxy_strength=0.8, random_state=42)


@pytest.fixture
def simple_model_and_data():
    """
    Very simple analytical case for testing admissible price computation.

    Two groups (S=0, S=1) with equal sizes. Model predicts:
      - Group 0: h = 100 for all
      - Group 1: h = 200 for all

    h_star = 150 for all (mean across groups).
    D_proxy = sqrt(mean(50^2)) / sqrt(mean(150^2)) = 50/150 = 1/3.
    """
    n = 100
    s = np.array([0] * 50 + [1] * 50, dtype=float)
    h = np.array([100.0] * 50 + [200.0] * 50)
    weights = np.ones(n)

    X = pl.DataFrame({
        "factor_a": np.random.default_rng(0).integers(0, 5, size=n).astype(float),
        "postcode": s,
    })

    return h, s, weights, X
