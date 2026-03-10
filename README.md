# insurance-fairness-diag

Proxy discrimination diagnostics for UK personal lines insurance pricing models.

## The problem

Under the Equality Act 2010 s.19 and FCA Consumer Duty (PRIN 2A.4), insurers cannot use pricing models that produce indirect discrimination against protected characteristics — even unintentionally. The challenge is that an "unaware" model (one that doesn't use gender, ethnicity, etc. directly) can still discriminate through proxy variables: postcode correlates with ethnicity, occupation correlates with gender, and so on.

Knowing that proxy discrimination *might* exist isn't enough. A pricing actuary needs to answer:

- **How much** proxy discrimination is in the current model? Is it material?
- **Which rating factors** are driving it? Is it postcode, vehicle group, occupation, or something else?
- **Who is affected?** Which policyholders are paying more or less than the discrimination-free counterfactual?

This library answers all three questions with a single `fit()` call.

## What it computes

**D_proxy** — the normalised L2-distance from the fitted price to the admissible (discrimination-free) price set. Scalar 0–1, where 0 means no proxy discrimination and values above 0.15 warrant investigation. Based on Lindholm, Richman, Tsanakas and Wüthrich (2026) EJOR, SSRN 4897265.

**Shapley effects** — which rating factors drive proxy discrimination, using the Owen (2014) random permutation estimator. Implemented natively (SALib does not have this variant). Results tell you that, say, "postcode district accounts for 62% of discrimination variance, followed by vehicle group at 24%."

**Per-policyholder proxy vulnerability** — monetary impact of proxy discrimination per individual, using the unaware vs aware premium benchmark from Côté, Côté and Charpentier (2025). Positive means the policyholder pays more than the fair counterfactual.

**RAG status** — traffic-light summary for regulatory documentation: green (D_proxy < 0.05), amber (0.05–0.15), red (> 0.15).

## Quick start

```python
import polars as pl
from insurance_fairness_diag import ProxyDiscriminationAudit

audit = ProxyDiscriminationAudit(
    model=my_glm,           # fitted sklearn/CatBoost model, WITHOUT sensitive_col
    X=df,                   # Polars DataFrame, INCLUDING sensitive_col
    y=df["claim_cost"],     # observed outcomes
    sensitive_col="postcode_area",
    rating_factors=["age_band", "vehicle_group", "ncd_years", "occupation"],
    exposure_col="exposure",
    n_perms=256,            # Owen permutations; 256 is good for up to ~20 factors
)

result = audit.fit()

print(result.summary())
# Proxy Discrimination Audit
#   Sensitive attribute : postcode_area
#   D_proxy             : 0.0923 (95% CI: [0.0811, 0.1035])
#   D_proxy monetary    : £18.46
#   RAG status          : AMBER
#
#   Top discriminatory factors:
#     1. vehicle_group: phi=0.4210 (£7.76) [red]
#     2. occupation: phi=0.3102 (£5.72) [red]
#     3. age_band: phi=0.1844 (£3.40) [amber]
#     4. ncd_years: phi=0.0844 (£1.56) [amber]

result.to_html("proxy_discrimination_audit.html")
result.to_json("proxy_discrimination_audit.json")
```

## API reference

### `ProxyDiscriminationAudit`

```python
ProxyDiscriminationAudit(
    model,            # fitted model with predict(X) method
    X,                # pl.DataFrame with all features + sensitive_col
    y,                # observed outcomes (np.ndarray or pl.Series)
    sensitive_col,    # e.g. "postcode_area", "gender", "ethnicity_proxy"
    rating_factors,   # list of legitimate factor names (not sensitive_col)
    exposure_col=None,
    reference_dist="observed",
    n_perms=256,
    surrogate_model=None,  # pre-fitted surrogate (advanced)
    subsample_n=10_000,
    random_state=42,
)
```

### `ProxyDiscriminationResult`

Returned by `audit.fit()`.

| Attribute | Type | Description |
|---|---|---|
| `d_proxy` | float | Normalised L2-distance, 0–1 |
| `d_proxy_ci` | (float, float) | Bootstrap 95% CI |
| `d_proxy_monetary` | float | d_proxy × mean premium (£) |
| `shapley_effects` | dict[str, ShapleyEffect] | Per-factor attribution |
| `local_scores` | pl.DataFrame | Per-policyholder scores |
| `benchmarks` | BenchmarkPremiums | Unaware/aware premiums |
| `rag` | str | "green" / "amber" / "red" |

## Methodology

**Admissible price (h*):** For each policyholder, h* is the exposure-weighted mean prediction within their sensitive group. This approximates E[h(X) | X_legitimate], marginalising out S-correlated variation. See Lindholm et al. (2022) ASTIN Bulletin.

**D_proxy formula:** `sqrt(E_w[(h - h*)^2]) / sqrt(E_w[h^2])`. Normalised so that D_proxy = 0 iff h = h* everywhere, and D_proxy < 1 always (assuming positive predictions).

**Shapley effects:** Owen (2014) random permutation estimator. Characteristic function `v(S) = Var[E[D | X_S]]` where D = h - h* is the discrimination residual. A surrogate RandomForestRegressor models D ~ f(X) for fast evaluation. n_perms=256 gives good accuracy for up to ~20 factors.

**Aware premium:** The model is refitted with S added as a feature (distillation: target = unaware predictions), then S is marginalised by averaging predictions over the empirical distribution of S.

## Regulatory alignment

This library supports audit documentation under:
- Equality Act 2010, s.19 (Indirect Discrimination)
- FCA PRIN 2A.4 (Consumer Duty — Fair Value)
- FCA FG22/5 paras 8.8–8.12 (Pricing Practices Guidance)

The HTML report generated by `result.to_html()` is structured as an audit trail with explicit regulatory references.

## Design choices

**No dependency on insurance-fairness.** This library is fully self-contained. The admissible price computation is re-implemented here rather than imported, keeping the library usable without the broader insurance-fairness stack.

**Polars throughout.** Input DataFrames are Polars. Internal computation uses numpy arrays. No pandas dependency.

**Native Owen 2014 implementation.** SALib implements Shapley sensitivity indices for variance-based GSA (Gamboa et al.), not the Owen (2014) estimator for model output variance decomposition. We implement Owen 2014 natively.

**Surrogate model approach.** The Shapley computation operates on a RandomForestRegressor fitted to the discrimination residual D = h - h*, not on the original pricing model. This makes the computation model-agnostic and fast.

## References

- Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance Pricing. *ASTIN Bulletin* 52(1), 55–89.
- Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of Proxy Discrimination. *EJOR*. SSRN 4897265.
- Owen, A.B. (2014). Sobol' indices and Shapley value. *SIAM/ASA JUQ* 2(1), 245–251.
- Côté, Côté, Charpentier (2025). Five premium benchmarks for proxy discrimination in insurance pricing.
- Biessy, G. (2024). Revisiting the Discrimination-Free Principle Through Shapley Values. *ASTIN Bulletin*.

## Installation

```bash
pip install insurance-fairness-diag
```

## Licence

MIT. See [LICENSE](LICENSE).

---

Built by [Burning Cost](https://github.com/burning-cost). Practitioner tools for UK insurance pricing teams.
