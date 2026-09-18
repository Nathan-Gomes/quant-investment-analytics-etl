"""Risk models: how the covariance an optimizer consumes is built.

Three of them, with the same interface, because the choice is a research decision
rather than an implementation detail and belongs in the open.

``sample``               the raw sample covariance. The baseline everything else
                         is measured against, and unusable on its own once the
                         number of assets approaches the number of observations.
``ledoit_wolf``          shrinkage toward a scaled identity. A good default at a
                         few dozen assets.
``statistical_factor``   a principal-component factor model, ``Σ = BBᵀ + D``.

The factor model is what makes a large universe tractable. A 500-name sample
covariance has 125,250 free parameters; two years of daily data supplies about
500 observations per name, so most of the matrix is noise, and its smallest
eigenvalues — the directions a variance minimizer loads into hardest — are the
least trustworthy part of it. Reducing to a handful of factors plus specific
variance replaces that with something estimable, and a quadratic form that costs
``k`` inner products rather than an ``n × n`` multiply.

The number of factors is not a knob tuned until the results look good. It is the
count of eigenvalues that sit above the Marchenko-Pastur upper edge — the largest
eigenvalue a correlation matrix of pure noise with the same shape would be
expected to produce. Components below it are indistinguishable from randomness.

What this is not: a vendor fundamental risk model. Barra, Axioma and MSCI build
factors from fundamental and industry data with decades of point-in-time history
behind them. This one derives its factors from the return covariance itself,
which needs no licensed data and is honest about what it is — statistical
factors have no economic name, and their interpretation is left to whoever reads
the attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRADING_DAYS = 252
MODELS = ("sample", "ledoit_wolf", "statistical_factor")


@dataclass
class RiskModel:
    """An estimated covariance and the structure it was built from."""

    covariance: np.ndarray
    kind: str
    observations: int
    shrinkage_intensity: float = 0.0
    exposures: np.ndarray | None = None       # B, assets x factors
    specific_variance: np.ndarray | None = None  # diagonal of D
    factor_variance: np.ndarray | None = None    # variance carried by each factor
    explained: float = 0.0
    noise_edge: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    @property
    def assets(self) -> int:
        return int(self.covariance.shape[0])

    @property
    def factors(self) -> int:
        return 0 if self.exposures is None else int(self.exposures.shape[1])

    def volatility(self, weights: np.ndarray) -> float:
        weights = np.asarray(weights, dtype=float)
        return float(np.sqrt(max(weights @ self.covariance @ weights, 0.0)))

    def condition_number(self) -> float:
        """How close the matrix is to singular.

        A large value means the optimizer is being invited to take a large
        position in a direction the data barely determines, which is the
        mechanism behind most unstable "optimal" portfolios.
        """
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        smallest = max(eigenvalues.min(), 1e-18)
        return float(eigenvalues.max() / smallest)

    def attribution(self, weights: np.ndarray) -> dict:
        """Split portfolio variance into factor and specific parts.

        Without a factor structure there is nothing to split, so the whole of it
        is reported as specific rather than invented.
        """
        weights = np.asarray(weights, dtype=float)
        total = float(weights @ self.covariance @ weights)
        if self.exposures is None or self.specific_variance is None:
            return {"total_variance": total, "factor_variance": 0.0,
                    "specific_variance": total, "factor_share": 0.0, "by_factor": []}
        factor_exposure = self.exposures.T @ weights
        by_factor = factor_exposure ** 2
        factor_total = float(by_factor.sum())
        specific = float((weights ** 2) @ self.specific_variance)
        return {
            "total_variance": total,
            "factor_variance": factor_total,
            "specific_variance": specific,
            "factor_share": factor_total / total if total > 0 else 0.0,
            "by_factor": [
                {"factor": index + 1, "exposure": float(factor_exposure[index]),
                 "variance": float(by_factor[index]),
                 "share": float(by_factor[index] / total) if total > 0 else 0.0}
                for index in range(len(by_factor))
            ],
        }


def sample_covariance(returns: np.ndarray, annualize: bool = True) -> np.ndarray:
    if returns.ndim != 2 or returns.shape[0] < 2:
        raise ValueError("Covariance needs at least two observations.")
    covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
    return covariance * (TRADING_DAYS if annualize else 1)


def ledoit_wolf_covariance(returns: np.ndarray, annualize: bool = True) -> tuple[np.ndarray, float]:
    """Ledoit and Wolf (2004) shrinkage toward a scaled identity."""
    observations, assets = returns.shape
    if observations < 2:
        raise ValueError("Covariance needs at least two observations.")
    centred = returns - returns.mean(axis=0)
    sample = centred.T @ centred / observations

    mu = np.trace(sample) / assets
    target = mu * np.eye(assets)
    dispersion = np.sum((sample - target) ** 2) / assets

    # Mean squared error of the sample entries, from the fourth moments. Written
    # as one contraction rather than a loop so it survives a large universe.
    squared = centred ** 2
    error = float(np.sum(squared.T @ squared) / observations - np.sum(sample ** 2))
    error = max(error, 0.0) / (observations * assets)
    error = min(error, dispersion)

    intensity = 0.0 if dispersion <= 0 else float(np.clip(error / dispersion, 0.0, 1.0))
    shrunk = intensity * target + (1 - intensity) * sample
    shrunk = shrunk * observations / (observations - 1)
    return shrunk * (TRADING_DAYS if annualize else 1), intensity


def marchenko_pastur_edge(observations: int, assets: int) -> float:
    """The largest eigenvalue expected from a correlation matrix of pure noise."""
    if observations <= 0:
        return 1.0
    ratio = assets / observations
    return float((1 + np.sqrt(ratio)) ** 2)


def statistical_factor_model(returns: np.ndarray, factors: int | None = None,
                             annualize: bool = True) -> RiskModel:
    """Principal-component factor model: ``Σ = BBᵀ + D``.

    Built on the correlation matrix so that a single volatile name cannot
    dominate the first component purely by scale, then returned to covariance
    units. The result is positive definite by construction, since the specific
    variances are floored above zero.
    """
    observations, assets = returns.shape
    if observations < 20:
        raise ValueError("A factor model needs at least 20 observations.")
    centred = returns - returns.mean(axis=0)
    deviation = centred.std(axis=0, ddof=1)
    deviation = np.where(deviation > 0, deviation, 1e-12)
    correlation = np.corrcoef(centred, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    edge = marchenko_pastur_edge(observations, assets)
    if factors is None:
        # Components above the noise band, always at least one and never so many
        # that the model stops being a reduction.
        above = int((eigenvalues > edge).sum())
        factors = int(np.clip(above, 1, max(1, min(assets - 1, observations // 10, 20))))
    factors = int(np.clip(factors, 1, min(assets - 1, observations - 1)))

    kept = np.maximum(eigenvalues[:factors], 1e-12)
    loadings = eigenvectors[:, :factors] * np.sqrt(kept)        # correlation units
    communality = np.sum(loadings ** 2, axis=1)
    # A name fully explained by the factors would have zero idiosyncratic risk,
    # which is never true and would make the matrix singular.
    specific = np.clip(1.0 - communality, 1e-6, None)

    scale = deviation * (np.sqrt(TRADING_DAYS) if annualize else 1.0)
    exposures = loadings * scale[:, None]
    specific_variance = specific * scale ** 2
    covariance = exposures @ exposures.T + np.diag(specific_variance)

    return RiskModel(
        covariance=covariance,
        kind="statistical_factor",
        observations=int(observations),
        exposures=exposures,
        specific_variance=specific_variance,
        factor_variance=(kept * np.mean(scale ** 2)),
        explained=float(kept.sum() / eigenvalues.sum()),
        noise_edge=edge,
        diagnostics={
            "factors": factors,
            "eigenvalues_above_noise": int((eigenvalues > edge).sum()),
            "leading_eigenvalues": [float(v) for v in eigenvalues[:min(8, assets)]],
        },
    )


def build(returns: np.ndarray, kind: str = "ledoit_wolf", factors: int | None = None) -> RiskModel:
    """The one entry point the rest of the system uses."""
    if kind not in MODELS:
        raise ValueError(f"Risk model must be one of {', '.join(MODELS)}.")
    returns = np.asarray(returns, dtype=float)
    if kind == "sample":
        covariance = sample_covariance(returns)
        model = RiskModel(covariance=covariance, kind=kind, observations=int(returns.shape[0]))
    elif kind == "ledoit_wolf":
        covariance, intensity = ledoit_wolf_covariance(returns)
        model = RiskModel(covariance=covariance, kind=kind, observations=int(returns.shape[0]),
                          shrinkage_intensity=intensity)
    else:
        model = statistical_factor_model(returns, factors=factors)
    model.diagnostics.setdefault("condition_number", model.condition_number())
    return model


def report(model: RiskModel, weights: np.ndarray | None = None) -> dict:
    """What the interface and the payload show about the risk model."""
    payload = {
        "kind": model.kind,
        "assets": model.assets,
        "observations": model.observations,
        "shrinkage_intensity": model.shrinkage_intensity,
        "factors": model.factors,
        "explained_variance": model.explained,
        "noise_edge": model.noise_edge,
        "condition_number": model.diagnostics.get("condition_number", model.condition_number()),
    }
    if weights is not None:
        payload["attribution"] = model.attribution(weights)
    return payload
