from dataclasses import dataclass
import math
import random

import numpy as np

from phase0_5_bayes import (
    BayesParticle,
    particle_log_likelihood,
    sample_bayes_particles,
)
from phase0_5_controls import (
    generate_matched_control_dataset,
)
from physics import (
    NextTransition,
)


DEFAULT_ESS_PARTICLES = 20_000
DEFAULT_ESS_TEST_PAIRS = 100


@dataclass
class ObservationESS:
    pair_id: int
    side: str
    slip_ess: float
    yield_ess: float
    slip_ess_fraction: float
    yield_ess_fraction: float


@dataclass
class ESSSummary:
    n_particles: int
    n_particles_per_class: int
    n_test_rows: int

    slip_min: float
    slip_p05: float
    slip_median: float
    slip_mean: float

    yield_min: float
    yield_p05: float
    yield_median: float
    yield_mean: float

    slip_fraction_min: float
    slip_fraction_p05: float
    slip_fraction_median: float

    yield_fraction_min: float
    yield_fraction_p05: float
    yield_fraction_median: float


def effective_sample_size_from_log_weights(
    log_weights: np.ndarray,
) -> float:
    """
    Compute importance-sampling effective sample size from
    unnormalized log weights.

    ESS = (sum w)^2 / sum(w^2)

    The computation is stabilized by subtracting the maximum
    log weight before exponentiation.
    """

    log_weights = np.asarray(
        log_weights,
        dtype=float,
    )

    if log_weights.ndim != 1:
        raise ValueError(
            "log_weights must be one-dimensional"
        )

    if len(
        log_weights
    ) == 0:
        raise ValueError(
            "log_weights must not be empty"
        )

    if not np.all(
        np.isfinite(
            log_weights
        )
    ):
        raise ValueError(
            "log_weights must be finite"
        )

    maximum = float(
        np.max(
            log_weights
        )
    )

    weights = np.exp(
        log_weights
        - maximum
    )

    sum_weights = float(
        np.sum(
            weights
        )
    )

    sum_squared_weights = float(
        np.sum(
            weights
            * weights
        )
    )

    if (
        sum_weights <= 0.0
        or sum_squared_weights <= 0.0
    ):
        raise RuntimeError(
            "Invalid importance weights"
        )

    ess = (
        sum_weights
        * sum_weights
        / sum_squared_weights
    )

    return float(
        ess
    )


def observation_class_ess(
    record,
    particles: list[BayesParticle],
) -> ObservationESS:
    """
    Compute class-conditional ESS for one legal observation.

    SUPPORT_SLIP and MATERIAL_YIELD particles are evaluated
    separately because the Bayes posterior compares two
    class-conditional marginal likelihoods.
    """

    slip_logs = []
    yield_logs = []

    for particle in particles:
        log_likelihood = (
            particle_log_likelihood(
                record=record,
                particle=particle,
            )
        )

        if (
            particle.label
            == NextTransition.SUPPORT_SLIP.value
        ):
            slip_logs.append(
                log_likelihood
            )

        elif (
            particle.label
            == NextTransition.MATERIAL_YIELD.value
        ):
            yield_logs.append(
                log_likelihood
            )

        else:
            raise ValueError(
                f"Unknown particle label: "
                f"{particle.label}"
            )

    if not slip_logs:
        raise ValueError(
            "No SUPPORT_SLIP particles"
        )

    if not yield_logs:
        raise ValueError(
            "No MATERIAL_YIELD particles"
        )

    slip_ess = (
        effective_sample_size_from_log_weights(
            np.asarray(
                slip_logs,
                dtype=float,
            )
        )
    )

    yield_ess = (
        effective_sample_size_from_log_weights(
            np.asarray(
                yield_logs,
                dtype=float,
            )
        )
    )

    return ObservationESS(
        pair_id=record.pair_id,
        side=record.side,
        slip_ess=slip_ess,
        yield_ess=yield_ess,
        slip_ess_fraction=(
            slip_ess
            / len(
                slip_logs
            )
        ),
        yield_ess_fraction=(
            yield_ess
            / len(
                yield_logs
            )
        ),
    )


def summarize_ess(
    observations: list[ObservationESS],
    n_particles: int,
) -> ESSSummary:
    """
    Summarize class-conditional ESS over the fixed test set.
    """

    if not observations:
        raise ValueError(
            "observations must not be empty"
        )

    if (
        n_particles <= 0
        or n_particles % 2 != 0
    ):
        raise ValueError(
            "n_particles must be positive and even"
        )

    slip_ess = np.asarray(
        [
            item.slip_ess
            for item in observations
        ],
        dtype=float,
    )

    yield_ess = np.asarray(
        [
            item.yield_ess
            for item in observations
        ],
        dtype=float,
    )

    slip_fraction = np.asarray(
        [
            item.slip_ess_fraction
            for item in observations
        ],
        dtype=float,
    )

    yield_fraction = np.asarray(
        [
            item.yield_ess_fraction
            for item in observations
        ],
        dtype=float,
    )

    return ESSSummary(
        n_particles=n_particles,
        n_particles_per_class=(
            n_particles
            // 2
        ),
        n_test_rows=len(
            observations
        ),

        slip_min=float(
            np.min(
                slip_ess
            )
        ),
        slip_p05=float(
            np.percentile(
                slip_ess,
                5.0,
            )
        ),
        slip_median=float(
            np.median(
                slip_ess
            )
        ),
        slip_mean=float(
            np.mean(
                slip_ess
            )
        ),

        yield_min=float(
            np.min(
                yield_ess
            )
        ),
        yield_p05=float(
            np.percentile(
                yield_ess,
                5.0,
            )
        ),
        yield_median=float(
            np.median(
                yield_ess
            )
        ),
        yield_mean=float(
            np.mean(
                yield_ess
            )
        ),

        slip_fraction_min=float(
            np.min(
                slip_fraction
            )
        ),
        slip_fraction_p05=float(
            np.percentile(
                slip_fraction,
                5.0,
            )
        ),
        slip_fraction_median=float(
            np.median(
                slip_fraction
            )
        ),

        yield_fraction_min=float(
            np.min(
                yield_fraction
            )
        ),
        yield_fraction_p05=float(
            np.percentile(
                yield_fraction,
                5.0,
            )
        ),
        yield_fraction_median=float(
            np.median(
                yield_fraction
            )
        ),
    )


def run_bayes_ess_diagnostic(
    n_particles: int = DEFAULT_ESS_PARTICLES,
    n_test_pairs: int = DEFAULT_ESS_TEST_PAIRS,
    particle_seed: int = 123,
    test_seed: int = 999,
) -> ESSSummary:
    """
    Run the exploratory Phase 0.5 Bayes ESS diagnostic.

    The diagnostic uses the same fixed particle and test seeds
    as the convergence study.

    Results are exploratory only and are used to determine
    whether the Monte Carlo evidence is supported by an
    adequate number of effective particles.
    """

    if (
        n_particles <= 0
        or n_particles % 2 != 0
    ):
        raise ValueError(
            "n_particles must be positive and even"
        )

    if n_test_pairs <= 0:
        raise ValueError(
            "n_test_pairs must be positive"
        )

    particles = (
        sample_bayes_particles(
            n_particles=n_particles,
            seed=particle_seed,
        )
    )

    random.seed(
        test_seed
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=n_test_pairs,
        )
    )

    observations = [
        observation_class_ess(
            record=record,
            particles=particles,
        )
        for record in (
            dataset.compliant_records
        )
    ]

    return summarize_ess(
        observations=observations,
        n_particles=n_particles,
    )


def main():
    result = (
        run_bayes_ess_diagnostic()
    )

    print(
        "=== Phase 0.5 Bayes ESS diagnostic ==="
    )

    print(
        "particles=",
        result.n_particles,
    )

    print(
        "particles_per_class=",
        result.n_particles_per_class,
    )

    print(
        "test_rows=",
        result.n_test_rows,
    )

    print()

    print(
        "SUPPORT_SLIP ESS"
    )

    print(
        "min=",
        round(
            result.slip_min,
            2,
        ),
    )

    print(
        "p05=",
        round(
            result.slip_p05,
            2,
        ),
    )

    print(
        "median=",
        round(
            result.slip_median,
            2,
        ),
    )

    print(
        "mean=",
        round(
            result.slip_mean,
            2,
        ),
    )

    print(
        "fraction_min=",
        round(
            result.slip_fraction_min,
            6,
        ),
    )

    print(
        "fraction_p05=",
        round(
            result.slip_fraction_p05,
            6,
        ),
    )

    print(
        "fraction_median=",
        round(
            result.slip_fraction_median,
            6,
        ),
    )

    print()

    print(
        "MATERIAL_YIELD ESS"
    )

    print(
        "min=",
        round(
            result.yield_min,
            2,
        ),
    )

    print(
        "p05=",
        round(
            result.yield_p05,
            2,
        ),
    )

    print(
        "median=",
        round(
            result.yield_median,
            2,
        ),
    )

    print(
        "mean=",
        round(
            result.yield_mean,
            2,
        ),
    )

    print(
        "fraction_min=",
        round(
            result.yield_fraction_min,
            6,
        ),
    )

    print(
        "fraction_p05=",
        round(
            result.yield_fraction_p05,
            6,
        ),
    )

    print(
        "fraction_median=",
        round(
            result.yield_fraction_median,
            6,
        ),
    )

    print()

    print(
        "STATUS: exploratory ESS diagnostic only"
    )


if __name__ == "__main__":
    main()