from dataclasses import dataclass
import random

from phase0_5_safe_cohort import (
    SafePairSample,
    generate_safe_pair_sample,
)
from physics import (
    Observation,
    run_safe_probe,
)


DEFAULT_FORCE_NOISE_STD = 0.20
DEFAULT_MOTION_NOISE_STD = 0.0002


@dataclass
class ObservationNoise:
    fx: float
    fy: float
    tip_dx: float
    tip_dy: float


@dataclass
class ObservationRecord:
    pair_id: int
    side: str
    label: str
    fx: float
    fy: float
    tip_dx: float
    tip_dy: float
    rho: float
    margin: float


@dataclass
class PairObservation:
    sample: SafePairSample
    noise: ObservationNoise
    record_a: ObservationRecord
    record_b: ObservationRecord


def sample_shared_noise(
    force_noise_std: float = DEFAULT_FORCE_NOISE_STD,
    motion_noise_std: float = DEFAULT_MOTION_NOISE_STD,
) -> ObservationNoise:
    """
    Sample one shared measurement-noise realization for a
    matched twin pair.

    Both twins receive the same additive noise offsets.

    This preserves the paired design and prevents independent
    measurement noise from becoming an accidental source of
    between-twin differences.
    """

    if force_noise_std < 0:
        raise ValueError(
            "force_noise_std must be nonnegative"
        )

    if motion_noise_std < 0:
        raise ValueError(
            "motion_noise_std must be nonnegative"
        )

    return ObservationNoise(
        fx=random.gauss(
            0.0,
            force_noise_std,
        ),
        fy=random.gauss(
            0.0,
            force_noise_std,
        ),
        tip_dx=random.gauss(
            0.0,
            motion_noise_std,
        ),
        tip_dy=random.gauss(
            0.0,
            motion_noise_std,
        ),
    )


def add_observation_noise(
    observation: Observation,
    noise: ObservationNoise,
) -> Observation:
    """
    Apply an explicit additive noise realization.
    """

    return Observation(
        fx=(
            observation.fx
            + noise.fx
        ),
        fy=(
            observation.fy
            + noise.fy
        ),
        tip_dx=(
            observation.tip_dx
            + noise.tip_dx
        ),
        tip_dy=(
            observation.tip_dy
            + noise.tip_dy
        ),
    )


def observe_safe_pair(
    sample: SafePairSample,
    noise: ObservationNoise | None = None,
    force_noise_std: float = DEFAULT_FORCE_NOISE_STD,
    motion_noise_std: float = DEFAULT_MOTION_NOISE_STD,
) -> PairObservation:
    """
    Observe both twins under exactly the same safe action.

    Physics is evaluated noiselessly first.

    The same measurement-noise realization is then added to
    both twins.
    """

    if noise is None:
        noise = (
            sample_shared_noise(
                force_noise_std=force_noise_std,
                motion_noise_std=motion_noise_std,
            )
        )

    result_a = run_safe_probe(
        params=sample.pair.world_a,
        action=sample.action,
        support_model="compliant",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    result_b = run_safe_probe(
        params=sample.pair.world_b,
        action=sample.action,
        support_model="compliant",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    if not (
        result_a.still_blocked
        and result_b.still_blocked
    ):
        raise RuntimeError(
            "Safe cohort invariant violated: "
            "one or both twins transitioned"
        )

    noisy_a = (
        add_observation_noise(
            result_a.observation,
            noise,
        )
    )

    noisy_b = (
        add_observation_noise(
            result_b.observation,
            noise,
        )
    )

    record_a = ObservationRecord(
        pair_id=sample.pair.pair_id,
        side="A",
        label=sample.pair.label_a.value,
        fx=noisy_a.fx,
        fy=noisy_a.fy,
        tip_dx=noisy_a.tip_dx,
        tip_dy=noisy_a.tip_dy,
        rho=sample.rho_a,
        margin=sample.margin_a,
    )

    record_b = ObservationRecord(
        pair_id=sample.pair.pair_id,
        side="B",
        label=sample.pair.label_b.value,
        fx=noisy_b.fx,
        fy=noisy_b.fy,
        tip_dx=noisy_b.tip_dx,
        tip_dy=noisy_b.tip_dy,
        rho=sample.rho_b,
        margin=sample.margin_b,
    )

    return PairObservation(
        sample=sample,
        noise=noise,
        record_a=record_a,
        record_b=record_b,
    )


def generate_observation_dataset(
    n_pairs: int,
) -> list[
    ObservationRecord
]:
    """
    Generate a paired Phase 0.5 observation dataset.

    Each pair contributes exactly two rows.
    """

    if n_pairs <= 0:
        raise ValueError(
            "n_pairs must be positive"
        )

    records = []

    for pair_id in range(
        n_pairs
    ):
        sample = (
            generate_safe_pair_sample(
                pair_id=pair_id,
            )
        )

        observed = (
            observe_safe_pair(
                sample
            )
        )

        records.extend(
            [
                observed.record_a,
                observed.record_b,
            ]
        )

    return records


def main():
    random.seed(
        42
    )

    records = (
        generate_observation_dataset(
            n_pairs=5,
        )
    )

    print(
        "=== Phase 0.5 observation dataset ==="
    )

    print(
        "rows=",
        len(records),
    )

    for record in records:
        print(
            record.pair_id,
            record.side,
            record.label,
            f"fx={record.fx:.4f}",
            f"fy={record.fy:.4f}",
            f"tip_dx={record.tip_dx:.6f}",
            f"tip_dy={record.tip_dy:.6f}",
            f"rho={record.rho:.4f}",
            f"margin={record.margin:.4f}",
        )


if __name__ == "__main__":
    main()