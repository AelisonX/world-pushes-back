from dataclasses import dataclass
import random

from phase0_5_dataset import (
    ObservationNoise,
    ObservationRecord,
    add_observation_noise,
    sample_shared_noise,
)
from phase0_5_safe_cohort import (
    SafePairSample,
    generate_safe_pair_sample,
)
from physics import (
    run_safe_probe,
)


CONTROL_FULL = "FULL"
CONTROL_FORCE_ONLY = "FORCE_ONLY"
CONTROL_MOTION_ONLY = "MOTION_ONLY"
CONTROL_SHAM = "SHAM"
CONTROL_LABEL_SHUFFLE = "LABEL_SHUFFLE"
CONTROL_RHO_CANARY = "RHO_CANARY"
CONTROL_RIGID_NULL = "RIGID_NULL"


@dataclass
class FeatureView:
    name: str
    features: list[list[float]]
    labels: list[str]
    pair_ids: list[int]
    sides: list[str]


@dataclass
class MatchedControlDataset:
    compliant_records: list[ObservationRecord]
    rigid_records: list[ObservationRecord]


def observe_pair_with_support_model(
    sample: SafePairSample,
    noise: ObservationNoise,
    support_model: str,
) -> tuple[
    ObservationRecord,
    ObservationRecord,
]:
    """
    Observe one safe matched pair under a selected support model.

    The same action and the same explicit measurement-noise
    realization are used for both twins.
    """

    if support_model not in {
        "rigid",
        "compliant",
    }:
        raise ValueError(
            "support_model must be "
            "'rigid' or 'compliant'"
        )

    result_a = run_safe_probe(
        params=sample.pair.world_a,
        action=sample.action,
        support_model=support_model,
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    result_b = run_safe_probe(
        params=sample.pair.world_b,
        action=sample.action,
        support_model=support_model,
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    if not (
        result_a.still_blocked
        and result_b.still_blocked
    ):
        raise RuntimeError(
            "Safe cohort invariant violated"
        )

    noisy_a = add_observation_noise(
        result_a.observation,
        noise,
    )

    noisy_b = add_observation_noise(
        result_b.observation,
        noise,
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

    return (
        record_a,
        record_b,
    )


def generate_matched_control_dataset(
    n_pairs: int,
) -> MatchedControlDataset:
    """
    Generate compliant and rigid observations from the same
    worlds, actions, and measurement-noise realizations.

    This makes the RIGID_NULL comparison matched rather than
    comparing unrelated randomly generated datasets.
    """

    if n_pairs <= 0:
        raise ValueError(
            "n_pairs must be positive"
        )

    compliant_records = []
    rigid_records = []

    for pair_id in range(
        n_pairs
    ):
        sample = generate_safe_pair_sample(
            pair_id=pair_id,
        )

        noise = sample_shared_noise()

        (
            compliant_a,
            compliant_b,
        ) = observe_pair_with_support_model(
            sample=sample,
            noise=noise,
            support_model="compliant",
        )

        (
            rigid_a,
            rigid_b,
        ) = observe_pair_with_support_model(
            sample=sample,
            noise=noise,
            support_model="rigid",
        )

        compliant_records.extend(
            [
                compliant_a,
                compliant_b,
            ]
        )

        rigid_records.extend(
            [
                rigid_a,
                rigid_b,
            ]
        )

    return MatchedControlDataset(
        compliant_records=compliant_records,
        rigid_records=rigid_records,
    )


def pair_preserving_shuffled_labels(
    records: list[ObservationRecord],
    seed: int = 42,
) -> list[str]:
    """
    Randomize labels while preserving one label of each type
    inside every matched pair.

    Each pair independently either keeps or swaps its two labels.

    This destroys the physical feature-label mapping without
    destroying pair balance.
    """

    grouped = {}

    for record in records:
        grouped.setdefault(
            record.pair_id,
            [],
        ).append(
            record
        )

    rng = random.Random(
        seed
    )

    shuffled_by_key = {}

    for (
        pair_id,
        pair_records,
    ) in grouped.items():
        if len(
            pair_records
        ) != 2:
            raise ValueError(
                f"pair_id={pair_id} "
                "must contain exactly two records"
            )

        labels = [
            pair_records[0].label,
            pair_records[1].label,
        ]

        if rng.random() < 0.5:
            labels.reverse()

        for (
            record,
            label,
        ) in zip(
            pair_records,
            labels,
        ):
            shuffled_by_key[
                (
                    record.pair_id,
                    record.side,
                )
            ] = label

    return [
        shuffled_by_key[
            (
                record.pair_id,
                record.side,
            )
        ]
        for record in records
    ]


def make_feature_view(
    records: list[ObservationRecord],
    control: str,
    seed: int = 42,
) -> FeatureView:
    """
    Convert observation records into one control-specific
    feature representation.
    """

    supported = {
        CONTROL_FULL,
        CONTROL_FORCE_ONLY,
        CONTROL_MOTION_ONLY,
        CONTROL_SHAM,
        CONTROL_LABEL_SHUFFLE,
        CONTROL_RHO_CANARY,
        CONTROL_RIGID_NULL,
    }

    if control not in supported:
        raise ValueError(
            f"Unknown control: {control}"
        )

    features = []

    for record in records:
        if control in {
            CONTROL_FULL,
            CONTROL_LABEL_SHUFFLE,
            CONTROL_RIGID_NULL,
        }:
            row = [
                record.fx,
                record.fy,
                record.tip_dx,
                record.tip_dy,
            ]

        elif control == CONTROL_FORCE_ONLY:
            row = [
                record.fx,
                record.fy,
            ]

        elif control == CONTROL_MOTION_ONLY:
            row = [
                record.tip_dx,
                record.tip_dy,
            ]

        elif control == CONTROL_SHAM:
            row = [
                0.0,
            ]

        elif control == CONTROL_RHO_CANARY:
            row = [
                record.rho,
            ]

        features.append(
            row
        )

    if control == CONTROL_LABEL_SHUFFLE:
        labels = (
            pair_preserving_shuffled_labels(
                records=records,
                seed=seed,
            )
        )
    else:
        labels = [
            record.label
            for record in records
        ]

    return FeatureView(
        name=control,
        features=features,
        labels=labels,
        pair_ids=[
            record.pair_id
            for record in records
        ],
        sides=[
            record.side
            for record in records
        ],
    )


def main():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=10,
        )
    )

    print(
        "=== Phase 0.5 control dataset ==="
    )

    print(
        "compliant_rows=",
        len(
            dataset.compliant_records
        ),
    )

    print(
        "rigid_rows=",
        len(
            dataset.rigid_records
        ),
    )

    for control in [
        CONTROL_FULL,
        CONTROL_FORCE_ONLY,
        CONTROL_MOTION_ONLY,
        CONTROL_SHAM,
        CONTROL_LABEL_SHUFFLE,
        CONTROL_RHO_CANARY,
    ]:
        view = make_feature_view(
            records=(
                dataset.compliant_records
            ),
            control=control,
        )

        print(
            control,
            "rows=",
            len(
                view.features
            ),
            "features_per_row=",
            len(
                view.features[0]
            ),
        )

    rigid_view = make_feature_view(
        records=dataset.rigid_records,
        control=CONTROL_RIGID_NULL,
    )

    print(
        CONTROL_RIGID_NULL,
        "rows=",
        len(
            rigid_view.features
        ),
        "features_per_row=",
        len(
            rigid_view.features[0]
        ),
    )


if __name__ == "__main__":
    main()