from dataclasses import dataclass
import random

from phase0_5_dataset import (
    ObservationRecord,
)


@dataclass
class DatasetSplit:
    train: list[ObservationRecord]
    test: list[ObservationRecord]
    train_pair_ids: set[int]
    test_pair_ids: set[int]


def group_records_by_pair(
    records: list[ObservationRecord],
) -> dict[
    int,
    list[ObservationRecord],
]:
    """
    Group dataset rows by pair_id.
    """

    grouped = {}

    for record in records:
        grouped.setdefault(
            record.pair_id,
            [],
        ).append(
            record
        )

    return grouped


def validate_pair_integrity(
    records: list[ObservationRecord],
) -> None:
    """
    Verify that every pair contributes exactly two rows
    with opposite A/B sides.
    """

    grouped = (
        group_records_by_pair(
            records
        )
    )

    for (
        pair_id,
        pair_records,
    ) in grouped.items():
        if len(
            pair_records
        ) != 2:
            raise ValueError(
                f"pair_id={pair_id} "
                "must contain exactly two rows"
            )

        sides = {
            record.side
            for record in pair_records
        }

        if sides != {
            "A",
            "B",
        }:
            raise ValueError(
                f"pair_id={pair_id} "
                "must contain exactly sides A and B"
            )

        labels = {
            record.label
            for record in pair_records
        }

        if len(
            labels
        ) != 2:
            raise ValueError(
                f"pair_id={pair_id} "
                "must contain opposite labels"
            )


def pair_level_train_test_split(
    records: list[ObservationRecord],
    test_fraction: float = 0.25,
    seed: int = 42,
) -> DatasetSplit:
    """
    Split a paired dataset by pair_id.

    Both twins from one pair always remain in the same split.
    """

    if not (
        0.0
        < test_fraction
        < 1.0
    ):
        raise ValueError(
            "test_fraction must be between 0 and 1"
        )

    validate_pair_integrity(
        records
    )

    grouped = (
        group_records_by_pair(
            records
        )
    )

    pair_ids = list(
        grouped.keys()
    )

    if len(
        pair_ids
    ) < 2:
        raise ValueError(
            "At least two pairs are required for a train/test split"
        )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        pair_ids
    )

    n_test_pairs = max(
        1,
        round(
            len(pair_ids)
            * test_fraction
        ),
    )

    n_test_pairs = min(
        n_test_pairs,
        len(pair_ids) - 1,
    )

    test_pair_ids = set(
        pair_ids[
            :n_test_pairs
        ]
    )

    train_pair_ids = set(
        pair_ids[
            n_test_pairs:
        ]
    )

    train = []

    test = []

    for pair_id in train_pair_ids:
        train.extend(
            grouped[
                pair_id
            ]
        )

    for pair_id in test_pair_ids:
        test.extend(
            grouped[
                pair_id
            ]
        )

    return DatasetSplit(
        train=train,
        test=test,
        train_pair_ids=train_pair_ids,
        test_pair_ids=test_pair_ids,
    )


def main():
    from phase0_5_dataset import (
        generate_observation_dataset,
    )

    random.seed(
        42
    )

    records = (
        generate_observation_dataset(
            n_pairs=20,
        )
    )

    split = (
        pair_level_train_test_split(
            records=records,
            test_fraction=0.25,
            seed=42,
        )
    )

    print(
        "=== Phase 0.5 pair-level split ==="
    )

    print(
        "train_pairs=",
        len(
            split.train_pair_ids
        ),
    )

    print(
        "test_pairs=",
        len(
            split.test_pair_ids
        ),
    )

    print(
        "train_rows=",
        len(
            split.train
        ),
    )

    print(
        "test_rows=",
        len(
            split.test
        ),
    )

    print(
        "overlap=",
        split.train_pair_ids
        & split.test_pair_ids,
    )


if __name__ == "__main__":
    main()