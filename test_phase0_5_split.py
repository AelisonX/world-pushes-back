import random

from phase0_5_dataset import (
    ObservationRecord,
    generate_observation_dataset,
)
from phase0_5_split import (
    group_records_by_pair,
    pair_level_train_test_split,
    validate_pair_integrity,
)


def test_group_records_by_pair_keeps_two_rows_per_pair():
    random.seed(
        42
    )

    records = (
        generate_observation_dataset(
            n_pairs=10,
        )
    )

    grouped = (
        group_records_by_pair(
            records
        )
    )

    assert (
        len(
            grouped
        )
        == 10
    )

    for pair_records in grouped.values():
        assert (
            len(
                pair_records
            )
            == 2
        )


def test_pair_level_split_has_no_pair_overlap():
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

    assert (
        split.train_pair_ids
        .isdisjoint(
            split.test_pair_ids
        )
    )


def test_each_pair_stays_whole_inside_one_split():
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

    train_counts = {}

    for record in split.train:
        train_counts[
            record.pair_id
        ] = (
            train_counts.get(
                record.pair_id,
                0,
            )
            + 1
        )

    test_counts = {}

    for record in split.test:
        test_counts[
            record.pair_id
        ] = (
            test_counts.get(
                record.pair_id,
                0,
            )
            + 1
        )

    assert (
        set(
            train_counts.values()
        )
        == {
            2
        }
    )

    assert (
        set(
            test_counts.values()
        )
        == {
            2
        }
    )


def test_split_preserves_all_rows():
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

    assert (
        len(
            split.train
        )
        + len(
            split.test
        )
        == len(
            records
        )
    )


def test_split_is_reproducible_for_same_seed():
    random.seed(
        42
    )

    records = (
        generate_observation_dataset(
            n_pairs=20,
        )
    )

    split_a = (
        pair_level_train_test_split(
            records=records,
            test_fraction=0.25,
            seed=123,
        )
    )

    split_b = (
        pair_level_train_test_split(
            records=records,
            test_fraction=0.25,
            seed=123,
        )
    )

    assert (
        split_a.train_pair_ids
        == split_b.train_pair_ids
    )

    assert (
        split_a.test_pair_ids
        == split_b.test_pair_ids
    )


def test_different_seeds_can_produce_different_pair_assignments():
    random.seed(
        42
    )

    records = (
        generate_observation_dataset(
            n_pairs=30,
        )
    )

    split_a = (
        pair_level_train_test_split(
            records=records,
            test_fraction=0.25,
            seed=1,
        )
    )

    split_b = (
        pair_level_train_test_split(
            records=records,
            test_fraction=0.25,
            seed=2,
        )
    )

    assert (
        split_a.test_pair_ids
        != split_b.test_pair_ids
    )


def test_invalid_test_fraction_is_rejected():
    random.seed(
        42
    )

    records = (
        generate_observation_dataset(
            n_pairs=10,
        )
    )

    for invalid_value in [
        0.0,
        1.0,
        -0.1,
        1.1,
    ]:
        try:
            pair_level_train_test_split(
                records=records,
                test_fraction=invalid_value,
            )
        except ValueError as error:
            assert (
                "test_fraction"
                in str(error)
            )
        else:
            raise AssertionError(
                "Expected invalid test_fraction to fail"
            )


def test_split_rejects_single_pair_dataset():
    random.seed(
        42
    )

    records = (
        generate_observation_dataset(
            n_pairs=1,
        )
    )

    try:
        pair_level_train_test_split(
            records=records,
        )
    except ValueError as error:
        assert (
            "At least two pairs"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected single-pair dataset to fail"
        )


def test_integrity_check_rejects_missing_twin():
    broken_records = [
        ObservationRecord(
            pair_id=7,
            side="A",
            label="MATERIAL_YIELD",
            fx=1.0,
            fy=1.0,
            tip_dx=0.001,
            tip_dy=-0.001,
            rho=0.2,
            margin=0.3,
        )
    ]

    try:
        validate_pair_integrity(
            broken_records
        )
    except ValueError as error:
        assert (
            "exactly two rows"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected incomplete pair to fail"
        )


def test_integrity_check_rejects_duplicate_side():
    broken_records = [
        ObservationRecord(
            pair_id=8,
            side="A",
            label="MATERIAL_YIELD",
            fx=1.0,
            fy=1.0,
            tip_dx=0.001,
            tip_dy=-0.001,
            rho=0.2,
            margin=0.3,
        ),
        ObservationRecord(
            pair_id=8,
            side="A",
            label="SUPPORT_SLIP",
            fx=1.0,
            fy=1.0,
            tip_dx=0.001,
            tip_dy=-0.001,
            rho=0.3,
            margin=0.4,
        ),
    ]

    try:
        validate_pair_integrity(
            broken_records
        )
    except ValueError as error:
        assert (
            "sides A and B"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected duplicate side to fail"
        )


def test_integrity_check_rejects_same_label_twins():
    broken_records = [
        ObservationRecord(
            pair_id=9,
            side="A",
            label="MATERIAL_YIELD",
            fx=1.0,
            fy=1.0,
            tip_dx=0.001,
            tip_dy=-0.001,
            rho=0.2,
            margin=0.3,
        ),
        ObservationRecord(
            pair_id=9,
            side="B",
            label="MATERIAL_YIELD",
            fx=1.0,
            fy=1.0,
            tip_dx=0.001,
            tip_dy=-0.001,
            rho=0.3,
            margin=0.4,
        ),
    ]

    try:
        validate_pair_integrity(
            broken_records
        )
    except ValueError as error:
        assert (
            "opposite labels"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected same-label twins to fail"
        )