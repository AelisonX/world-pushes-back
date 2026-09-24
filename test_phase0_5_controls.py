import random

from phase0_5_controls import (
    CONTROL_FORCE_ONLY,
    CONTROL_FULL,
    CONTROL_LABEL_SHUFFLE,
    CONTROL_MOTION_ONLY,
    CONTROL_RHO_CANARY,
    CONTROL_SHAM,
    generate_matched_control_dataset,
    make_feature_view,
    pair_preserving_shuffled_labels,
)


def test_matched_control_dataset_has_same_number_of_rows():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=10,
        )
    )

    assert (
        len(
            dataset.compliant_records
        )
        == 20
    )

    assert (
        len(
            dataset.rigid_records
        )
        == 20
    )


def test_rigid_and_compliant_records_match_pair_identity():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=10,
        )
    )

    compliant_keys = [
        (
            record.pair_id,
            record.side,
            record.label,
        )
        for record
        in dataset.compliant_records
    ]

    rigid_keys = [
        (
            record.pair_id,
            record.side,
            record.label,
        )
        for record
        in dataset.rigid_records
    ]

    assert (
        compliant_keys
        == rigid_keys
    )


def test_force_observations_match_between_support_models():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=10,
        )
    )

    for (
        compliant,
        rigid,
    ) in zip(
        dataset.compliant_records,
        dataset.rigid_records,
    ):
        assert (
            abs(
                compliant.fx
                - rigid.fx
            )
            < 1e-12
        )

        assert (
            abs(
                compliant.fy
                - rigid.fy
            )
            < 1e-12
        )


def test_rigid_and_compliant_can_differ_in_horizontal_motion():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=20,
        )
    )

    differences = [
        abs(
            compliant.tip_dx
            - rigid.tip_dx
        )
        for (
            compliant,
            rigid,
        ) in zip(
            dataset.compliant_records,
            dataset.rigid_records,
        )
    ]

    assert (
        max(
            differences
        )
        > 0.0
    )


def test_full_view_has_four_features():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=5,
        )
    )

    view = make_feature_view(
        records=dataset.compliant_records,
        control=CONTROL_FULL,
    )

    assert all(
        len(row)
        == 4
        for row in view.features
    )


def test_force_only_view_has_two_features():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=5,
        )
    )

    view = make_feature_view(
        records=dataset.compliant_records,
        control=CONTROL_FORCE_ONLY,
    )

    assert all(
        len(row)
        == 2
        for row in view.features
    )


def test_motion_only_view_has_two_features():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=5,
        )
    )

    view = make_feature_view(
        records=dataset.compliant_records,
        control=CONTROL_MOTION_ONLY,
    )

    assert all(
        len(row)
        == 2
        for row in view.features
    )


def test_sham_view_contains_only_zero():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=5,
        )
    )

    view = make_feature_view(
        records=dataset.compliant_records,
        control=CONTROL_SHAM,
    )

    assert all(
        row
        == [
            0.0
        ]
        for row in view.features
    )


def test_rho_canary_contains_only_rho():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=5,
        )
    )

    view = make_feature_view(
        records=dataset.compliant_records,
        control=CONTROL_RHO_CANARY,
    )

    for (
        row,
        record,
    ) in zip(
        view.features,
        dataset.compliant_records,
    ):
        assert (
            row
            == [
                record.rho
            ]
        )


def test_label_shuffle_preserves_pair_balance():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=20,
        )
    )

    records = (
        dataset.compliant_records
    )

    shuffled = (
        pair_preserving_shuffled_labels(
            records=records,
            seed=123,
        )
    )

    grouped = {}

    for (
        record,
        label,
    ) in zip(
        records,
        shuffled,
    ):
        grouped.setdefault(
            record.pair_id,
            [],
        ).append(
            label
        )

    for labels in grouped.values():
        assert (
            len(
                set(
                    labels
                )
            )
            == 2
        )


def test_label_shuffle_changes_some_assignments():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=50,
        )
    )

    records = (
        dataset.compliant_records
    )

    original = [
        record.label
        for record in records
    ]

    shuffled = (
        pair_preserving_shuffled_labels(
            records=records,
            seed=123,
        )
    )

    assert (
        original
        != shuffled
    )


def test_label_shuffle_is_reproducible():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=20,
        )
    )

    records = (
        dataset.compliant_records
    )

    shuffled_a = (
        pair_preserving_shuffled_labels(
            records=records,
            seed=999,
        )
    )

    shuffled_b = (
        pair_preserving_shuffled_labels(
            records=records,
            seed=999,
        )
    )

    assert (
        shuffled_a
        == shuffled_b
    )


def test_unknown_control_is_rejected():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=5,
        )
    )

    try:
        make_feature_view(
            records=dataset.compliant_records,
            control="TOTALLY_FAKE_CONTROL",
        )
    except ValueError as error:
        assert (
            "Unknown control"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected unknown control to fail"
        )


def test_nonpositive_pair_count_is_rejected():
    try:
        generate_matched_control_dataset(
            n_pairs=0,
        )
    except ValueError as error:
        assert (
            "n_pairs"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected nonpositive pair count to fail"
        )