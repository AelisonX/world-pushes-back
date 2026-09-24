import random

import numpy as np

from phase0_5_controls import (
    CONTROL_FULL,
    CONTROL_SHAM,
    generate_matched_control_dataset,
)
from phase0_5_evaluate import (
    evaluate_control,
    fnr_at_max_fpr,
    labels_to_binary,
    records_for_pair_ids,
    train_and_score,
)
from phase0_5_split import (
    pair_level_train_test_split,
)
from physics import (
    NextTransition,
)


def test_labels_to_binary_uses_support_slip_as_positive():
    labels = [
        NextTransition.MATERIAL_YIELD.value,
        NextTransition.SUPPORT_SLIP.value,
    ]

    encoded = (
        labels_to_binary(
            labels
        )
    )

    assert (
        encoded.tolist()
        == [
            0,
            1,
        ]
    )


def test_labels_to_binary_rejects_unknown_label():
    try:
        labels_to_binary(
            [
                "MATERIAL_YIELD",
                "ALIEN_INVASION",
            ]
        )
    except ValueError as error:
        assert (
            "Unknown labels"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected unknown label to fail"
        )


def test_records_for_pair_ids_selects_only_requested_pairs():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=10,
        )
    )

    selected = (
        records_for_pair_ids(
            records=dataset.compliant_records,
            pair_ids={
                1,
                3,
                7,
            },
        )
    )

    assert {
        record.pair_id
        for record in selected
    } == {
        1,
        3,
        7,
    }

    assert (
        len(
            selected
        )
        == 6
    )


def test_fnr_at_max_fpr_is_zero_for_perfect_scores():
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    scores = np.array(
        [
            0.1,
            0.2,
            0.8,
            0.9,
        ]
    )

    fnr = (
        fnr_at_max_fpr(
            y_true=y_true,
            scores=scores,
            max_fpr=0.10,
        )
    )

    assert (
        abs(
            fnr
        )
        < 1e-12
    )


def test_invalid_max_fpr_is_rejected():
    y_true = np.array(
        [
            0,
            1,
        ]
    )

    scores = np.array(
        [
            0.1,
            0.9,
        ]
    )

    for invalid_value in [
        -0.1,
        1.0,
        1.1,
    ]:
        try:
            fnr_at_max_fpr(
                y_true=y_true,
                scores=scores,
                max_fpr=invalid_value,
            )
        except ValueError as error:
            assert (
                "max_fpr"
                in str(error)
            )
        else:
            raise AssertionError(
                "Expected invalid max_fpr to fail"
            )


def test_evaluation_preserves_pair_level_split():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=40,
        )
    )

    split = (
        pair_level_train_test_split(
            records=dataset.compliant_records,
            test_fraction=0.25,
            seed=42,
        )
    )

    result = (
        evaluate_control(
            train_records=split.train,
            test_records=split.test,
            control=CONTROL_FULL,
            seed=42,
        )
    )

    assert (
        result.train_pairs
        == 30
    )

    assert (
        result.test_pairs
        == 10
    )

    assert (
        result.train_rows
        == 60
    )

    assert (
        result.test_rows
        == 20
    )


def test_auc_stays_inside_valid_range():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=60,
        )
    )

    split = (
        pair_level_train_test_split(
            records=dataset.compliant_records,
            test_fraction=0.25,
            seed=42,
        )
    )

    result = (
        evaluate_control(
            train_records=split.train,
            test_records=split.test,
            control=CONTROL_FULL,
            seed=42,
        )
    )

    assert (
        0.0
        <= result.roc_auc
        <= 1.0
    )


def test_fnr_stays_inside_valid_range():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=60,
        )
    )

    split = (
        pair_level_train_test_split(
            records=dataset.compliant_records,
            test_fraction=0.25,
            seed=42,
        )
    )

    result = (
        evaluate_control(
            train_records=split.train,
            test_records=split.test,
            control=CONTROL_FULL,
            seed=42,
        )
    )

    assert (
        0.0
        <= result.fnr_at_fpr_10
        <= 1.0
    )


def test_sham_auc_is_chance_for_balanced_test_set():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=80,
        )
    )

    split = (
        pair_level_train_test_split(
            records=dataset.compliant_records,
            test_fraction=0.25,
            seed=42,
        )
    )

    result = (
        evaluate_control(
            train_records=split.train,
            test_records=split.test,
            control=CONTROL_SHAM,
            seed=42,
        )
    )

    assert (
        abs(
            result.roc_auc
            - 0.5
        )
        < 1e-12
    )


def test_train_and_score_rejects_single_class_training_set():
    from phase0_5_controls import (
        FeatureView,
    )

    train_view = FeatureView(
        name="BROKEN",
        features=[
            [
                0.0
            ],
            [
                1.0
            ],
        ],
        labels=[
            NextTransition.MATERIAL_YIELD.value,
            NextTransition.MATERIAL_YIELD.value,
        ],
        pair_ids=[
            1,
            2,
        ],
        sides=[
            "A",
            "A",
        ],
    )

    test_view = FeatureView(
        name="TEST",
        features=[
            [
                0.0
            ],
            [
                1.0
            ],
        ],
        labels=[
            NextTransition.MATERIAL_YIELD.value,
            NextTransition.SUPPORT_SLIP.value,
        ],
        pair_ids=[
            3,
            3,
        ],
        sides=[
            "A",
            "B",
        ],
    )

    try:
        train_and_score(
            train_view=train_view,
            test_view=test_view,
        )
    except ValueError as error:
        assert (
            "Training set"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected single-class training set to fail"
        )