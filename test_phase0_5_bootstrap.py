import numpy as np

from phase0_5_bootstrap import (
    pair_cluster_bootstrap_auc,
)


def test_bootstrap_ci_stays_inside_auc_range():
    y_true = np.array(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ]
    )

    scores = np.array(
        [
            0.1,
            0.9,
            0.2,
            0.8,
            0.3,
            0.7,
            0.4,
            0.6,
        ]
    )

    pair_ids = [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]

    (
        ci_low,
        ci_high,
    ) = pair_cluster_bootstrap_auc(
        y_true=y_true,
        scores=scores,
        pair_ids=pair_ids,
        n_bootstrap=200,
        seed=42,
    )

    assert (
        0.0
        <= ci_low
        <= 1.0
    )

    assert (
        0.0
        <= ci_high
        <= 1.0
    )

    assert (
        ci_low
        <= ci_high
    )


def test_perfect_classifier_bootstrap_ci_is_perfect():
    y_true = np.array(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ]
    )

    scores = np.array(
        [
            0.1,
            0.9,
            0.2,
            0.8,
            0.3,
            0.7,
            0.4,
            0.6,
        ]
    )

    pair_ids = [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]

    (
        ci_low,
        ci_high,
    ) = pair_cluster_bootstrap_auc(
        y_true=y_true,
        scores=scores,
        pair_ids=pair_ids,
        n_bootstrap=200,
        seed=42,
    )

    assert (
        abs(
            ci_low
            - 1.0
        )
        < 1e-12
    )

    assert (
        abs(
            ci_high
            - 1.0
        )
        < 1e-12
    )


def test_bootstrap_is_reproducible_for_same_seed():
    y_true = np.array(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ]
    )

    scores = np.array(
        [
            0.2,
            0.8,
            0.4,
            0.6,
            0.7,
            0.3,
            0.1,
            0.9,
        ]
    )

    pair_ids = [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]

    result_a = (
        pair_cluster_bootstrap_auc(
            y_true=y_true,
            scores=scores,
            pair_ids=pair_ids,
            n_bootstrap=200,
            seed=123,
        )
    )

    result_b = (
        pair_cluster_bootstrap_auc(
            y_true=y_true,
            scores=scores,
            pair_ids=pair_ids,
            n_bootstrap=200,
            seed=123,
        )
    )

    assert (
        result_a
        == result_b
    )


def test_bootstrap_rejects_mismatched_lengths():
    try:
        pair_cluster_bootstrap_auc(
            y_true=np.array(
                [
                    0,
                    1,
                ]
            ),
            scores=np.array(
                [
                    0.2,
                ]
            ),
            pair_ids=[
                0,
                0,
            ],
            n_bootstrap=100,
        )
    except ValueError as error:
        assert (
            "equal length"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected mismatched lengths to fail"
        )


def test_bootstrap_rejects_nonpositive_count():
    try:
        pair_cluster_bootstrap_auc(
            y_true=np.array(
                [
                    0,
                    1,
                    0,
                    1,
                ]
            ),
            scores=np.array(
                [
                    0.1,
                    0.9,
                    0.2,
                    0.8,
                ]
            ),
            pair_ids=[
                0,
                0,
                1,
                1,
            ],
            n_bootstrap=0,
        )
    except ValueError as error:
        assert (
            "n_bootstrap"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected nonpositive bootstrap count to fail"
        )


def test_bootstrap_rejects_incomplete_pair_cluster():
    try:
        pair_cluster_bootstrap_auc(
            y_true=np.array(
                [
                    0,
                    1,
                    0,
                ]
            ),
            scores=np.array(
                [
                    0.1,
                    0.9,
                    0.2,
                ]
            ),
            pair_ids=[
                0,
                0,
                1,
            ],
            n_bootstrap=100,
        )
    except ValueError as error:
        assert (
            "exactly two rows"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected incomplete pair cluster to fail"
        )