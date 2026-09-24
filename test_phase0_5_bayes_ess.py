import math

import numpy as np

from phase0_5_bayes_ess import (
    ObservationESS,
    effective_sample_size_from_log_weights,
    summarize_ess,
)


def test_equal_weights_have_full_effective_sample_size():
    log_weights = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    ess = (
        effective_sample_size_from_log_weights(
            log_weights
        )
    )

    assert (
        abs(
            ess
            - 4.0
        )
        < 1e-12
    )


def test_single_dominant_weight_has_ess_near_one():
    log_weights = np.array(
        [
            0.0,
            -100.0,
            -100.0,
            -100.0,
        ]
    )

    ess = (
        effective_sample_size_from_log_weights(
            log_weights
        )
    )

    assert (
        abs(
            ess
            - 1.0
        )
        < 1e-10
    )


def test_ess_is_invariant_to_log_weight_offset():
    original = np.array(
        [
            -1.0,
            -2.0,
            -3.0,
            -4.0,
        ]
    )

    shifted = (
        original
        + 500.0
    )

    ess_original = (
        effective_sample_size_from_log_weights(
            original
        )
    )

    ess_shifted = (
        effective_sample_size_from_log_weights(
            shifted
        )
    )

    assert (
        abs(
            ess_original
            - ess_shifted
        )
        < 1e-12
    )


def test_ess_rejects_empty_array():
    try:
        effective_sample_size_from_log_weights(
            np.array(
                []
            )
        )
    except ValueError as error:
        assert (
            "must not be empty"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected empty log weights to fail"
        )


def test_ess_rejects_nonfinite_weights():
    try:
        effective_sample_size_from_log_weights(
            np.array(
                [
                    0.0,
                    math.inf,
                ]
            )
        )
    except ValueError as error:
        assert (
            "finite"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected nonfinite log weights to fail"
        )


def test_summary_reports_expected_statistics():
    observations = [
        ObservationESS(
            pair_id=0,
            side="A",
            slip_ess=10.0,
            yield_ess=20.0,
            slip_ess_fraction=0.10,
            yield_ess_fraction=0.20,
        ),
        ObservationESS(
            pair_id=0,
            side="B",
            slip_ess=30.0,
            yield_ess=40.0,
            slip_ess_fraction=0.30,
            yield_ess_fraction=0.40,
        ),
    ]

    summary = (
        summarize_ess(
            observations=observations,
            n_particles=200,
        )
    )

    assert (
        summary.n_particles
        == 200
    )

    assert (
        summary.n_particles_per_class
        == 100
    )

    assert (
        summary.n_test_rows
        == 2
    )

    assert (
        summary.slip_min
        == 10.0
    )

    assert (
        summary.slip_median
        == 20.0
    )

    assert (
        summary.yield_min
        == 20.0
    )

    assert (
        summary.yield_median
        == 30.0
    )


def test_summary_rejects_empty_observations():
    try:
        summarize_ess(
            observations=[],
            n_particles=200,
        )
    except ValueError as error:
        assert (
            "must not be empty"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected empty observations to fail"
        )


def test_summary_rejects_invalid_particle_count():
    observations = [
        ObservationESS(
            pair_id=0,
            side="A",
            slip_ess=10.0,
            yield_ess=10.0,
            slip_ess_fraction=0.1,
            yield_ess_fraction=0.1,
        )
    ]

    for invalid_count in [
        0,
        -2,
        101,
    ]:
        try:
            summarize_ess(
                observations=observations,
                n_particles=invalid_count,
            )
        except ValueError as error:
            assert (
                "positive and even"
                in str(error)
            )
        else:
            raise AssertionError(
                "Expected invalid particle count to fail"
            )