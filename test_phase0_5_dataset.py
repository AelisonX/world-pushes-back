import random

from phase0_5_dataset import (
    ObservationNoise,
    add_observation_noise,
    generate_observation_dataset,
    observe_safe_pair,
    sample_shared_noise,
)
from phase0_5_safe_cohort import (
    generate_safe_pair_sample,
)
from physics import (
    NextTransition,
    Observation,
)


def test_shared_noise_can_be_applied_exactly():
    observation = Observation(
        fx=1.0,
        fy=2.0,
        tip_dx=0.003,
        tip_dy=-0.004,
    )

    noise = ObservationNoise(
        fx=0.1,
        fy=-0.2,
        tip_dx=0.0003,
        tip_dy=-0.0004,
    )

    noisy = (
        add_observation_noise(
            observation,
            noise,
        )
    )

    assert (
        abs(
            noisy.fx
            - 1.1
        )
        < 1e-12
    )

    assert (
        abs(
            noisy.fy
            - 1.8
        )
        < 1e-12
    )

    assert (
        abs(
            noisy.tip_dx
            - 0.0033
        )
        < 1e-12
    )

    assert (
        abs(
            noisy.tip_dy
            + 0.0044
        )
        < 1e-12
    )


def test_observed_pair_keeps_same_pair_id():
    random.seed(
        42
    )

    sample = (
        generate_safe_pair_sample(
            pair_id=123,
        )
    )

    observed = (
        observe_safe_pair(
            sample=sample,
            noise=ObservationNoise(
                fx=0.0,
                fy=0.0,
                tip_dx=0.0,
                tip_dy=0.0,
            ),
        )
    )

    assert (
        observed.record_a.pair_id
        == 123
    )

    assert (
        observed.record_b.pair_id
        == 123
    )


def test_observed_pair_has_opposite_labels():
    random.seed(
        42
    )

    sample = (
        generate_safe_pair_sample(
            pair_id=1,
        )
    )

    observed = (
        observe_safe_pair(
            sample=sample,
            noise=ObservationNoise(
                fx=0.0,
                fy=0.0,
                tip_dx=0.0,
                tip_dy=0.0,
            ),
        )
    )

    labels = {
        observed.record_a.label,
        observed.record_b.label,
    }

    assert labels == {
        NextTransition.MATERIAL_YIELD.value,
        NextTransition.SUPPORT_SLIP.value,
    }


def test_twins_receive_identical_force_noise_offset():
    random.seed(
        42
    )

    sample = (
        generate_safe_pair_sample(
            pair_id=2,
        )
    )

    zero_noise = ObservationNoise(
        fx=0.0,
        fy=0.0,
        tip_dx=0.0,
        tip_dy=0.0,
    )

    known_noise = ObservationNoise(
        fx=0.37,
        fy=-0.19,
        tip_dx=0.00031,
        tip_dy=-0.00017,
    )

    clean = (
        observe_safe_pair(
            sample=sample,
            noise=zero_noise,
        )
    )

    noisy = (
        observe_safe_pair(
            sample=sample,
            noise=known_noise,
        )
    )

    delta_a_fx = (
        noisy.record_a.fx
        - clean.record_a.fx
    )

    delta_b_fx = (
        noisy.record_b.fx
        - clean.record_b.fx
    )

    delta_a_fy = (
        noisy.record_a.fy
        - clean.record_a.fy
    )

    delta_b_fy = (
        noisy.record_b.fy
        - clean.record_b.fy
    )

    assert (
        abs(
            delta_a_fx
            - known_noise.fx
        )
        < 1e-12
    )

    assert (
        abs(
            delta_b_fx
            - known_noise.fx
        )
        < 1e-12
    )

    assert (
        abs(
            delta_a_fy
            - delta_b_fy
        )
        < 1e-12
    )


def test_twins_receive_identical_motion_noise_offset():
    random.seed(
        42
    )

    sample = (
        generate_safe_pair_sample(
            pair_id=3,
        )
    )

    zero_noise = ObservationNoise(
        fx=0.0,
        fy=0.0,
        tip_dx=0.0,
        tip_dy=0.0,
    )

    known_noise = ObservationNoise(
        fx=0.0,
        fy=0.0,
        tip_dx=0.00045,
        tip_dy=-0.00023,
    )

    clean = (
        observe_safe_pair(
            sample=sample,
            noise=zero_noise,
        )
    )

    noisy = (
        observe_safe_pair(
            sample=sample,
            noise=known_noise,
        )
    )

    delta_a_dx = (
        noisy.record_a.tip_dx
        - clean.record_a.tip_dx
    )

    delta_b_dx = (
        noisy.record_b.tip_dx
        - clean.record_b.tip_dx
    )

    delta_a_dy = (
        noisy.record_a.tip_dy
        - clean.record_a.tip_dy
    )

    delta_b_dy = (
        noisy.record_b.tip_dy
        - clean.record_b.tip_dy
    )

    assert (
        abs(
            delta_a_dx
            - known_noise.tip_dx
        )
        < 1e-12
    )

    assert (
        abs(
            delta_b_dx
            - known_noise.tip_dx
        )
        < 1e-12
    )

    assert (
        abs(
            delta_a_dy
            - delta_b_dy
        )
        < 1e-12
    )


def test_dataset_has_exactly_two_rows_per_pair():
    random.seed(
        42
    )

    records = (
        generate_observation_dataset(
            n_pairs=10,
        )
    )

    assert (
        len(records)
        == 20
    )

    counts = {}

    for record in records:
        counts[
            record.pair_id
        ] = (
            counts.get(
                record.pair_id,
                0,
            )
            + 1
        )

    assert (
        set(
            counts.values()
        )
        == {
            2
        }
    )


def test_dataset_pair_ids_are_complete():
    random.seed(
        42
    )

    records = (
        generate_observation_dataset(
            n_pairs=10,
        )
    )

    pair_ids = {
        record.pair_id
        for record in records
    }

    assert (
        pair_ids
        == set(
            range(
                10
            )
        )
    )


def test_dataset_contains_balanced_labels_by_construction():
    random.seed(
        42
    )

    records = (
        generate_observation_dataset(
            n_pairs=20,
        )
    )

    labels = [
        record.label
        for record in records
    ]

    assert (
        labels.count(
            NextTransition.MATERIAL_YIELD.value
        )
        == 20
    )

    assert (
        labels.count(
            NextTransition.SUPPORT_SLIP.value
        )
        == 20
    )


def test_sample_shared_noise_rejects_negative_std():
    try:
        sample_shared_noise(
            force_noise_std=-0.1,
        )
    except ValueError as error:
        assert (
            "force_noise_std"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected negative force noise std to fail"
        )

    try:
        sample_shared_noise(
            motion_noise_std=-0.1,
        )
    except ValueError as error:
        assert (
            "motion_noise_std"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected negative motion noise std to fail"
        )


def test_dataset_rejects_nonpositive_pair_count():
    try:
        generate_observation_dataset(
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